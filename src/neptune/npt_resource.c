/*
 * Copyright 2026 Turing Software LLC
 * SPDX-License-Identifier: MIT
 *
 * D3D11 resource manipulation for the RESOURCE_{UPDATE, MAP, UNMAP}
 * transport commands.  Resolves host resources via the object table,
 * walks the COM vtable for D3D11 Map/Unmap/Update, and (for MAP)
 * stashes sync-map state per (resource, subresource) so the matching
 * UNMAP can write back.
 */

#include "npt_resource.h"

#include <stdio.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <sys/stat.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "npt_com.h"
#include "npt_context.h"
#include "npt_hostmem.h"
#include "npt_library.h"
#include "npt_transport_defs.h"

#include "neptune-protocol/npt_protocol_defs.h"
#include "neptune-protocol/npt_protocol_host_dispatch_types.h"
#include "neptune-protocol/npt_protocol_host_id3d12resource.h"

/* ---- buffer trace (R8_UINT glyph buffers): host side of the guest "FMT guest buf-*" lines.
 * Buffers with BIND_SHADER_RESOURCE (not CONSTANT_BUFFER); 400 lines max. */
static volatile int npt_fmt_buf_count;

static bool
npt_trace_buf_wanted(struct npt_context *ctx, uint64_t resource_id, void *resource)
{
   if (npt_fmt_buf_count >= 400) return false;
   if (!npt_context_object_is(ctx, resource_id, NPT_OBJECT_TYPE_ID3D11BUFFER)) return false;
   PFN_ID3D11Buffer_GetDesc gd = NPT_COM_VTBL_FUNC(PFN_ID3D11Buffer_GetDesc, npt_com_vtable(resource),
                                                   NPT_VTBL_ID3D11Buffer_GetDesc);
   if (!gd) return false;
   D3D11_BUFFER_DESC d; gd(resource, &d);
   if (!(d.BindFlags & D3D11_BIND_SHADER_RESOURCE) || (d.BindFlags & D3D11_BIND_CONSTANT_BUFFER)) return false;
   __atomic_fetch_add(&npt_fmt_buf_count, 1, __ATOMIC_RELAXED);
   return true;
}

static volatile int npt_fmt_vb_count;

static bool
npt_trace_vb_wanted(struct npt_context *ctx, uint64_t resource_id, void *resource)
{
   if (npt_fmt_vb_count >= 300) return false;
   if (!npt_context_object_is(ctx, resource_id, NPT_OBJECT_TYPE_ID3D11BUFFER)) return false;
   PFN_ID3D11Buffer_GetDesc gd = NPT_COM_VTBL_FUNC(PFN_ID3D11Buffer_GetDesc, npt_com_vtable(resource), NPT_VTBL_ID3D11Buffer_GetDesc);
   if (!gd) return false;
   D3D11_BUFFER_DESC d; gd(resource, &d);
   if (!(d.BindFlags & D3D11_BIND_VERTEX_BUFFER) || d.ByteWidth < 16384 || d.ByteWidth > (4u << 20)) return false;
   __atomic_fetch_add(&npt_fmt_vb_count, 1, __ATOMIC_RELAXED);
   return true;
}

static void
npt_trace_buf_sum(const uint8_t *p, uint64_t n, uint64_t *fnv, uint32_t *nz, uint32_t *last, char *hex)
{
   uint64_t h = 1469598103934665603ull; uint32_t z = 0, l = 0;
   for (uint64_t i = 0; p && i < n; i++) { h ^= p[i]; h *= 1099511628211ull; if (p[i]) { z++; l = (uint32_t)i; } }
   hex[0] = 0;
   for (uint64_t i = 0; p && i < 16 && i < n; i++) snprintf(hex + i * 2, 3, "%02x", p[i]);
   *fnv = h; *nz = z; *last = l;
}

static void
npt_trace_buf_line(const char *tag, uint64_t resource_id, uint32_t access, uint64_t n,
                   const uint8_t *shm, const uint8_t *mapped)
{
   uint64_t fs, fm; uint32_t zs, zm, ls, lm; char hs[40], hm[40];
   npt_trace_buf_sum(shm, n, &fs, &zs, &ls, hs);
   npt_trace_buf_sum(mapped, n, &fm, &zm, &lm, hm);
   npt_log("FMT host buf-%s res=0x%" PRIx64 " access=0x%x n=%" PRIu64 " shmem_fnv=%016" PRIx64
           " nonzero=%u lastNonzero=%u first16=%s | %s_fnv=%016" PRIx64 " nonzero=%u lastNonzero=%u %s",
           tag, resource_id, access, n, fs, zs, ls, hs, mapped ? "mapped" : "-", fm, zm, lm,
           mapped ? (fs == fm ? "MATCH" : "MISMATCH") : "");
}


/* C19 diagnostics: identities are registered NPT ids, never fd/inode guesses. */
static bool npt_caption_trace_configured(void)
{
   const char *enabled = getenv("NPT_SHARED_GUARD_TRACE");
   const char *marker = getenv("NPT_CAPTION_TRACE_FILE");
   return enabled && !strcmp(enabled, "1") && marker && marker[0];
}

bool npt_caption_trace_enabled(void)
{
   static unsigned activated;
   if (!npt_caption_trace_configured()) return false; /* no FS check */
   if (__atomic_load_n(&activated, __ATOMIC_ACQUIRE)) return true;
   const char *marker = getenv("NPT_CAPTION_TRACE_FILE");
   if (access(marker, F_OK)) return false;
   /* One fresh window per process. Removing/replacing the marker never
    * resets a budget; after activation no operation stats the filesystem. */
   __atomic_store_n(&activated, 1, __ATOMIC_RELEASE);
   return true;
}

static unsigned npt_caption_take(unsigned *counter, unsigned limit)
{
   unsigned old = __atomic_load_n(counter, __ATOMIC_RELAXED);
   while (old < limit) {
      if (__atomic_compare_exchange_n(counter, &old, old + 1, false,
                                       __ATOMIC_RELAXED, __ATOMIC_RELAXED)) return old + 1;
   }
   return false;
}


/* C25 CPU-only PS bytes: a separate startup opt-in from the late C24 draw
 * marker. No shader-cache setting or GPU operation is changed. Attempts,
 * including failed/partial files, consume the one process-wide budget. */
static unsigned npt_caption_dxbc_attempts;

void npt_caption_capture_pixel_shader(struct npt_dispatch_context *dispatch,
                                      uint64_t device_id, uint64_t guest_id,
                                      void *shader, HRESULT result,
                                      const void *bytecode, size_t bytecode_bytes,
                                      uint64_t wire_bytes)
{
   struct npt_context *ctx = npt_context_from_dispatch(dispatch);
   if (!ctx || !ctx->debug_name || strcmp(ctx->debug_name, "dwm.exe") ||
       NPT_FAILED(result) || !shader)
      return;
   const char *directory = getenv("NPT_PS_CAPTURE_DIR");
   if (!directory || directory[0] != '/' ||
       strnlen(directory, PATH_MAX) == PATH_MAX ||
       __atomic_load_n(&npt_caption_dxbc_attempts, __ATOMIC_RELAXED) >= 256)
      return; /* unconfigured: no filesystem operation */
   char marker[PATH_MAX];
   int marker_length = snprintf(marker, sizeof(marker), "%s/enable", directory);
   if (marker_length < 0 || (size_t)marker_length >= sizeof(marker)) return;
   struct stat marker_stat;
   if (lstat(marker, &marker_stat) || !S_ISREG(marker_stat.st_mode))
      return;
   unsigned sample = npt_caption_take(&npt_caption_dxbc_attempts, 256);
   if (!sample) return;

   const uint8_t *bytes = bytecode;
   uint32_t declared = 0, chunks = 0;
   const char *reason = "invalid-dxbc";
   int error = 0, dirfd = -1, fd = -1;
   size_t written = 0;
   uint64_t canonical = 0;
   char hash[33] = {0}, filename[160] = {0};
   /* The decoded allocation is bounded by wire_bytes, not the separate
    * BytecodeLength field. Header reads happen only after both are checked. */
   if (!bytes || wire_bytes < 32 || wire_bytes > 65536 ||
       bytecode_bytes != wire_bytes || memcmp(bytes, "DXBC", 4))
      goto done;
   memcpy(&declared, bytes + 24, sizeof(declared));
   memcpy(&chunks, bytes + 28, sizeof(chunks));
   if (declared != bytecode_bytes || chunks > (bytecode_bytes - 32) / 4)
      goto done;
   for (unsigned i = 0; i < 16; ++i)
      snprintf(hash + 2 * i, 3, "%02x", bytes[4 + i]);
   canonical = npt_context_trace_native_id(ctx, shader);
   if (!canonical) { reason = "unregistered-shader"; goto done; }
   int length = snprintf(filename, sizeof(filename),
      "ps-p%lu-c%u-o%016" PRIx64 "-h%s-s%03u.dxbc",
      (unsigned long)getpid(), ctx->ctx_id, canonical, hash, sample);
   if (length < 0 || (size_t)length >= sizeof(filename)) {
      reason = "filename"; goto done;
   }
   dirfd = open(directory, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
   if (dirfd < 0) { reason = "open-directory"; error = errno; goto done; }
   fd = openat(dirfd, filename,
               O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0600);
   if (fd < 0) { reason = "create-new"; error = errno; goto done; }
   struct stat file_stat;
   if (fstat(fd, &file_stat)) {
      reason = "file-type"; error = errno; goto done;
   }
   if (!S_ISREG(file_stat.st_mode)) {
      reason = "file-type"; error = EINVAL; goto done;
   }
   /* At most128 writes, including <=8 EINTRs. Positive short writes advance;
    * zero/error/exhaustion remains UNKNOWN with the partial file preserved. */
   unsigned calls = 0, interrupts = 0;
   while (written < bytecode_bytes && calls++ < 128) {
      ssize_t count = write(fd, bytes + written, bytecode_bytes - written);
      if (count > 0 && (size_t)count <= bytecode_bytes - written) {
         written += (size_t)count;
      } else if (count < 0 && errno == EINTR && ++interrupts <= 8) {
         continue;
      } else {
         reason = "write"; error = count < 0 ? errno : EIO; goto done;
      }
   }
   if (written != bytecode_bytes) { reason = "write-budget"; error = EIO; goto done; }
   reason = "captured";
 done:
   if (fd >= 0 && close(fd)) { reason = "close"; error = errno; }
   if (dirfd >= 0) close(dirfd);
   npt_log("CAPTION_DXBC sample=%u cap=256 ctx=%u device=0x%" PRIx64
           " guest=0x%" PRIx64 " canonical=0x%" PRIx64
           " hr=0x%x bytes=%zu wire=%" PRIu64 " declared=%u chunks=%u"
           " dxvkKey=fs.%s file=%s written=%zu errno=%d result=%s reason=%s",
           sample, ctx->ctx_id, device_id, guest_id, canonical, (unsigned)result,
           bytecode_bytes, wire_bytes, declared, chunks, hash, filename, written, error,
           !strcmp(reason, "captured") ? "CAPTURED" : "UNKNOWN", reason);
}

unsigned npt_caption_trace_alias_sample(void)
{
   static unsigned aliases;
   /* Registration metadata must survive until late window activation.
    * Explicitly configuring the marker opts into these128light rows from
    * startup; copy/readback budgets stay untouched until file creation. */
   return npt_caption_trace_configured() ? npt_caption_take(&aliases, 128) : 0;
}

struct caption_wire { const uint8_t *p; size_t left; };
static bool caption_read(struct caption_wire *w, void *out, size_t bytes)
{
   if (bytes > w->left) return false;
   memcpy(out, w->p, bytes); w->p += bytes; w->left -= bytes; return true;
}

/* The generated encoder writes these scalars consecutively (no struct ABI
 * padding); simple-pointer presence is a uint64 count, not a CPU pointer.
 * Unknown/truncated/skewed commands are ignored by this diagnostic only. */
void npt_caption_trace_copy(struct npt_context *ctx,
                            const struct npt_command_header *h,
                            const void *wire, size_t available)
{
   static unsigned copies;
   if (__atomic_load_n(&copies, __ATOMIC_RELAXED) >= 128 ||
       !npt_caption_trace_enabled() || h->cmd_size > available ||
       h->cmd_size < sizeof(*h)) return;
   const bool whole = h->cmd_type == NPT_CMD_TYPE(255, NPT_IFACE_ID_ID3D11DeviceContext,
                                                 NPT_METHOD_ID3D11DeviceContext_CopyResource);
   const bool region = h->cmd_type == NPT_CMD_TYPE(255, NPT_IFACE_ID_ID3D11DeviceContext,
                                                  NPT_METHOD_ID3D11DeviceContext_CopySubresourceRegion);
   const bool region1 = h->cmd_type == NPT_CMD_TYPE(255, NPT_IFACE_ID_ID3D11DeviceContext1,
                                                   NPT_METHOD_ID3D11DeviceContext1_CopySubresourceRegion1);
   if (!whole && !region && !region1) return;
   struct caption_wire w = { (const uint8_t *)wire + sizeof(*h), h->cmd_size - sizeof(*h) };
   uint64_t dst, src, has_box = 0;
   uint32_t dst_sub = 0, src_sub = 0, xyz[3] = {0}, box[6] = {0}, flags = 0;
   if (!caption_read(&w, &dst, sizeof(dst))) return;
   if (!whole && (!caption_read(&w, &dst_sub, sizeof(dst_sub)) ||
                  !caption_read(&w, xyz, sizeof(xyz)))) return;
   if (!caption_read(&w, &src, sizeof(src))) return;
   if (!whole && (!caption_read(&w, &src_sub, sizeof(src_sub)) ||
                  !caption_read(&w, &has_box, sizeof(has_box)) || has_box > 1 ||
                  (has_box && !caption_read(&w, box, sizeof(box))))) return;
   if (region1 && !caption_read(&w, &flags, sizeof(flags))) return;
   if (w.left) return;
   const unsigned sample = npt_caption_take(&copies, 128);
   if (!sample) return;
   uint64_t src_primary, dst_primary;
   void *s = npt_context_trace_hold_texture(ctx, src, &src_primary);
   void *d = npt_context_trace_hold_texture(ctx, dst, &dst_primary);
   D3D11_TEXTURE2D_DESC sd = {0}, dd = {0};
   if (s) {
      PFN_ID3D11Texture2D_GetDesc gd = NPT_COM_VTBL_FUNC(PFN_ID3D11Texture2D_GetDesc,
         npt_com_vtable(s), NPT_VTBL_ID3D11Texture2D_GetDesc);
      gd(s, &sd);
   }
   if (d) {
      PFN_ID3D11Texture2D_GetDesc gd = NPT_COM_VTBL_FUNC(PFN_ID3D11Texture2D_GetDesc,
         npt_com_vtable(d), NPT_VTBL_ID3D11Texture2D_GetDesc);
      gd(d, &dd);
   }
   npt_log("CAPTION_COPY sample=%u cap=128 ctx=%u dc=0x%" PRIx64 " op=%s src=0x%" PRIx64 " srcCanonical=0x%" PRIx64
           " dst=0x%" PRIx64 " dstCanonical=0x%" PRIx64 " known=%u/%u fmt=%u/%u size=%ux%u/%ux%u sub=%u/%u"
           " dstXYZ=%u,%u,%u hasBox=%" PRIu64 " box=%u,%u,%u,%u,%u,%u flags=0x%x phase=before-dispatch-not-GPU-completion",
           sample, ctx->ctx_id, h->object_id, whole ? "CopyResource" : region1 ? "CopyRegion1" : "CopyRegion",
           src, src_primary, dst, dst_primary, s != NULL, d != NULL, sd.Format, dd.Format,
           sd.Width, sd.Height, dd.Width, dd.Height, src_sub, dst_sub, xyz[0], xyz[1], xyz[2],
           has_box, box[0], box[1], box[2], box[3], box[4], box[5], flags);
   if (s) npt_com_release(s);
   if (d) npt_com_release(d);
}


/* CPU state only: PS0..7 and RTV0. Other stages/slots/UAVs are not covered. */
struct caption_view_state {
   uint64_t view, resource;
   uint32_t view_format, dimension, texture2d, format, width, height;
};

static void caption_view_snapshot(struct npt_context *ctx, void *view, bool ps,
                                  struct caption_view_state *s)
{
   memset(s, 0, sizeof(*s));
   if (!view) return;
   s->view = npt_context_trace_native_id(ctx, view);
   if (ps) {
      D3D11_SHADER_RESOURCE_VIEW_DESC d = {0};
      PFN_ID3D11ShaderResourceView_GetDesc gd = NPT_COM_VTBL_FUNC(PFN_ID3D11ShaderResourceView_GetDesc,
         npt_com_vtable(view), NPT_VTBL_ID3D11ShaderResourceView_GetDesc);
      if (gd) { gd(view, &d); s->view_format = d.Format; s->dimension = d.ViewDimension; }
   } else {
      D3D11_RENDER_TARGET_VIEW_DESC d = {0};
      PFN_ID3D11RenderTargetView_GetDesc gd = NPT_COM_VTBL_FUNC(PFN_ID3D11RenderTargetView_GetDesc,
         npt_com_vtable(view), NPT_VTBL_ID3D11RenderTargetView_GetDesc);
      if (gd) { gd(view, &d); s->view_format = d.Format; s->dimension = d.ViewDimension; }
   }
   PFN_ID3D11View_GetResource get = NPT_COM_VTBL_FUNC(PFN_ID3D11View_GetResource,
      npt_com_vtable(view), NPT_VTBL_ID3D11View_GetResource);
   ID3D11Resource *res = NULL;
   if (get) get(view, &res);
   if (!res) return;
   s->resource = npt_context_trace_native_id(ctx, res);
   ID3D11Texture2D *tex = NULL;
   if (!NPT_FAILED(npt_com_query_interface(res, &NPT_IID_ID3D11Texture2D, (void **)&tex)) && tex) {
      D3D11_TEXTURE2D_DESC d = {0};
      PFN_ID3D11Texture2D_GetDesc gd = NPT_COM_VTBL_FUNC(PFN_ID3D11Texture2D_GetDesc,
         npt_com_vtable(tex), NPT_VTBL_ID3D11Texture2D_GetDesc);
      if (gd) { gd(tex, &d); s->texture2d = 1; s->format = d.Format; s->width = d.Width; s->height = d.Height; }
      npt_com_release(tex);
   }
   npt_com_release(res);
}

static uint32_t caption_draw_bytes(uint32_t command)
{
#define CAPTION_DRAW_CASE(name, bytes) case NPT_CMD_TYPE(255, NPT_IFACE_ID_ID3D11DeviceContext, NPT_METHOD_ID3D11DeviceContext_##name): return bytes
   switch (command) {
   CAPTION_DRAW_CASE(Draw, 8);
   CAPTION_DRAW_CASE(DrawIndexed, 12);
   CAPTION_DRAW_CASE(DrawInstanced, 16);
   CAPTION_DRAW_CASE(DrawIndexedInstanced, 20);
   CAPTION_DRAW_CASE(DrawAuto, 0);
   CAPTION_DRAW_CASE(DrawIndexedInstancedIndirect, 12);
   CAPTION_DRAW_CASE(DrawInstancedIndirect, 12);
   default: return UINT32_MAX;
   }
#undef CAPTION_DRAW_CASE
}

void npt_caption_trace_draw(struct npt_context *ctx,
                            const struct npt_command_header *h, size_t available)
{
   static unsigned draws;
   const uint32_t args = caption_draw_bytes(h->cmd_type);
   if (args == UINT32_MAX || h->cmd_size != sizeof(*h) + args || h->cmd_size > available ||
       __atomic_load_n(&draws, __ATOMIC_RELAXED) >= 32 ||
       !atomic_load_explicit(&ctx->caption_rect_seen, memory_order_acquire) ||
       !npt_caption_trace_enabled()) return;
   void *dc = npt_context_trace_hold_dc(ctx, h->object_id);
   if (!dc) return;
   const unsigned sample = npt_caption_take(&draws, 32);
   if (!sample) { npt_com_release(dc); return; }
   PFN_ID3D11DeviceContext_PSGetShaderResources psget = NPT_COM_VTBL_FUNC(PFN_ID3D11DeviceContext_PSGetShaderResources,
      npt_com_vtable(dc), NPT_VTBL_ID3D11DeviceContext_PSGetShaderResources);
   PFN_ID3D11DeviceContext_OMGetRenderTargets omget = NPT_COM_VTBL_FUNC(PFN_ID3D11DeviceContext_OMGetRenderTargets,
      npt_com_vtable(dc), NPT_VTBL_ID3D11DeviceContext_OMGetRenderTargets);
   PFN_ID3D11DeviceContext_PSGetShader shaderget = NPT_COM_VTBL_FUNC(PFN_ID3D11DeviceContext_PSGetShader,
      npt_com_vtable(dc), NPT_VTBL_ID3D11DeviceContext_PSGetShader);
   PFN_ID3D11DeviceContext_OMGetBlendState blendget = NPT_COM_VTBL_FUNC(PFN_ID3D11DeviceContext_OMGetBlendState,
      npt_com_vtable(dc), NPT_VTBL_ID3D11DeviceContext_OMGetBlendState);
   ID3D11ShaderResourceView *ps[8] = {0};
   ID3D11RenderTargetView *rtv = NULL;
   ID3D11PixelShader *shader = NULL;
   ID3D11BlendState *blend = NULL;
   FLOAT factor[4] = {0}; UINT mask = 0;
   if (psget) psget(dc, 0, 8, ps);
   if (omget) omget(dc, 1, &rtv, NULL);
   if (shaderget) shaderget(dc, &shader, NULL, NULL);
   if (blendget) blendget(dc, &blend, factor, &mask);
   const uint64_t shader_id = npt_context_trace_native_id(ctx, shader);
   const uint64_t blend_id = npt_context_trace_native_id(ctx, blend);
   D3D11_BLEND_DESC bd = {0}; bool blend_known = false;
   if (blend) {
      PFN_ID3D11BlendState_GetDesc gd = NPT_COM_VTBL_FUNC(PFN_ID3D11BlendState_GetDesc,
         npt_com_vtable(blend), NPT_VTBL_ID3D11BlendState_GetDesc);
      if (gd) { gd(blend, &bd); blend_known = true; }
   }
   struct caption_view_state dst;
   caption_view_snapshot(ctx, rtv, false, &dst);
   /* Each entry: slot:viewID/resourceID/viewFmt/viewDim/texture2D/resourceFmt/WxH.
    * Null PS slots are zero entries; psMask distinguishes unregistered nonnull.
    * This fixed buffer bounds log output without truncating COM cleanup. */
   char inputs[1024] = {0}; size_t used = 0; bool clipped = false;
   unsigned nonnull = 0;
   for (unsigned i = 0; i < 8; ++i) {
      struct caption_view_state src;
      if (ps[i]) nonnull |= 1u << i;
      caption_view_snapshot(ctx, ps[i], true, &src);
      if (!clipped) {
         int n = snprintf(inputs + used, sizeof(inputs) - used,
            "%s%u:%" PRIx64 "/%" PRIx64 "/%u/%u/%u/%u/%ux%u",
            i ? "," : "", i, src.view, src.resource, src.view_format, src.dimension,
            src.texture2d, src.format, src.width, src.height);
         if (n < 0 || (size_t)n >= sizeof(inputs) - used) clipped = true;
         else used += (size_t)n;
      }
      if (ps[i]) npt_com_release(ps[i]);
   }
   npt_log("CAPTION_DRAW sample=%u cap=32 ctx=%u dc=0x%" PRIx64 " cmd=0x%x coverage=PS0..7,RTV0 getters=%u/%u/%u/%u psMask=0x%x"
           " psShader=0x%" PRIx64 " psPresent=%u dstView=0x%" PRIx64 " dstRes=0x%" PRIx64
           " dstViewFmt=%u dstDimension=%u dstTexture2D=%u dstFmt=%u dstSize=%ux%u"
           " blend=0x%" PRIx64 " blendNull=%u blendDescKnown=%u alphaCoverage=%u independentBlend=%u"
           " rt0Blend=%u,%u,%u,%u,%u,%u,%u,0x%x factor=%g,%g,%g,%g sampleMask=0x%x"
           " psSlots=%s clipped=%u phase=per-getter-CPU-state-before-draw-not-GPU-completion rowEnd=CAPTION_DRAW_END",
           sample, ctx->ctx_id, h->object_id, h->cmd_type, psget != NULL, omget != NULL, shaderget != NULL, blendget != NULL, nonnull,
           shader_id, shader != NULL, dst.view, dst.resource, dst.view_format, dst.dimension, dst.texture2d, dst.format, dst.width, dst.height,
           blend_id, blend == NULL, blend_known, bd.AlphaToCoverageEnable, bd.IndependentBlendEnable,
           bd.RenderTarget[0].BlendEnable, bd.RenderTarget[0].SrcBlend, bd.RenderTarget[0].DestBlend, bd.RenderTarget[0].BlendOp,
           bd.RenderTarget[0].SrcBlendAlpha, bd.RenderTarget[0].DestBlendAlpha, bd.RenderTarget[0].BlendOpAlpha, bd.RenderTarget[0].RenderTargetWriteMask,
           (double)factor[0], (double)factor[1], (double)factor[2], (double)factor[3], mask, inputs, clipped);
   if (shader) npt_com_release(shader);
   if (blend) npt_com_release(blend);
   if (rtv) npt_com_release(rtv);
   npt_com_release(dc);
}

struct caption_colors {
   uint32_t pixels, alpha0, alpha255, alpha_other, magenta, green, transparent_rgb, rgb_gt_alpha;
   char first[129];
};

/* Caller already proved tight rows <=64KiB and the pitched source span valid.
 * Color counts are observations, not assumptions about premultiplication. */
static void caption_input_colors(const uint8_t *p, uint32_t pitch, uint32_t width,
                                  uint32_t height, uint32_t format,
                                  struct caption_colors *s)
{
   memset(s, 0, sizeof(*s));
   unsigned first = 0;
   for (uint32_t y = 0; y < height; ++y) {
      for (uint32_t x = 0; x < width; ++x) {
         const uint8_t *v = p + (uint64_t)y * pitch + (uint64_t)x * 4;
         const uint8_t red = v[format == 87 ? 2 : 0], green = v[1];
         const uint8_t blue = v[format == 87 ? 0 : 2], alpha = v[3];
         s->pixels++; s->alpha0 += alpha == 0; s->alpha255 += alpha == 255;
         s->alpha_other += alpha != 0 && alpha != 255;
         s->magenta += red == 255 && green == 0 && blue == 255;
         s->green += red == 0 && green == 255 && blue == 0;
         s->transparent_rgb += alpha == 0 && (red || green || blue);
         s->rgb_gt_alpha += red > alpha || green > alpha || blue > alpha;
         for (unsigned b = 0; b < 4 && first < 64; ++b, ++first)
            snprintf(s->first + first * 2, 3, "%02x", v[b]);
      }
   }
}

static bool caption_rect_valid(const D3D11_TEXTURE2D_DESC *d, uint32_t sub,
                               uint32_t pitch, uint32_t bytes,
                               uint32_t x0, uint32_t y0, uint32_t x1, uint32_t y1)
{
   if (sub || (d->Format != 28 && d->Format != 87) || d->SampleDesc.Count != 1 ||
       !d->MipLevels || !d->ArraySize || d->Width > 4096 || d->Height > 4096 ||
       d->Height <= 512 || x0 >= x1 || y0 >= y1 || x1 > d->Width || y1 > d->Height) return false;
   const uint64_t row = (uint64_t)(x1 - x0) * 4;
   const uint64_t rows = y1 - y0;
   return pitch >= row && row * rows <= 65536 &&
      (rows - 1) * pitch + row <= bytes;
}

static void caption_rect_hash(const uint8_t *p, uint32_t pitch, uint32_t row,
                               uint32_t rows, uint64_t *hash, uint32_t *nonzero)
{
   *hash = 1469598103934665603ull; *nonzero = 0;
   for (uint32_t y = 0; y < rows; ++y)
      for (uint32_t x = 0; x < row; ++x) {
         const uint8_t v = p[(uint64_t)y * pitch + x];
         *hash ^= v; *hash *= 1099511628211ull; *nonzero += v != 0;
      }
}

static void npt_trace_atlas_rect(struct npt_context *ctx, void *dc, void *resource,
                                 uint64_t id, const D3D11_TEXTURE2D_DESC *desc,
                                 uint32_t sub, uint32_t pitch, uint32_t bytes,
                                 uint32_t x0, uint32_t y0, uint32_t x1, uint32_t y1,
                                 const void *payload, bool immediate)
{
   static unsigned samples;
   if (__atomic_load_n(&samples, __ATOMIC_RELAXED) >= 32 ||
       !npt_caption_trace_enabled() || !immediate || !payload ||
       !caption_rect_valid(desc, sub, pitch, bytes, x0, y0, x1, y1)) return;
   const unsigned sample = npt_caption_take(&samples, 32);
   if (!sample) return;
   const uint32_t width = x1 - x0, height = y1 - y0, row = width * 4;
   uint64_t input_hash, output_hash = 0; uint32_t input_nz, output_nz = 0;
   caption_rect_hash(payload, pitch, row, height, &input_hash, &input_nz);
   struct caption_colors colors;
   caption_input_colors(payload, pitch, width, height, desc->Format, &colors);
   npt_log("CAPTION_INPUT sample=%u cap=32 ctx=%u object=0x%" PRIx64 " canonical=0x%" PRIx64
           " fmt=%u image=%ux%u box=%u,%u,%u,%u pixels=%u alpha0=%u alpha255=%u alphaOther=%u"
           " rgbMagenta=%u rgbGreen=%u transparentRGB=%u rgbGtAlpha=%u firstTightBytes64=%s phase=CPU-payload-not-GPU-pixels",
           sample, ctx->ctx_id, id, npt_context_trace_primary_id(ctx, id), desc->Format,
           desc->Width, desc->Height, x0, y0, x1, y1, colors.pixels, colors.alpha0, colors.alpha255,
           colors.alpha_other, colors.magenta, colors.green, colors.transparent_rgb, colors.rgb_gt_alpha, colors.first);
   atomic_store_explicit(&ctx->caption_rect_seen, true, memory_order_release);
   ID3D11Device *dev = NULL; ID3D11Texture2D *st = NULL;
   HRESULT create_hr = NPT_E_FAIL, map_hr = NPT_E_FAIL;
   bool copied = false, readback = false;
   uint32_t mapped_pitch = 0;
   PFN_ID3D11DeviceChild_GetDevice get_dev = NPT_COM_VTBL_FUNC(PFN_ID3D11DeviceChild_GetDevice,
      npt_com_vtable(resource), NPT_VTBL_ID3D11DeviceChild_GetDevice);
   get_dev(resource, &dev);
   if (dev) {
      D3D11_TEXTURE2D_DESC sd = *desc;
      sd.Width = width; sd.Height = height; sd.MipLevels = 1; sd.ArraySize = 1;
      sd.Usage = D3D11_USAGE_STAGING; sd.BindFlags = 0;
      sd.CPUAccessFlags = D3D11_CPU_ACCESS_READ; sd.MiscFlags = 0;
      PFN_ID3D11Device_CreateTexture2D create = NPT_COM_VTBL_FUNC(PFN_ID3D11Device_CreateTexture2D,
         npt_com_vtable(dev), NPT_VTBL_ID3D11Device_CreateTexture2D);
      create_hr = create(dev, &sd, NULL, &st);
      if (create_hr == NPT_S_OK && st) {
         PFN_ID3D11DeviceContext_CopySubresourceRegion copy = NPT_COM_VTBL_FUNC(PFN_ID3D11DeviceContext_CopySubresourceRegion,
            npt_com_vtable(dc), NPT_VTBL_ID3D11DeviceContext_CopySubresourceRegion);
         PFN_ID3D11DeviceContext_Map map = NPT_COM_VTBL_FUNC(PFN_ID3D11DeviceContext_Map,
            npt_com_vtable(dc), NPT_VTBL_ID3D11DeviceContext_Map);
         PFN_ID3D11DeviceContext_Unmap unmap = NPT_COM_VTBL_FUNC(PFN_ID3D11DeviceContext_Unmap,
            npt_com_vtable(dc), NPT_VTBL_ID3D11DeviceContext_Unmap);
         const D3D11_BOX box = { x0, y0, 0, x1, y1, 1 };
         copy(dc, (ID3D11Resource *)st, 0, 0, 0, 0, resource, sub, &box);
         copied = true;
         D3D11_MAPPED_SUBRESOURCE m = {0};
         map_hr = map(dc, (ID3D11Resource *)st, 0, D3D11_MAP_READ,
                      D3D11_MAP_FLAG_DO_NOT_WAIT, &m);
         if (map_hr == NPT_S_OK) {
            mapped_pitch = m.RowPitch;
            if (m.pData && m.RowPitch >= row && (uint64_t)m.RowPitch * height <= 65536) {
               caption_rect_hash(m.pData, m.RowPitch, row, height, &output_hash, &output_nz);
               readback = true;
            }
            unmap(dc, (ID3D11Resource *)st, 0);
         }
      }
   }
   npt_log("CAPTION_RECT sample=%u cap=32 ctx=%u object=0x%" PRIx64 " canonical=0x%" PRIx64
           " fmt=%u image=%ux%u box=%u,%u,%u,%u staging=%ux%u tightBytes=%u pitch=%u payload=%u"
           " inputFnv=%016" PRIx64 " inputNonzero=%u copied=%u createHr=0x%x mapHr=0x%x mappedPitch=%u"
           " readback=%u outputFnv=%016" PRIx64 " outputNonzero=%u result=%s no-wait-no-retry",
           sample, ctx->ctx_id, id, npt_context_trace_primary_id(ctx, id), desc->Format,
           desc->Width, desc->Height, x0, y0, x1, y1, width, height, row * height,
           pitch, bytes, input_hash, input_nz, copied, (unsigned)create_hr, (unsigned)map_hr,
           mapped_pitch, readback, output_hash, output_nz,
           readback ? (input_hash == output_hash ? "MATCH" : "MISMATCH") : "UNKNOWN");
   if (st) npt_com_release(st);
   if (dev) npt_com_release(dev);
}

/* ---- texture update trace (initial-data path of small textures, any format): payload tight-row FNV, and after the
 * update a fresh host staging copy read back and hashed the same way.  Compare with the guest "FMT guest tex-init". */
static volatile int npt_fmt_tex_count;

static uint32_t
npt_trace_bpp(uint32_t f)
{
   if (f >= 1 && f <= 4) return 16;
   if (f >= 5 && f <= 8) return 12;
   if ((f >= 9 && f <= 22)) return 8;
   if (f >= 23 && f <= 47) return 4;
   if (f >= 48 && f <= 59) return 2;
   if (f >= 60 && f <= 65) return 1;
   if (f >= 85 && f <= 93) return 4;
   return 0;
}

static void
npt_trace_tex_update(struct npt_context *ctx, void *imm_ctx, void *resource, uint64_t resource_id,
                     uint32_t subresource, uint32_t row_pitch, uint32_t byte_size, bool has_box,
                     uint32_t bx0, uint32_t by0, uint32_t bx1, uint32_t by1,
                     uint32_t bz0, uint32_t bz1, const void *payload, bool immediate)
{
   if (npt_fmt_tex_count >= 300 && !npt_caption_trace_enabled()) return;
   if (!npt_context_object_is(ctx, resource_id, NPT_OBJECT_TYPE_ID3D11TEXTURE2D)) return;
   PFN_ID3D11Texture2D_GetDesc gd = NPT_COM_VTBL_FUNC(PFN_ID3D11Texture2D_GetDesc, npt_com_vtable(resource), NPT_VTBL_ID3D11Texture2D_GetDesc);
   if (!gd) return;
   D3D11_TEXTURE2D_DESC d; gd(resource, &d);
   if (d.Height > 512) {
      if (!has_box || (bz0 == 0 && bz1 == 1))
         npt_trace_atlas_rect(ctx, imm_ctx, resource, resource_id, &d, subresource, row_pitch, byte_size,
            has_box ? bx0 : 0, has_box ? by0 : 0, has_box ? bx1 : d.Width, has_box ? by1 : d.Height, payload, immediate);
      return;
   }
   if (d.Width > 4096 || npt_fmt_tex_count >= 300) return;
   if (!has_box) { bx0 = 0; by0 = 0; bx1 = d.Width; by1 = d.Height; }
   if (bx1 <= bx0 || by1 <= by0 || bx1 > d.Width || by1 > d.Height) return;
   const uint32_t bpp = npt_trace_bpp(d.Format);
   if (!bpp || subresource != 0) return;
   __atomic_fetch_add(&npt_fmt_tex_count, 1, __ATOMIC_RELAXED);
   const uint8_t *p = payload; uint64_t h = 1469598103934665603ull, h2 = h; uint32_t nz = 0, nz2 = 0;
   for (uint32_t y = 0; y < by1 - by0; y++)
      for (uint32_t x = 0; x < (bx1 - bx0) * bpp; x++) {
         uint64_t o = (uint64_t)y * row_pitch + x; uint8_t v = o < byte_size ? p[o] : 0;
         h ^= v; h *= 1099511628211ull; nz += v != 0;
      }
   /* fresh staging readback of what DXVK now holds */
   char rb[64] = "no-readback";
   PFN_ID3D11Device_CreateTexture2D mk = NULL; (void)mk;
   ID3D11Device *dev = NULL;
   PFN_ID3D11DeviceChild_GetDevice get_dev = NPT_COM_VTBL_FUNC(PFN_ID3D11DeviceChild_GetDevice, npt_com_vtable(resource), NPT_VTBL_ID3D11DeviceChild_GetDevice);
   if (get_dev) get_dev(resource, &dev);
   if (dev && immediate) {
      D3D11_TEXTURE2D_DESC sd = d; sd.Usage = D3D11_USAGE_STAGING; sd.BindFlags = 0; sd.CPUAccessFlags = D3D11_CPU_ACCESS_READ; sd.MiscFlags = 0;
      PFN_ID3D11Device_CreateTexture2D ct = NPT_COM_VTBL_FUNC(PFN_ID3D11Device_CreateTexture2D, npt_com_vtable(dev), NPT_VTBL_ID3D11Device_CreateTexture2D);
      ID3D11Texture2D *st = NULL;
      if (ct && !NPT_FAILED(ct(dev, &sd, NULL, &st)) && st) {
         PFN_ID3D11DeviceContext_CopyResource cr = NPT_COM_VTBL_FUNC(PFN_ID3D11DeviceContext_CopyResource, npt_com_vtable(imm_ctx), NPT_VTBL_ID3D11DeviceContext_CopyResource);
         PFN_ID3D11DeviceContext_Map mp = NPT_COM_VTBL_FUNC(PFN_ID3D11DeviceContext_Map, npt_com_vtable(imm_ctx), NPT_VTBL_ID3D11DeviceContext_Map);
         PFN_ID3D11DeviceContext_Unmap um = NPT_COM_VTBL_FUNC(PFN_ID3D11DeviceContext_Unmap, npt_com_vtable(imm_ctx), NPT_VTBL_ID3D11DeviceContext_Unmap);
         D3D11_MAPPED_SUBRESOURCE m; memset(&m, 0, sizeof(m));
         if (cr && mp && um) {
            cr(imm_ctx, (ID3D11Resource *)st, (ID3D11Resource *)resource);
            if (!NPT_FAILED(mp(imm_ctx, (ID3D11Resource *)st, 0, D3D11_MAP_READ, 0, &m)) && m.pData) {
               for (uint32_t y = by0; y < by1; y++)
                  for (uint32_t x = bx0 * bpp; x < bx1 * bpp; x++) { uint8_t v = ((const uint8_t *)m.pData)[(uint64_t)y * m.RowPitch + x]; h2 ^= v; h2 *= 1099511628211ull; nz2 += v != 0; }
               snprintf(rb, sizeof(rb), "dxvk_fnv=%016" PRIx64 " nonzero=%u", h2, nz2);
               um(imm_ctx, (ID3D11Resource *)st, 0);
            }
         }
         npt_com_release(st);
      }
   }
   if (dev) npt_com_release(dev);
   npt_log("FMT host tex-update id=0x%" PRIx64 " box=[%u,%u)x[%u,%u)%s fmt=%u %ux%u usage=%u bind=0x%x rowPitch=%u payload=%u "
           "payload_tightFnv=%016" PRIx64 " nonzero=%u | %s %s", resource_id, bx0, bx1, by0, by1, has_box ? "" : "(full)", d.Format, d.Width, d.Height,
           d.Usage, d.BindFlags, row_pitch, byte_size, h, nz, rb,
           !strncmp(rb, "dxvk_fnv", 8) ? (h2 == h ? "MATCH" : "MISMATCH") : "");
}

void
npt_resource_update(struct npt_context *ctx,
                    uint64_t context_id,
                    uint64_t resource_id, uint32_t subresource,
                    uint32_t row_pitch, uint32_t depth_pitch,
                    uint32_t byte_size,
                    bool has_box,
                    uint32_t box_left, uint32_t box_top, uint32_t box_front,
                    uint32_t box_right, uint32_t box_bottom, uint32_t box_back,
                    const void *payload)
{
   /* Recover the device and immediate context from the resource.
    * Both calls add refs that we release after UpdateSubresource. */
   void *resource = npt_context_lookup_object(ctx, NULL, resource_id,
                                              NPT_OBJECT_TYPE_IUNKNOWN);
   if (!resource) {
      npt_log("resource_update: NULL resource");
      return;
   }

   /* A D3D12 resource is not an ID3D11DeviceChild: slot 3 of its vtable is
    * GetPrivateData, so the D3D11 path below would call the wrong method
    * entirely and the update would be lost with no error anywhere. Route it
    * through the D3D12 API, which takes the same (subresource, box, pitches)
    * shape. WriteToSubresource requires the resource to be mapped, and the
    * guest's no-pointer Map is local bookkeeping that never reaches the host,
    * so the bracket has to be applied here. */
   if (npt_context_object_is(ctx, resource_id, NPT_OBJECT_TYPE_ID3D12RESOURCE)) {
      PFN_ID3D12Resource_WriteToSubresource wts12 =
         NPT_COM_VTBL_FUNC(PFN_ID3D12Resource_WriteToSubresource,
                           npt_com_vtable(resource),
                           NPT_VTBL_ID3D12Resource_WriteToSubresource);
      if (!wts12) {
         npt_log("resource_update: no D3D12 WriteToSubresource");
         return;
      }
      D3D12_BOX box12;
      const D3D12_BOX *box12_arg = NULL;
      if (has_box) {
         box12.left   = box_left;
         box12.top    = box_top;
         box12.front  = box_front;
         box12.right  = box_right;
         box12.bottom = box_bottom;
         box12.back   = box_back;
         box12_arg = &box12;
      }
      PFN_ID3D12Resource_Map map12 =
         NPT_COM_VTBL_FUNC(PFN_ID3D12Resource_Map, npt_com_vtable(resource),
                           NPT_VTBL_ID3D12Resource_Map);
      PFN_ID3D12Resource_Unmap unmap12 =
         NPT_COM_VTBL_FUNC(PFN_ID3D12Resource_Unmap, npt_com_vtable(resource),
                           NPT_VTBL_ID3D12Resource_Unmap);
      const bool mapped = map12 && !NPT_FAILED(map12(resource, subresource,
                                                     NULL, NULL));
      HRESULT hr = wts12(resource, subresource, box12_arg, payload,
                         row_pitch, depth_pitch);
      if (mapped && unmap12)
         unmap12(resource, subresource, NULL);
      if (NPT_FAILED(hr))
         npt_log("resource_update: D3D12 WriteToSubresource sub %u failed "
                 "0x%08x (mapped=%d pitch=%u/%u)", subresource,
                 (unsigned)hr, (int)mapped, row_pitch, depth_pitch);
      return;
   }

   ID3D11Device *device = NULL;
   PFN_ID3D11DeviceChild_GetDevice get_dev =
      NPT_COM_VTBL_FUNC(PFN_ID3D11DeviceChild_GetDevice,
                        npt_com_vtable(resource),
                        NPT_VTBL_ID3D11DeviceChild_GetDevice);
   get_dev(resource, &device);
   if (!device) {
      npt_log("resource_update: GetDevice returned NULL");
      return;
   }

   /* context_id selects the context the update is recorded on: 0 is the
    * device's immediate context, otherwise a deferred ID3D11DeviceContext
    * the guest created (its UpdateSubresource lands in that context's
    * command list).  An unknown id falls back to the immediate context. */
   ID3D11DeviceContext *imm_ctx = NULL;
   bool imm_ctx_owned = false;
   if (context_id) {
      imm_ctx = npt_context_lookup_object(ctx, NULL, context_id,
                                          NPT_OBJECT_TYPE_ID3D11DEVICECONTEXT);
      if (!imm_ctx)
         npt_log("resource_update: unknown context 0x%" PRIx64
                 ", using the immediate context", context_id);
   }
   if (!imm_ctx) {
      PFN_ID3D11Device_GetImmediateContext get_ctx =
         NPT_COM_VTBL_FUNC(PFN_ID3D11Device_GetImmediateContext,
                           npt_com_vtable(device),
                           NPT_VTBL_ID3D11Device_GetImmediateContext);
      get_ctx(device, &imm_ctx);
      imm_ctx_owned = true;
   }
   if (!imm_ctx) {
      npt_log("resource_update: GetImmediateContext returned NULL");
      npt_com_release(device);
      return;
   }

   D3D11_BOX box;
   const D3D11_BOX *box_arg = NULL;
   if (has_box) {
      box.left   = box_left;
      box.top    = box_top;
      box.front  = box_front;
      box.right  = box_right;
      box.bottom = box_bottom;
      box.back   = box_back;
      box_arg = &box;
   }

   PFN_ID3D11DeviceContext_UpdateSubresource update =
      NPT_COM_VTBL_FUNC(PFN_ID3D11DeviceContext_UpdateSubresource,
                        npt_com_vtable(imm_ctx),
                        NPT_VTBL_ID3D11DeviceContext_UpdateSubresource);
   if (npt_trace_buf_wanted(ctx, resource_id, resource)) {
      uint64_t f; uint32_t z, l; char hx[40];
      npt_trace_buf_sum(payload, byte_size, &f, &z, &l, hx);
      npt_log("FMT host buf-update res=0x%" PRIx64 " box=%s[%u,%u) payload n=%u fnv=%016" PRIx64
              " nonzero=%u lastNonzero=%u first16=%s", resource_id, has_box ? "yes" : "no",
              has_box ? box_left : 0, has_box ? box_right : 0, byte_size, f, z, l, hx);
   }
   update(imm_ctx, resource, subresource, box_arg, payload,
          row_pitch, depth_pitch);
   npt_trace_tex_update(ctx, imm_ctx, resource, resource_id, subresource, row_pitch, byte_size, has_box, box_left, box_top, box_right, box_bottom, box_front, box_back, payload, imm_ctx_owned);

   if (imm_ctx_owned)
      npt_com_release(imm_ctx);
   npt_com_release(device);
}

/* MAP and UNMAP name their API family with context_id: zero means a
 * D3D12 resource the guest maps directly, non-zero the ID3D11DeviceContext
 * to map it through.  Each family keeps an unrelated method at the slots
 * the other's path calls (ID3D12Resource::Map is slot 8, SetEvictionPriority
 * on an ID3D11Resource), so a context_id disagreeing with what the resource
 * is would call the wrong method and report nothing.  The field still
 * selects -- a resource registered under no precise type answers to either
 * family -- and the object table only vetoes a definite contradiction. */
static bool
npt_resource_family_matches(struct npt_context *ctx, uint64_t resource_id,
                            uint64_t context_id, const char *what)
{
   const npt_object_type want = context_id ? NPT_OBJECT_TYPE_ID3D11RESOURCE
                                           : NPT_OBJECT_TYPE_ID3D12RESOURCE;
   if (npt_context_object_is(ctx, resource_id, want))
      return true;

   npt_log("%s: resource 0x%" PRIx64 " is not a D3D%s resource "
           "(context_id 0x%" PRIx64 ")", what, resource_id,
           context_id ? "11" : "12", context_id);
   return false;
}

/* Returns 0 on invalid flags (causes D3D11 Map to fail). */
static D3D11_MAP
npt_access_flags_to_d3d11_map(uint32_t access_flags)
{
   const bool read = access_flags & NPT_MAP_ACCESS_READ;
   const bool write = access_flags & NPT_MAP_ACCESS_WRITE;

   if (read && write)
      return D3D11_MAP_READ_WRITE;
   if (read)
      return D3D11_MAP_READ;
   if (access_flags & NPT_MAP_ACCESS_DISCARD)
      return D3D11_MAP_WRITE_DISCARD;
   if (access_flags & NPT_MAP_ACCESS_NO_OVERWRITE)
      return D3D11_MAP_WRITE_NO_OVERWRITE;
   if (write)
      return D3D11_MAP_WRITE;

   return (D3D11_MAP)0;
}

/* ---- per-context sync-map bookkeeping --------------------------------- */

static struct npt_sync_map_entry *
npt_sync_map_find(struct npt_context *ctx, uint64_t resource_id,
                  uint32_t subresource)
{
   for (uint32_t i = 0; i < ctx->sync_maps.count; ++i) {
      struct npt_sync_map_entry *e = &ctx->sync_maps.entries[i];
      if (e->resource_id == resource_id && e->subresource == subresource)
         return e;
   }
   return NULL;
}

static struct npt_sync_map_entry *
npt_sync_map_add(struct npt_context *ctx)
{
   if (ctx->sync_maps.count == ctx->sync_maps.cap) {
      uint32_t cap = ctx->sync_maps.cap ? ctx->sync_maps.cap * 2 : 8;
      struct npt_sync_map_entry *e =
         realloc(ctx->sync_maps.entries, cap * sizeof(*e));
      if (!e)
         return NULL;
      ctx->sync_maps.entries = e;
      ctx->sync_maps.cap = cap;
   }
   return &ctx->sync_maps.entries[ctx->sync_maps.count++];
}

static void
npt_sync_map_remove(struct npt_context *ctx, struct npt_sync_map_entry *e)
{
   uint32_t idx = (uint32_t)(e - ctx->sync_maps.entries);
   ctx->sync_maps.entries[idx] =
      ctx->sync_maps.entries[--ctx->sync_maps.count];
}

static HRESULT
npt_resource_map_pinned(struct npt_context *ctx,
                        uint64_t context_id, uint64_t resource_id,
                        uint32_t subresource, uint32_t access_flags,
                        uint32_t api_map_flags,
                        struct npt_resource *shmem_res,
                        uint64_t read_range_begin,
                        uint64_t read_range_end,
                        uint64_t byte_size,
                        uint32_t mip_rows, uint32_t mip_depth,
                        uint32_t shmem_offset,
                        uint32_t *out_row_pitch, uint32_t *out_depth_pitch,
                        uint32_t *out_mapped_size,
                        uint32_t *out_external_cookie)
{
   *out_row_pitch = 0;
   *out_depth_pitch = 0;
   *out_mapped_size = 0;
   *out_external_cookie = NPT_EXTERNAL_COOKIE_NONE;

   void *resource = npt_context_lookup_object(ctx, NULL, resource_id,
                                              NPT_OBJECT_TYPE_IUNKNOWN);
   if (!resource) {
      npt_log("map_resource: NULL resource");
      return NPT_E_FAIL;
   }
   if (!npt_resource_family_matches(ctx, resource_id, context_id,
                                    "map_resource"))
      return NPT_E_INVALIDARG;

   if ((uint64_t)shmem_offset >= shmem_res->size) {
      npt_log("map_resource: shmem_offset %u exceeds shmem size %zu",
              shmem_offset, shmem_res->size);
      return NPT_E_FAIL;
   }

   D3D11_MAPPED_SUBRESOURCE mapped;
   memset(&mapped, 0, sizeof(mapped));
   HRESULT hr;
   uint32_t external_cookie = NPT_EXTERNAL_COOKIE_NONE;

   if (!context_id) {
      /* RowPitch/DepthPitch stay 0: this path serves buffers, and the
       * guest computes texture layouts via GetCopyableFootprints. */
      D3D12_RANGE read_range;
      const D3D12_RANGE *rr = NULL;
      if (read_range_begin != NPT_MAP_RANGE_NULL) {
         read_range.Begin = (SIZE_T)read_range_begin;
         read_range.End = (SIZE_T)read_range_end;
         rr = &read_range;
      }

      void *pData = NULL;
      PFN_ID3D12Resource_Map map12 =
         NPT_COM_VTBL_FUNC(PFN_ID3D12Resource_Map,
                           npt_com_vtable(resource),
                           NPT_VTBL_ID3D12Resource_Map);
      if (!map12) {
         npt_log("map_resource: no D3D12 Map");
         return NPT_E_FAIL;
      }
      hr = map12(resource, subresource, rr, &pData);
      if (NPT_FAILED(hr))
         return hr;
      if (!pData) {
         npt_log("map_resource: D3D12 Map returned NULL data");
         return NPT_E_FAIL;
      }
      mapped.pData = pData;
   } else {
      void *imm_ctx = npt_context_lookup_object(ctx, NULL, context_id,
                                                NPT_OBJECT_TYPE_ID3D11DEVICECONTEXT);
      if (!imm_ctx) {
         npt_log("map_resource: NULL immediate context");
         return NPT_E_FAIL;
      }

      D3D11_MAP d3d11_map_type = npt_access_flags_to_d3d11_map(access_flags);

      /* May block on GPU sync; the guest's spin on the ring head
       * propagates the stall. */
      PFN_ID3D11DeviceContext_Map map_fn =
         NPT_COM_VTBL_FUNC(PFN_ID3D11DeviceContext_Map,
                           npt_com_vtable(imm_ctx),
                           NPT_VTBL_ID3D11DeviceContext_Map);
      hr = map_fn(imm_ctx, resource, subresource,
                  d3d11_map_type, api_map_flags, &mapped);

      if (NPT_FAILED(hr))
         return hr;

      /* A buffer bound to guest shmems may have mapped one of them: the
       * guest then writes it in place, and nothing is copied here or at
       * UNMAP. */
      const struct npt_d3d_library *lib = npt_renderer_get_library();
      uint32_t cookie;
      if (lib && lib->pfn_d3d11_buffer_external_cookie &&
          lib->pfn_d3d11_buffer_external_cookie(resource, mapped.pData,
                                                &cookie) == 0)
         external_cookie = cookie;
   }

   /* Memcpy bound for READ (and matching WRITE on the Unmap path):
    *   1. textures: the D3D pitches over the guest's row/slice extent,
    *      every slice but the last at the full DepthPitch (a DepthPitch
    *      below one slice is treated as unpadded);
    *   2. buffers: byte_size;
    *   3. shmem size as fallback.
    * Clamped to byte_size -- the guest's slot -- and to the shmem, so
    * neither D3D's mapped region nor a neighbouring slot of the shared
    * map pool is overrun. */
   uint64_t bound;
   if (mip_rows && mip_depth) {
      const uint64_t slice = (uint64_t)mapped.RowPitch * mip_rows;
      const uint64_t strides =
         (mip_depth > 1u && mapped.DepthPitch >= slice)
            ? (uint64_t)mapped.DepthPitch * (mip_depth - 1u)
            : slice * (mip_depth - 1u);
      bound = strides + slice;
   } else {
      bound = byte_size ? byte_size : shmem_res->size - shmem_offset;
   }
   if (byte_size && bound > byte_size)
      bound = byte_size;
   if (bound > shmem_res->size - shmem_offset)
      bound = shmem_res->size - shmem_offset;
   const bool external = external_cookie != NPT_EXTERNAL_COOKIE_NONE;
   const uint32_t mapped_size = external ? 0 : (uint32_t)bound;

   if ((access_flags & NPT_MAP_ACCESS_READ) && !external)
      memcpy((uint8_t *)shmem_res->u.data + shmem_offset, mapped.pData,
             mapped_size);

   struct npt_sync_map_entry *entry =
      npt_sync_map_find(ctx, resource_id, subresource);
   if (entry) {
      /* Double-map of the same subresource: D3D11 disallows it, so
       * this indicates guest state drift; replace the stale entry. */
      npt_log("map_resource: resource 0x%" PRIx64 " sub %u already "
              "mapped, replacing stale entry", resource_id, subresource);
   } else {
      entry = npt_sync_map_add(ctx);
      if (!entry) {
         npt_log("map_resource: sync map table OOM");
         return NPT_E_OUTOFMEMORY;
      }
   }
   *entry = (struct npt_sync_map_entry){
      .resource_id  = resource_id,
      .subresource  = subresource,
      .access_flags = access_flags,
      .mapped_data  = mapped.pData,
      .mapped_size  = mapped_size,
      .persistent   = !!(access_flags & NPT_MAP_ACCESS_PERSISTENT),
      .external     = external,
   };

   *out_row_pitch = mapped.RowPitch;
   *out_depth_pitch = mapped.DepthPitch;
   *out_mapped_size = mapped_size;
   *out_external_cookie = external_cookie;
   return hr;
}


/* ---- A8/R8 map write-back trace (white-text chase) -----------------------
 * FNV-1a 64 over the same region the guest hashed at Unmap (rec.byte_size),
 * of (a) the guest's shmem slot and (b) the host D3D11 mapped pointer after the
 * write-back, and (c) after Unmap, a fresh host Map(READ) of the staging
 * texture.  Rate-limited to 300 lines. */
static volatile int npt_fmt_trace_count;

static uint64_t
npt_trace_fnv(const void *p, uint64_t n)
{
   const uint8_t *b = p; uint64_t h = 1469598103934665603ull;
   for (uint64_t i = 0; i < n; i++) { h ^= b[i]; h *= 1099511628211ull; }
   return h;
}

static void
npt_trace_hex16(const void *p, uint64_t n, char *out /* >= 33 */)
{
   const uint8_t *b = p; uint64_t m = n < 16 ? n : 16;
   for (uint64_t i = 0; i < m; i++) snprintf(out + i * 2, 3, "%02x", b[i]);
   out[m * 2] = 0;
}

/* Returns true (and fills *d) when resource_id is an A8/R8 Texture2D. */
static bool
npt_trace_is_a8r8(struct npt_context *ctx, uint64_t resource_id, void *resource,
                  D3D11_TEXTURE2D_DESC *d)
{
   if (npt_fmt_trace_count >= 300) return false;
   if (!npt_context_object_is(ctx, resource_id, NPT_OBJECT_TYPE_ID3D11TEXTURE2D))
      return false;
   PFN_ID3D11Texture2D_GetDesc gd =
      NPT_COM_VTBL_FUNC(PFN_ID3D11Texture2D_GetDesc, npt_com_vtable(resource),
                        NPT_VTBL_ID3D11Texture2D_GetDesc);
   if (!gd) return false;
   gd(resource, d);
   return d->Format == 65 || (d->Format >= 60 && d->Format <= 64);
}

static void
npt_trace_unmap_write(struct npt_context *ctx, void *imm_ctx, void *resource,
                      uint64_t resource_id, uint32_t subresource, const char *tag,
                      const void *slot_src, const void *mapped, uint64_t n,
                      uint32_t access_flags)
{
   D3D11_TEXTURE2D_DESC d;
   if (!npt_trace_is_a8r8(ctx, resource_id, resource, &d)) return;
   __atomic_fetch_add(&npt_fmt_trace_count, 1, __ATOMIC_RELAXED);
   char hs[40], hm[40];
   npt_trace_hex16(slot_src, n, hs);
   npt_trace_hex16(mapped, n, hm);
   const uint64_t fs = npt_trace_fnv(slot_src, n), fm = npt_trace_fnv(mapped, n);
   npt_log("FMT unmap-write[%s] res=0x%" PRIx64 " sub=%u fmt=%u %ux%u usage=%u cpu=0x%x "
           "access=0x%x n=%" PRIu64 " shmem_fnv=%016" PRIx64 " shmem16=%s "
           "mapped_fnv=%016" PRIx64 " mapped16=%s %s",
           tag, resource_id, subresource, d.Format, d.Width, d.Height, d.Usage,
           d.CPUAccessFlags, access_flags, n, fs, hs, fm, hm,
           fs == fm ? "MATCH" : "MISMATCH");
   (void)imm_ctx;
}

/* After Unmap: Map(READ) the staging texture again and hash the same region. */
static void
npt_trace_after_unmap(struct npt_context *ctx, void *imm_ctx, void *resource,
                      uint64_t resource_id, uint32_t subresource, uint64_t n,
                      uint64_t expect_fnv)
{
   D3D11_TEXTURE2D_DESC d;
   if (npt_fmt_trace_count >= 300 + 100) return;
   if (!npt_context_object_is(ctx, resource_id, NPT_OBJECT_TYPE_ID3D11TEXTURE2D))
      return;
   PFN_ID3D11Texture2D_GetDesc gd =
      NPT_COM_VTBL_FUNC(PFN_ID3D11Texture2D_GetDesc, npt_com_vtable(resource),
                        NPT_VTBL_ID3D11Texture2D_GetDesc);
   if (!gd) return;
   gd(resource, &d);
   if (!(d.Format == 65 || (d.Format >= 60 && d.Format <= 64))) return;
   if (d.Usage != D3D11_USAGE_STAGING || !(d.CPUAccessFlags & D3D11_CPU_ACCESS_READ)) return;
   PFN_ID3D11DeviceContext_Map map_fn =
      NPT_COM_VTBL_FUNC(PFN_ID3D11DeviceContext_Map, npt_com_vtable(imm_ctx),
                        NPT_VTBL_ID3D11DeviceContext_Map);
   PFN_ID3D11DeviceContext_Unmap unmap_fn =
      NPT_COM_VTBL_FUNC(PFN_ID3D11DeviceContext_Unmap, npt_com_vtable(imm_ctx),
                        NPT_VTBL_ID3D11DeviceContext_Unmap);
   D3D11_MAPPED_SUBRESOURCE m; memset(&m, 0, sizeof(m));
   if (NPT_FAILED(map_fn(imm_ctx, resource, subresource, D3D11_MAP_READ, 0, &m)) || !m.pData) {
      npt_log("FMT unmap-readback res=0x%" PRIx64 " Map(READ) failed", resource_id);
      return;
   }
   char h[40]; npt_trace_hex16(m.pData, n, h);
   const uint64_t f = npt_trace_fnv(m.pData, n);
   npt_log("FMT unmap-readback res=0x%" PRIx64 " sub=%u rowPitch=%u n=%" PRIu64
           " dxvk_fnv=%016" PRIx64 " dxvk16=%s %s", resource_id, subresource,
           m.RowPitch, n, f, h, f == expect_fnv ? "MATCH" : "MISMATCH");
   unmap_fn(imm_ctx, resource, subresource);
}


static HRESULT
npt_resource_unmap_pinned(struct npt_context *ctx,
                          uint64_t context_id, uint64_t resource_id,
                          uint32_t subresource,
                          struct npt_resource *shmem_res,
                          uint32_t shmem_offset, uint64_t byte_size,
                          uint32_t access_flags,
                          uint64_t written_range_begin,
                          uint64_t written_range_end)
{
   void *resource = npt_context_lookup_object(ctx, NULL, resource_id,
                                              NPT_OBJECT_TYPE_IUNKNOWN);
   if (!resource) {
      npt_log("unmap_resource: NULL resource");
      return NPT_E_FAIL;
   }
   if (!npt_resource_family_matches(ctx, resource_id, context_id,
                                    "unmap_resource"))
      return NPT_E_INVALIDARG;

   /* Bounds check the slot window. */
   if ((uint64_t)shmem_offset + byte_size > shmem_res->size) {
      npt_log("unmap_resource: slot window [%u, %u+%" PRIu64 ") "
              "exceeds shmem size %zu",
              shmem_offset, shmem_offset, byte_size,
              shmem_res->size);
      return NPT_E_FAIL;
   }
   const uint8_t *slot_src =
      (const uint8_t *)shmem_res->u.data + shmem_offset;

   if (access_flags) {
      /* Rename-ring path: no prior MAP_RESOURCE; replay the full
       * Map + memcpy + Unmap cycle here. */
      if (!context_id) {
         npt_log("unmap_resource: access_flags path requires context_id");
         return NPT_E_FAIL;
      }

      void *imm_ctx = npt_context_lookup_object(ctx, NULL, context_id,
                                                 NPT_OBJECT_TYPE_ID3D11DEVICECONTEXT);
      if (!imm_ctx) {
         npt_log("unmap_resource: NULL immediate context");
         return NPT_E_FAIL;
      }

      D3D11_MAP d3d11_map_type =
         npt_access_flags_to_d3d11_map(access_flags);

      PFN_ID3D11DeviceContext_Map map_fn =
         NPT_COM_VTBL_FUNC(PFN_ID3D11DeviceContext_Map,
                           npt_com_vtable(imm_ctx),
                           NPT_VTBL_ID3D11DeviceContext_Map);
      PFN_ID3D11DeviceContext_Unmap unmap_fn =
         NPT_COM_VTBL_FUNC(PFN_ID3D11DeviceContext_Unmap,
                           npt_com_vtable(imm_ctx),
                           NPT_VTBL_ID3D11DeviceContext_Unmap);

      D3D11_MAPPED_SUBRESOURCE mapped;
      memset(&mapped, 0, sizeof(mapped));
      HRESULT hr = map_fn(imm_ctx, resource, subresource,
                          d3d11_map_type, 0, &mapped);
      if (NPT_FAILED(hr)) {
         npt_log("unmap_resource: rename-ring Map failed 0x%08x", hr);
         return NPT_E_FAIL;
      }

      if ((access_flags & NPT_MAP_ACCESS_WRITE) && mapped.pData) {
         uint64_t bound = byte_size;
         if ((uint64_t)shmem_offset + bound > shmem_res->size)
            bound = shmem_res->size - shmem_offset;
         memcpy(mapped.pData, slot_src, (size_t)bound);
         if (npt_trace_buf_wanted(ctx, resource_id, resource))
            npt_trace_buf_line("unmap-rename", resource_id, access_flags, bound, slot_src, mapped.pData);
         if (npt_trace_vb_wanted(ctx, resource_id, resource)) {
            npt_log("FMT host vb-unmap-rename res=0x%" PRIx64 " access=0x%x shmem_offset=%u n=%" PRIu64 " (the whole slot is written back)",
                    resource_id, access_flags, shmem_offset, bound);
            npt_trace_buf_line("vb-unmap-rename", resource_id, access_flags, bound, slot_src, mapped.pData);
         }
         npt_trace_unmap_write(ctx, imm_ctx, resource, resource_id, subresource,
                               "rename", slot_src, mapped.pData, bound, access_flags);
      }

      unmap_fn(imm_ctx, resource, subresource);
      return NPT_S_OK;
   }

   /* Paired with a prior MAP_RESOURCE. */
   struct npt_sync_map_entry *entry =
      npt_sync_map_find(ctx, resource_id, subresource);
   if (!entry) {
      npt_log("unmap_resource: resource 0x%" PRIx64 " sub %u not mapped",
              resource_id, subresource);
      return NPT_E_FAIL;
   }

   if (entry->persistent) {
      npt_log("unmap_resource: resource 0x%" PRIx64 " is persistent, "
              "ignoring unmap", resource_id);
      return NPT_E_FAIL;
   }

   uint64_t trace_n = 0, trace_fnv = 0;
   if ((entry->access_flags & NPT_MAP_ACCESS_WRITE) && !entry->external) {
      /* MIN(guest byte_size, recorded mapped_size) so we don't
       * overrun the D3D region with stale/padded SHM bytes.  0 means
       * "use the full mapped_size". */
      uint64_t bound = byte_size ? byte_size : entry->mapped_size;
      if (bound > entry->mapped_size)
         bound = entry->mapped_size;
      memcpy(entry->mapped_data, slot_src, (size_t)bound);
      if (npt_trace_buf_wanted(ctx, resource_id, resource))
         npt_trace_buf_line("unmap-sync", resource_id, entry->access_flags, bound, slot_src, entry->mapped_data);
      npt_trace_unmap_write(ctx, NULL, resource, resource_id, subresource,
                            "sync", slot_src, entry->mapped_data, bound,
                            entry->access_flags);
      trace_n = bound;
      trace_fnv = npt_trace_fnv(slot_src, bound);
   }

   if (!context_id) {
      D3D12_RANGE written_range;
      const D3D12_RANGE *wr = NULL;
      if (written_range_begin != NPT_MAP_RANGE_NULL) {
         written_range.Begin = (SIZE_T)written_range_begin;
         written_range.End = (SIZE_T)written_range_end;
         wr = &written_range;
      }
      PFN_ID3D12Resource_Unmap unmap12 =
         NPT_COM_VTBL_FUNC(PFN_ID3D12Resource_Unmap,
                           npt_com_vtable(resource),
                           NPT_VTBL_ID3D12Resource_Unmap);
      if (!unmap12) {
         npt_log("unmap_resource: no D3D12 Unmap");
         return NPT_E_FAIL;
      }
      unmap12(resource, subresource, wr);

      npt_sync_map_remove(ctx, entry);
      return NPT_S_OK;
   }

   void *imm_ctx = npt_context_lookup_object(ctx, NULL, context_id,
                                              NPT_OBJECT_TYPE_ID3D11DEVICECONTEXT);
   if (!imm_ctx) {
      npt_log("unmap_resource: NULL immediate context");
      return NPT_E_FAIL;
   }

   PFN_ID3D11DeviceContext_Unmap unmap_fn =
      NPT_COM_VTBL_FUNC(PFN_ID3D11DeviceContext_Unmap,
                        npt_com_vtable(imm_ctx),
                        NPT_VTBL_ID3D11DeviceContext_Unmap);
   unmap_fn(imm_ctx, resource, subresource);
   if (trace_n)
      npt_trace_after_unmap(ctx, imm_ctx, resource, resource_id, subresource,
                            trace_n, trace_fnv);

   npt_sync_map_remove(ctx, entry);
   return NPT_S_OK;
}

/* MAP/UNMAP copy through the guest's shmem for the whole call, outside
 * resource_mutex, so the shmem resource is pinned around it. */
HRESULT
npt_resource_map(struct npt_context *ctx,
                 uint64_t context_id, uint64_t resource_id,
                 uint32_t subresource, uint32_t access_flags,
                 uint32_t api_map_flags, uint32_t shmem_res_id,
                 uint64_t read_range_begin,
                 uint64_t read_range_end,
                 uint64_t byte_size,
                 uint32_t mip_rows, uint32_t mip_depth,
                 uint32_t shmem_offset,
                 uint32_t *out_row_pitch, uint32_t *out_depth_pitch,
                 uint32_t *out_mapped_size, uint32_t *out_external_cookie)
{
   *out_row_pitch = 0;
   *out_depth_pitch = 0;
   *out_mapped_size = 0;
   *out_external_cookie = NPT_EXTERNAL_COOKIE_NONE;

   struct npt_resource *shmem_res =
      npt_context_pin_shm_resource(ctx, shmem_res_id);
   if (!shmem_res) {
      npt_log("map_resource: invalid SHM resource %u", shmem_res_id);
      return NPT_E_FAIL;
   }
   const HRESULT hr = npt_resource_map_pinned(
      ctx, context_id, resource_id, subresource, access_flags,
      api_map_flags, shmem_res, read_range_begin, read_range_end,
      byte_size, mip_rows, mip_depth, shmem_offset,
      out_row_pitch, out_depth_pitch, out_mapped_size,
      out_external_cookie);
   npt_context_unpin_resource(ctx, shmem_res);
   return hr;
}

HRESULT
npt_resource_bind_d3d11_buffer_shmem(struct npt_context *ctx,
                                     uint64_t buffer_id,
                                     uint32_t shmem_res_id,
                                     uint64_t byte_size,
                                     uint32_t cookie, uint32_t reserved)
{
   const struct npt_d3d_library *lib = npt_renderer_get_library();
   if (!lib || !lib->pfn_d3d11_buffer_bind_external_fd)
      return NPT_E_NOTIMPL;
   if (!byte_size || cookie == NPT_EXTERNAL_COOKIE_NONE || reserved)
      return NPT_E_INVALIDARG;

   void *buffer = npt_context_lookup_object(ctx, NULL, buffer_id,
                                            NPT_OBJECT_TYPE_ID3D11BUFFER);
   if (!buffer) {
      npt_log("bind_d3d11_buffer_shmem: NULL buffer");
      return NPT_E_INVALIDARG;
   }

   /* The backend dups the fd and maps it itself, so the pin only has to
    * cover the call. */
   struct npt_resource *shmem =
      npt_context_pin_shm_resource(ctx, shmem_res_id);
   if (!shmem) {
      npt_log("bind_d3d11_buffer_shmem: invalid SHM resource %u",
              shmem_res_id);
      return NPT_E_INVALIDARG;
   }
   HRESULT hr;
   if (shmem->u.fd < 0 || byte_size > shmem->size) {
      npt_log("bind_d3d11_buffer_shmem: res %u (fd %d, %zu bytes) cannot "
              "back %" PRIu64 " bytes", shmem_res_id, shmem->u.fd,
              shmem->size, byte_size);
      hr = NPT_E_INVALIDARG;
   } else {
#ifdef __linux__
      /* dxvk imports a dmabuf: the blob's own, or a udmabuf of its
       * memfd. */
      const int fd = npt_hostmem_blob_dmabuf(shmem, 0, shmem->size,
                                             "bind_d3d11_buffer_shmem");
#else
      const int fd = shmem->u.fd;
#endif
      if (fd < 0) {
         hr = NPT_E_FAIL;
      } else {
         hr = (HRESULT)lib->pfn_d3d11_buffer_bind_external_fd(
            buffer, fd, shmem->size, cookie);
#ifdef __linux__
         close(fd);
#endif
      }
   }
   npt_context_unpin_resource(ctx, shmem);
   return hr;
}

HRESULT
npt_resource_unmap(struct npt_context *ctx,
                   uint64_t context_id, uint64_t resource_id,
                   uint32_t subresource, uint32_t shmem_res_id,
                   uint32_t shmem_offset, uint64_t byte_size,
                   uint32_t access_flags,
                   uint64_t written_range_begin,
                   uint64_t written_range_end)
{
   struct npt_resource *shmem_res =
      npt_context_pin_shm_resource(ctx, shmem_res_id);
   if (!shmem_res) {
      npt_log("unmap_resource: invalid SHM resource %u", shmem_res_id);
      return NPT_E_FAIL;
   }
   const HRESULT hr = npt_resource_unmap_pinned(
      ctx, context_id, resource_id, subresource, shmem_res, shmem_offset,
      byte_size, access_flags, written_range_begin, written_range_end);
   npt_context_unpin_resource(ctx, shmem_res);
   return hr;
}
