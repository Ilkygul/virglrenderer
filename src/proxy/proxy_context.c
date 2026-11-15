/*
 * Copyright 2021 Google LLC
 * SPDX-License-Identifier: MIT
 */

#include "proxy_context.h"

#include <fcntl.h>
#include <poll.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <unistd.h>

#include "virtgpu_drm.h"

#include "server/render_protocol.h"
#include "util/anon_file.h"
#include "util/bitscan.h"
#include "virgl_fence.h"
#include "virgl_shm_pool.h"

#include "proxy_client.h"

struct proxy_fence {
   uint32_t flags;
   uint32_t seqno;
   uint64_t fence_id;
   struct list_head head;
};

/* DroidVM: resources whose blob_id==0 shmem lives in the venus_host pool carry
 * their pool slice here (the table value; NULL for everything else), so the
 * slice goes back to the pool when the resource leaves the context. */
struct proxy_pool_alloc {
   void *ptr;
   uint64_t size;
};

static void
proxy_pool_alloc_release(struct proxy_pool_alloc *pa)
{
   if (!pa)
      return;
   virgl_shm_pool_free(pa->ptr, pa->size);
   free(pa);
}

static inline void
proxy_context_resource_add_data(struct proxy_context *ctx,
                                uint32_t res_id,
                                struct proxy_pool_alloc *pa)
{
   assert(!_mesa_hash_table_search(ctx->resource_table, (void *)(uintptr_t)res_id));
   _mesa_hash_table_insert(ctx->resource_table, (void *)(uintptr_t)res_id, pa);
}

static inline void
proxy_context_resource_add(struct proxy_context *ctx, uint32_t res_id)
{
   proxy_context_resource_add_data(ctx, res_id, NULL);
}

static inline bool
proxy_context_resource_find(struct proxy_context *ctx, uint32_t res_id)
{
   return _mesa_hash_table_search(ctx->resource_table, (void *)(uintptr_t)res_id);
}

static inline void
proxy_context_resource_remove(struct proxy_context *ctx, uint32_t res_id)
{
   struct hash_entry *entry =
      _mesa_hash_table_search(ctx->resource_table, (void *)(uintptr_t)res_id);
   if (!entry)
      return;
   proxy_pool_alloc_release(entry->data);
   _mesa_hash_table_remove(ctx->resource_table, entry);
}

static inline bool
proxy_context_resource_table_init(struct proxy_context *ctx)
{
   ctx->resource_table = _mesa_hash_table_create_u32_keys(NULL);
   return ctx->resource_table;
}

static inline void
proxy_context_resource_table_fini(struct proxy_context *ctx)
{
   hash_table_foreach (ctx->resource_table, entry)
      proxy_pool_alloc_release(entry->data);
   _mesa_hash_table_destroy(ctx->resource_table, NULL);
}

static bool
proxy_fence_is_signaled(const struct proxy_fence *fence, uint32_t cur_seqno)
{
   /* takes wrapping into account */
   const uint32_t d = cur_seqno - fence->seqno;
   return d < INT32_MAX;
}

static struct proxy_fence *
proxy_context_alloc_fence(struct proxy_context *ctx)
{
   struct proxy_fence *fence = NULL;

   if (proxy_renderer.flags & VIRGL_RENDERER_ASYNC_FENCE_CB)
      mtx_lock(&ctx->free_fences_mutex);

   if (!list_is_empty(&ctx->free_fences)) {
      fence = list_first_entry(&ctx->free_fences, struct proxy_fence, head);
      list_del(&fence->head);
   }

   if (proxy_renderer.flags & VIRGL_RENDERER_ASYNC_FENCE_CB)
      mtx_unlock(&ctx->free_fences_mutex);

   return fence ? fence : malloc(sizeof(*fence));
}

static void
proxy_context_free_fence(struct proxy_context *ctx, struct proxy_fence *fence)
{
   if (proxy_renderer.flags & VIRGL_RENDERER_ASYNC_FENCE_CB)
      mtx_lock(&ctx->free_fences_mutex);

   list_add(&fence->head, &ctx->free_fences);

   if (proxy_renderer.flags & VIRGL_RENDERER_ASYNC_FENCE_CB)
      mtx_unlock(&ctx->free_fences_mutex);
}

static uint32_t
proxy_context_load_timeline_seqno(struct proxy_context *ctx, uint32_t ring_idx)
{
   return atomic_load(&ctx->timeline_seqnos[ring_idx]);
}

static bool
proxy_context_retire_timeline_fences_locked(struct proxy_context *ctx,
                                            uint32_t ring_idx,
                                            uint32_t cur_seqno)
{
   struct proxy_timeline *timeline = &ctx->timelines[ring_idx];
   bool force_retire_all = false;

   /* check if the socket has been disconnected (i.e., the other end has
    * crashed) if no progress is made after a while
    */
   if (timeline->cur_seqno == cur_seqno && !list_is_empty(&timeline->fences)) {
      timeline->cur_seqno_stall_count++;
      if (timeline->cur_seqno_stall_count < 100 ||
          proxy_socket_is_connected(&ctx->socket))
         return false;

      /* A disconnected server is never positive GPU-job completion.
       * Hold this exact ring; caller's failstop/VM lifecycle owns recovery. */
      if (ctx->base.gpu_jobs_enabled && ring_idx == 0)
         return false;
      /* Legacy contexts/rings retain their existing disconnect policy. */
      force_retire_all = true;
   }

   timeline->cur_seqno = cur_seqno;
   timeline->cur_seqno_stall_count = 0;

   list_for_each_entry_safe (struct proxy_fence, fence, &timeline->fences, head) {
      if (!proxy_fence_is_signaled(fence, timeline->cur_seqno) && !force_retire_all)
         return false;

      ctx->base.fence_retire(&ctx->base, ring_idx, fence->fence_id);

      /* The fence is done; drop the fd submit_fence registered for it (see
       * virgl_fence_retire).  Retirement here is driven by the timeline, not
       * by the fd, so nothing else will ever drop it -- and force_retire_all
       * retires precisely the fences whose fd is never going to signal. */
      virgl_fence_retire(fence->fence_id);

      list_del(&fence->head);
      proxy_context_free_fence(ctx, fence);
   }

   return true;
}

static void
proxy_context_retire_fences_internal(struct proxy_context *ctx)
{
   if (ctx->sync_thread.fence_eventfd >= 0)
      flush_eventfd(ctx->sync_thread.fence_eventfd);

   if (proxy_renderer.flags & VIRGL_RENDERER_ASYNC_FENCE_CB)
      mtx_lock(&ctx->timeline_mutex);

   uint64_t new_busy_mask = 0;
   uint64_t old_busy_mask = ctx->timeline_busy_mask;
   while (old_busy_mask) {
      const uint32_t ring_idx = u_bit_scan64(&old_busy_mask);
      const uint32_t cur_seqno = proxy_context_load_timeline_seqno(ctx, ring_idx);
      if (!proxy_context_retire_timeline_fences_locked(ctx, ring_idx, cur_seqno))
         new_busy_mask |= 1ull << ring_idx;
   }

   ctx->timeline_busy_mask = new_busy_mask;

   if (proxy_renderer.flags & VIRGL_RENDERER_ASYNC_FENCE_CB)
      mtx_unlock(&ctx->timeline_mutex);
}

static int
proxy_context_sync_thread(void *arg)
{
   struct proxy_context *ctx = arg;
   struct pollfd poll_fds[2] = {
      [0] = {
         .fd = ctx->sync_thread.fence_eventfd,
         .events = POLLIN,
      },
      [1] = {
         .fd = ctx->socket.fd,
      },
   };

   assert(proxy_renderer.flags & VIRGL_RENDERER_ASYNC_FENCE_CB);

   while (!ctx->sync_thread.stop) {
      const int ret = poll(poll_fds, ARRAY_SIZE(poll_fds), -1);
      if (ret <= 0) {
         if (ret < 0 && (errno == EINTR || errno == EAGAIN))
            continue;

         proxy_log("failed to poll fence eventfd");
         break;
      }

      proxy_context_retire_fences_internal(ctx);
   }

   return 0;
}

/* Out-of-band fence routing.  Only Neptune event-ring fences qualify:
 * their retirement pairs with an ARM that travels independently (the host
 * parks whichever side arrives first), so registering them ahead of queued
 * submits is legal -- and registering them promptly is the point, since on
 * the main socket a fence queues behind the dispatch thread's in-flight
 * submits and every present-gate wake lags that backlog.  Low rings retire
 * in dispatch order ("everything before me was consumed") and have to stay
 * on the ordered main socket.  The ring split mirrors the host's
 * NPT_EVENT_RING_BASE convention: the upper half of the timeline range. */
static bool
proxy_context_fence_ring_is_oob(const struct proxy_context *ctx,
                                uint32_t ring_idx)
{
   return ctx->capset_id == VIRTGPU_DRM_CAPSET_NEPTUNE &&
          ring_idx >= PROXY_CONTEXT_TIMELINE_COUNT / 2;
}

static bool
proxy_context_fence_socket_attach(struct proxy_context *ctx)
{
   int fds[2];
   if (socketpair(AF_UNIX, SOCK_STREAM, 0, fds)) {
      proxy_log("fence channel: socketpair failed");
      return false;
   }
   for (int i = 0; i < 2; i++) {
      const int flags = fcntl(fds[i], F_GETFD);
      if (flags >= 0)
         fcntl(fds[i], F_SETFD, flags | FD_CLOEXEC);
   }

   const struct render_context_op_attach_fence_socket_request req = {
      .header.op = RENDER_CONTEXT_OP_ATTACH_FENCE_SOCKET,
   };
   if (!proxy_socket_send_request_with_fds(&ctx->socket, &req, sizeof(req),
                                           &fds[1], 1)) {
      proxy_log("fence channel: attach failed");
      close(fds[0]);
      close(fds[1]);
      return false;
   }
   close(fds[1]);
   ctx->fence_socket_fd = fds[0];
   proxy_log("fence channel attached for ctx %d (event-ring fences "
             "out-of-band)", ctx->base.ctx_id);
   return true;
}

static bool
proxy_context_fence_oob_submit(struct proxy_context *ctx,
                               uint32_t flags,
                               uint32_t ring_idx,
                               uint32_t seqno)
{
   if (ctx->fence_socket_fd == -1 && !proxy_context_fence_socket_attach(ctx))
      ctx->fence_socket_fd = -2;
   if (ctx->fence_socket_fd < 0)
      return false;

   const struct render_context_op_submit_fence_request req = {
      .header.op = RENDER_CONTEXT_OP_SUBMIT_FENCE,
      .flags = flags,
      .ring_index = ring_idx,
      .seqno = seqno,
   };
   const uint8_t *cur = (const uint8_t *)&req;
   size_t left = sizeof(req);
   while (left) {
      const ssize_t s = send(ctx->fence_socket_fd, cur, left, MSG_NOSIGNAL);
      if (s < 0) {
         if (errno == EINTR || errno == EAGAIN)
            continue;
         proxy_log("fence channel: send failed (errno %d); falling back to "
                   "the main socket", errno);
         close(ctx->fence_socket_fd);
         ctx->fence_socket_fd = -2;
         return false;
      }
      cur += s;
      left -= (size_t)s;
   }
   return true;
}

static uint32_t
proxy_context_enable_gpu_jobs(struct virgl_context *base)
{
   struct proxy_context *ctx = (struct proxy_context *)base;
   static atomic_uint generations = 1;
   if (ctx->capset_id != VIRTGPU_DRM_CAPSET_NEPTUNE || ctx->gpu_jobs_poisoned)
      return 0;
   if (base->gpu_jobs_enabled)
      return VIRGL_RENDERER_NATIVE_JOB_CAP_MAGIC;
   uint32_t generation = atomic_fetch_add_explicit(&generations, 1, memory_order_relaxed);
   if (!generation)
      generation = atomic_fetch_add_explicit(&generations, 1, memory_order_relaxed);
   const struct render_context_op_enable_gpu_jobs_request req = {
      .header.op = RENDER_CONTEXT_OP_ENABLE_GPU_JOBS,
      .version = 0,
      .ctx_id = base->ctx_id,
      .generation = generation,
   };
   struct render_context_op_enable_gpu_jobs_reply reply = { 0 };
   if (!proxy_socket_send_request(&ctx->socket, &req, sizeof(req)) ||
       !proxy_socket_receive_reply(&ctx->socket, &reply, sizeof(reply)) ||
       reply.version != 0 || reply.ctx_id != base->ctx_id ||
       reply.generation != generation ||
       reply.magic != VIRGL_RENDERER_NATIVE_JOB_CAP_MAGIC ||
       reply.caps != VIRGL_RENDERER_NATIVE_JOB_CAP_GPU_RETIRE ||
       reply.reserved[0] || reply.reserved[1] || reply.reserved[2])
      return 0;
   /* First validated REQ invokes this CAP lazily. Parent protection ACK
    * precedes any PREPARE/GPU work; old parent cannot advertise support. */
   if (!proxy_client_protect_context(ctx->client, base->ctx_id, generation)) {
      ctx->gpu_jobs_poisoned = true;
      return 0;
   }
   ctx->gpu_jobs_generation = generation;
   base->gpu_jobs_enabled = true;
   return reply.magic;
}

static int
proxy_context_prepare_gpu_job(struct virgl_context *base, uint64_t front_fence_id)
{
   struct proxy_context *ctx = (struct proxy_context *)base;
   if (ctx->capset_id != VIRTGPU_DRM_CAPSET_NEPTUNE || !base->gpu_jobs_enabled ||
       !ctx->gpu_jobs_generation || ctx->gpu_jobs_poisoned || ctx->gpu_job_prepared) {
      ctx->gpu_jobs_poisoned = true;
      return -EINVAL;
   }
   uint32_t seqno = ctx->timelines[0].next_seqno++;
   if (!seqno)
      seqno = ctx->timelines[0].next_seqno++;
   const struct render_context_op_prepare_gpu_job_request req = {
      .header.op = RENDER_CONTEXT_OP_PREPARE_GPU_JOB,
      .ctx_id = base->ctx_id, .generation = ctx->gpu_jobs_generation,
      .seqno = seqno, .front_fence_id = front_fence_id,
   };
   struct render_context_op_prepare_gpu_job_reply reply = { 0 };
   if (!proxy_socket_send_request(&ctx->socket, &req, sizeof(req)) ||
       !proxy_socket_receive_reply(&ctx->socket, &reply, sizeof(reply)) ||
       reply.ctx_id != base->ctx_id || reply.generation != ctx->gpu_jobs_generation ||
       reply.seqno != seqno || reply.front_fence_id != front_fence_id ||
       reply.magic != VIRGL_RENDERER_NATIVE_JOB_CAP_MAGIC ||
       reply.reserved[0] || reply.reserved[1]) {
      /* Native may already have applied PREPARE. Never attach it to another
       * submit/fence after a lost ACK. Keep enabled to suppress fake retire. */
      ctx->gpu_jobs_poisoned = true;
      return -EIO;
   }
   ctx->gpu_job_prepared = true;
   ctx->gpu_job_front_fence_id = front_fence_id;
   ctx->gpu_job_seqno = seqno;
   return 0;
}

static int
proxy_context_submit_fence(struct virgl_context *base,
                           uint32_t flags,
                           uint32_t ring_idx,
                           uint64_t fence_id)
{
   struct proxy_context *ctx = (struct proxy_context *)base;
   const uint64_t old_busy_mask = ctx->timeline_busy_mask;

   if (ring_idx >= PROXY_CONTEXT_TIMELINE_COUNT)
      return -EINVAL;
   if ((flags & VIRGL_RENDERER_FENCE_FLAG_GPU_JOB_COMPLETE) &&
       (ctx->capset_id != VIRTGPU_DRM_CAPSET_NEPTUNE || ring_idx != 0 ||
        !base->gpu_jobs_enabled || !ctx->gpu_jobs_generation))
      return -EINVAL;

   const bool gpu_job = flags & VIRGL_RENDERER_FENCE_FLAG_GPU_JOB_COMPLETE;
   if (ring_idx == 0 && (ctx->gpu_jobs_poisoned ||
       (gpu_job && (!ctx->gpu_job_prepared || ctx->gpu_job_front_fence_id != fence_id)) ||
       (!gpu_job && ctx->gpu_job_prepared))) {
      ctx->gpu_jobs_poisoned = true;
      return -EINVAL;
   }
   struct proxy_timeline *timeline = &ctx->timelines[ring_idx];
   struct proxy_fence *fence = proxy_context_alloc_fence(ctx);
   if (!fence)
      return -ENOMEM;

   fence->flags = flags;
   fence->seqno = gpu_job ? ctx->gpu_job_seqno : timeline->next_seqno++;
   fence->fence_id = fence_id;

   if (proxy_renderer.flags & VIRGL_RENDERER_ASYNC_FENCE_CB)
      mtx_lock(&ctx->timeline_mutex);

   list_addtail(&fence->head, &timeline->fences);
   ctx->timeline_busy_mask |= 1ull << ring_idx;

   if (proxy_renderer.flags & VIRGL_RENDERER_ASYNC_FENCE_CB)
      mtx_unlock(&ctx->timeline_mutex);

   /* Event-ring fences take the out-of-band channel: registered by the
    * server's dedicated fence thread instead of queueing behind the
    * dispatch thread's in-flight submits, and the QEMU virtio thread
    * pays no reply round trip.  Any channel failure falls through to
    * the ordered synchronous path below. */
   if (proxy_context_fence_ring_is_oob(ctx, ring_idx) &&
       proxy_context_fence_oob_submit(ctx, flags, ring_idx, fence->seqno))
      return 0;

   const struct render_context_op_submit_fence_request req = {
      .header.op = RENDER_CONTEXT_OP_SUBMIT_FENCE,
      .flags = flags,
      .ring_index = ring_idx,
      .seqno = fence->seqno,
   };
   if (proxy_socket_send_request(&ctx->socket, &req, sizeof(req))) {
      if (gpu_job)
         ctx->gpu_job_prepared = false;
      return 0;
   }
   if (ctx->base.gpu_jobs_enabled && ring_idx == 0)
      ctx->gpu_jobs_poisoned = true;

   /* recover timeline fences and busy_mask on submit_fence request failure */
   if (proxy_renderer.flags & VIRGL_RENDERER_ASYNC_FENCE_CB)
      mtx_lock(&ctx->timeline_mutex);

   list_del(&fence->head);
   ctx->timeline_busy_mask = old_busy_mask;

   if (proxy_renderer.flags & VIRGL_RENDERER_ASYNC_FENCE_CB)
      mtx_unlock(&ctx->timeline_mutex);

   proxy_context_free_fence(ctx, fence);
   proxy_log("failed to submit fence");
   return -1;
}

static void
proxy_context_retire_fences(struct virgl_context *base)
{
   struct proxy_context *ctx = (struct proxy_context *)base;

   assert(!(proxy_renderer.flags & VIRGL_RENDERER_ASYNC_FENCE_CB));
   proxy_context_retire_fences_internal(ctx);
}

static int
proxy_context_get_fencing_fd(struct virgl_context *base)
{
   struct proxy_context *ctx = (struct proxy_context *)base;

   assert(!(proxy_renderer.flags & VIRGL_RENDERER_ASYNC_FENCE_CB));
   return ctx->sync_thread.fence_eventfd;
}

static int
proxy_context_submit_cmd(struct virgl_context *base, const void *buffer, size_t size)
{
   struct proxy_context *ctx = (struct proxy_context *)base;

   if (ctx->gpu_jobs_poisoned)
      return -EIO;
   if (!size)
      return 0;

   struct render_context_op_submit_cmd_request req = {
      .header.op = RENDER_CONTEXT_OP_SUBMIT_CMD,
      .size = size,
   };

   const size_t inlined = MIN2(size, sizeof(req.cmd));
   memcpy(req.cmd, buffer, inlined);

   if (!proxy_socket_send_request(&ctx->socket, &req, sizeof(req))) {
      proxy_log("failed to submit cmd");
      if (ctx->base.gpu_jobs_enabled) ctx->gpu_jobs_poisoned = true;
      return -1;
   }

   if (size > inlined) {
      if (!proxy_socket_send_request(&ctx->socket, (const char *)buffer + inlined,
                                     size - inlined)) {
         proxy_log("failed to submit large cmd buffer");
         if (ctx->base.gpu_jobs_enabled) ctx->gpu_jobs_poisoned = true;
         return -1;
      }
   }

   return 0;
}

static bool
validate_resource_fd_shm(int fd, uint64_t expected_size)
{
#if defined(HAVE_MEMFD_CREATE)
   static const int blocked_seals = F_SEAL_WRITE;

   const int seals = fcntl(fd, F_GET_SEALS);
   if (seals & blocked_seals) {
      proxy_log("failed to validate shm seals(%d): blocked(%d)", seals, blocked_seals);
      return false;
   }
#endif

   struct stat st;
   if (fstat(fd, &st) < 0) {
      proxy_log("failed to fstat shm fd");
      return false;
   }
   if ((uint64_t)st.st_size < expected_size) {
      proxy_log("shm size(%" PRIu64 ") smaller than expected(%" PRIu64 ")",
                (uint64_t)st.st_size, expected_size);
      return false;
   }

   return true;
}

static inline int
add_required_seals_to_fd(int fd)
{
#if defined(HAVE_MEMFD_CREATE)
   return fcntl(fd, F_ADD_SEALS, F_SEAL_SEAL | F_SEAL_SHRINK | F_SEAL_GROW);
#else
   // if memfd_create is not available pretend file sealing worked
   (void)fd;
   return 0;
#endif
}

static int
proxy_context_get_blob(struct virgl_context *base,
                       uint32_t res_id,
                       uint64_t blob_id,
                       uint64_t blob_size,
                       uint32_t blob_flags,
                       struct virgl_context_blob *blob)
{
   /* RENDER_CONTEXT_OP_CREATE_RESOURCE implies resource attach, thus proxy tracks
    * resources created here to avoid double attaching the same resource when proxy is on
    * attach_resource callback.
    */
   struct proxy_context *ctx = (struct proxy_context *)base;

   /* DroidVM: Neptune blob_id==0 shmems (rings, ~2 MiB) come out of the
    * boot-shared venus_host pool, exactly like Venus transport shmems.  This
    * process (the VMM) sub-allocates the slice, so the blob's map_ptr is a VA
    * inside VENUS_POOL_HOST_VA and the VMM answers the guest's map with
    * MAP_INFO_POOL + offset -- no runtime registration, which times out on
    * Gunyah.  The render server gets a dup of the pool fd plus the file
    * offset and maps that window instead of creating a memfd.  Any failure
    * (no pool, exhausted, old server reply) falls back to the server memfd. */
   struct proxy_pool_alloc *pool = NULL;
   uint64_t pool_off = 0;
   int pool_fd = -1;
   if (blob_id == 0 && ctx->capset_id == VIRTGPU_DRM_CAPSET_NEPTUNE &&
       virgl_shm_pool_available()) {
      const uint64_t page_size = getpagesize();
      const uint64_t alloc_size = (blob_size + page_size - 1) & ~(page_size - 1);
      void *pool_ptr = virgl_shm_pool_alloc(alloc_size, &pool_off);
      pool_fd = pool_ptr ? virgl_shm_pool_dup_fd() : -1;
      pool = pool_fd >= 0 ? malloc(sizeof(*pool)) : NULL;
      if (pool) {
         pool->ptr = pool_ptr;
         pool->size = alloc_size;
      } else if (pool_ptr) {
         virgl_shm_pool_free(pool_ptr, alloc_size);
      }
      if (!pool && pool_fd >= 0) {
         close(pool_fd);
         pool_fd = -1;
      }
   }

   struct render_context_op_create_resource_request req = {
      .header.op = RENDER_CONTEXT_OP_CREATE_RESOURCE,
      .res_id = res_id,
      .blob_id = blob_id,
      .blob_size = blob_size,
      .blob_flags = blob_flags,
   };
   if (pool) {
      req.has_shm_fd = 1;
      req.shm_fd_offset = virgl_shm_pool_fd_offset() + pool_off;
   }

   struct render_context_op_create_resource_reply reply;
   int reply_fd;
   int reply_fd_count;
   for (;;) {
      const bool sent =
         pool ? proxy_socket_send_request_with_fds(&ctx->socket, &req, sizeof(req),
                                                   &pool_fd, 1)
              : proxy_socket_send_request(&ctx->socket, &req, sizeof(req));
      if (pool_fd >= 0) {
         close(pool_fd);
         pool_fd = -1;
      }
      if (!sent) {
         proxy_log("failed to get blob %" PRIu64, blob_id);
         proxy_pool_alloc_release(pool);
         return -1;
      }

      if (!proxy_socket_receive_reply_with_fds(&ctx->socket, &reply, sizeof(reply),
                                               &reply_fd, 1, &reply_fd_count)) {
         proxy_log("failed to get reply of blob %" PRIu64, blob_id);
         proxy_pool_alloc_release(pool);
         return -1;
      }

      if (reply_fd_count || !pool)
         break;

      /* the server refused the pool window; retry with its own memfd */
      proxy_log("server rejected pool shmem for res %u, falling back to memfd", res_id);
      proxy_pool_alloc_release(pool);
      pool = NULL;
      req.has_shm_fd = 0;
      req.shm_fd_offset = 0;
   }

   if (!reply_fd_count) {
      proxy_log("invalid reply for blob %" PRIu64, blob_id);
      return -1;
   }

   if (pool && reply.fd_type != VIRGL_RESOURCE_FD_SHM) {
      proxy_log("pool shmem for res %u came back as fd_type %d", res_id, reply.fd_type);
      close(reply_fd);
      proxy_pool_alloc_release(pool);
      return -1;
   }
   if (pool) {
      /* The reply fd is the server's dup of the pool fd; use a dup of our own
       * (as Venus does) and hand crosvm the pool VA as map_ptr so
       * virgl_pool_offset() sees a pool-resident resource.  The pool fd is
       * the VMM's shared region, so no seal/size validation applies. */
      close(reply_fd);
      const int res_fd = virgl_shm_pool_dup_fd();
      if (res_fd < 0) {
         proxy_log("failed to dup pool fd for res %u", res_id);
         proxy_pool_alloc_release(pool);
         return -1;
      }
      blob->type = VIRGL_RESOURCE_FD_SHM;
      blob->u.fd = res_fd;
      blob->map_info = VIRGL_RENDERER_MAP_CACHE_CACHED;
      blob->map_ptr = pool->ptr;
      blob->fd_offset = req.shm_fd_offset;
      blob->export_format = reply.export_format;
      proxy_log("res %u (%" PRIu64 " bytes) in venus_host pool: off 0x%" PRIx64
                " va %p", res_id, blob_size, pool_off, pool->ptr);
      proxy_context_resource_add_data(ctx, res_id, pool);
      return 0;
   }

   bool reply_fd_valid = false;
   switch (reply.fd_type) {
   case VIRGL_RESOURCE_FD_DMABUF:
      /* TODO validate the fd is dmabuf >= blob_size */
      reply_fd_valid = true;
      break;
   case VIRGL_RESOURCE_FD_OPAQUE:
      /* this will be validated when imported by the client */
      reply_fd_valid = true;
      break;
   case VIRGL_RESOURCE_FD_SHM:
      /* validate the seals and size here */
      reply_fd_valid = !add_required_seals_to_fd(reply_fd) &&
                       validate_resource_fd_shm(reply_fd, blob_size);
      break;
   default:
      break;
   }
   if (!reply_fd_valid) {
      proxy_log("invalid fd type %d for blob %" PRIu64, reply.fd_type, blob_id);
      close(reply_fd);
      return -1;
   }

   blob->type = reply.fd_type;
   blob->u.fd = reply_fd;
   blob->map_info = reply.map_info;

   if (reply.fd_type == VIRGL_RESOURCE_FD_OPAQUE)
      blob->vulkan_info = reply.vulkan_info;

   blob->export_format = reply.export_format;

   proxy_context_resource_add(ctx, res_id);

   return 0;
}

static int
proxy_context_transfer_3d(struct virgl_context *base,
                          struct virgl_resource *res,
                          UNUSED const struct vrend_transfer_info *info,
                          UNUSED int transfer_mode)
{
   struct proxy_context *ctx = (struct proxy_context *)base;

   proxy_log("no transfer support for ctx %d and res %d", ctx->base.ctx_id, res->res_id);
   return -1;
}

static void
proxy_context_detach_resource(struct virgl_context *base, struct virgl_resource *res)
{
   struct proxy_context *ctx = (struct proxy_context *)base;
   const uint32_t res_id = res->res_id;

   /* avoid detaching resource not belonging to this context */
   if (!proxy_context_resource_find(ctx, res_id))
      return;

   const struct render_context_op_destroy_resource_request req = {
      .header.op = RENDER_CONTEXT_OP_DESTROY_RESOURCE,
      .res_id = res_id,
   };
   if (!proxy_socket_send_request(&ctx->socket, &req, sizeof(req)))
      proxy_log("failed to detach res %d", res_id);

   proxy_context_resource_remove(ctx, res_id);
}

static bool
proxy_context_export_classic_layout(struct virgl_resource *res, int *fd,
                                    struct virgl_import_layout_v1 *layout)
{
   struct virgl_renderer_export_query q = {0};
   q.hdr.stype = VIRGL_RENDERER_STRUCTURE_TYPE_EXPORT_QUERY;
   q.hdr.stype_version = 0; q.hdr.size = sizeof(q);
   q.in_resource_id = res->res_id; q.in_export_fds = 1;
   for (unsigned i=0;i<4;i++) q.out_fds[i] = -1;
   const int ret = virgl_renderer_execute(&q, sizeof(q));
   bool ok = ret == 0 && q.out_num_fds == 1 && q.out_fds[0] >= 0;
   for (unsigned i=1;i<4;i++)
      if (q.out_fds[i] >= 0 || q.out_strides[i] || q.out_offsets[i]) ok = false;
   const off_t size = ok ? lseek(q.out_fds[0], 0, SEEK_END) : -1;
   if (size <= 0) ok = false;
   if (ok) {
      *layout = (struct virgl_import_layout_v1){
         .magic=VIRGL_IMPORT_LAYOUT_MAGIC, .version=1, .size=sizeof(*layout),
         .flags=VIRGL_IMPORT_LAYOUT_CLASSIC, .fourcc=q.out_fourcc, .plane_count=1,
         .modifier=q.out_modifier, .allocation_size=(uint64_t)size,
         .planes={{q.out_offsets[0], q.out_strides[0]}},
      };
      ok = virgl_import_layout_valid(layout);
   }
   if (ok) { *fd=q.out_fds[0]; q.out_fds[0]=-1; }
   for (unsigned i=0;i<4;i++) if (q.out_fds[i]>=0) close(q.out_fds[i]);
   return ok;
}

static void
proxy_context_attach_resource(struct virgl_context *base, struct virgl_resource *res)
{
   struct proxy_context *ctx = (struct proxy_context *)base;
   const uint32_t res_id = res->res_id;

   /* avoid importing resources created from RENDER_CONTEXT_OP_CREATE_RESOURCE */
   if (proxy_context_resource_find(ctx, res_id))
      return;

   /* The current render protocol only supports importing dma-buf, shm or pipe resource
    * that can be exported to dma-buf. A protocol change is needed when there exists use
    * case for importing external Vulkan opaque resource.
    */
   if (res->fd_type != VIRGL_RESOURCE_FD_INVALID &&
       res->fd_type != VIRGL_RESOURCE_FD_DMABUF &&
       res->fd_type != VIRGL_RESOURCE_FD_SHM) {
      proxy_log("failed to attach res %d with fd_type %d", res_id, res->fd_type);
      return;
   }

   enum virgl_resource_fd_type res_fd_type = res->fd_type;
   int res_fd = res->fd;
   uint64_t res_size = res->map_size;
   bool close_res_fd = false;
   struct virgl_import_layout_v1 layout = {0};
   /* A retained FD may already carry native blob metadata. Preserve that
    * exact baseline forwarding path; never replace it with a new GL export. */
   const bool classic_layout = res->fd_type == VIRGL_RESOURCE_FD_INVALID &&
      res->pipe_resource && ctx->capset_id == VIRTGPU_DRM_CAPSET_NEPTUNE;
   if (classic_layout) {
      if (!proxy_context_export_classic_layout(res, &res_fd, &layout)) {
         proxy_log("classic res %u has no supported authoritative export layout; refusing guessed import", res_id);
         return;
      }
      res_fd_type = VIRGL_RESOURCE_FD_DMABUF;
      res_size = layout.allocation_size;
      close_res_fd = true;
   } else if (res_fd_type == VIRGL_RESOURCE_FD_INVALID) {
      /* importable pipe resouce can only export as dma-buf */
      res_fd_type = virgl_resource_export_fd(res, &res_fd);
      if (res_fd_type != VIRGL_RESOURCE_FD_DMABUF) {
         /* close fd for unexpected fd type from succeeded export */
         if (res_fd_type != VIRGL_RESOURCE_FD_INVALID)
            close(res_fd);
         proxy_log("exported res %d to unexpected fd_type %d", res_id, res_fd_type);
         return;
      }

      /* get the actual dma-buf size here because:
       * - pipe resource created by vrend has a zero map_size
       * - blob resource created by vrend can have non-zero fake map_size
       */
      res_size = lseek(res_fd, 0, SEEK_END);

      close_res_fd = true;
   }

   if (classic_layout) {
      const struct render_context_op_import_resource_layout_v1_request req = {
         .header.op=RENDER_CONTEXT_OP_IMPORT_RESOURCE_LAYOUT_V1,
         .res_id=res_id, .fd_type=res_fd_type, .layout=layout,
      };
      struct render_context_op_import_resource_layout_v1_reply reply = {0};
      bool ok = proxy_socket_send_request_with_fds(&ctx->socket, &req, sizeof(req), &res_fd, 1) &&
         proxy_socket_receive_reply(&ctx->socket, &reply, sizeof(reply)) &&
         reply.magic==VIRGL_IMPORT_LAYOUT_MAGIC && reply.version==1 && reply.res_id==res_id && reply.accepted==1;
      if (close_res_fd) close(res_fd);
      if (!ok) {
         proxy_log("classic layout import rejected for res %u; no guessed-layout fallback", res_id);
         return;
      }
      proxy_context_resource_add(ctx, res_id);
      return;
   }

   /* the proxy ignores iovs since transfer_3d is not supported */
   const struct render_context_op_import_resource_request req = {
      .header.op = RENDER_CONTEXT_OP_IMPORT_RESOURCE,
      .res_id = res_id,
      .fd_type = res_fd_type,
      .size = res_size,
   };
   if (!proxy_socket_send_request_with_fds(&ctx->socket, &req, sizeof(req), &res_fd, 1))
      proxy_log("failed to attach res %d", res_id);

   if (res_fd >= 0 && close_res_fd)
      close(res_fd);

   proxy_context_resource_add(ctx, res_id);
}

static void
proxy_context_gpu_jobs_failstop(struct proxy_context *ctx)
{
   proxy_log("strict GPU context %u teardown UNKNOWN; retaining context/backing, no kill/retire",
             ctx->base.ctx_id);
   /* A failure is not quiescence. Do not return into Rust/VM memory teardown.
    * Full VMExit/pool-grant recovery on UNKNOWN is not proven by this hold. */
   for (;;) {
      const struct timespec delay = { .tv_sec = 1 };
      thrd_sleep(&delay, NULL);
   }
}

static void
proxy_context_drain_gpu_jobs(struct proxy_context *ctx)
{
   const struct render_context_op_drain_gpu_jobs_request req = {
      .header.op = RENDER_CONTEXT_OP_DRAIN_GPU_JOBS,
      .ctx_id = ctx->base.ctx_id, .generation = ctx->gpu_jobs_generation,
   };
   for (;;) {
      struct render_context_op_drain_gpu_jobs_reply reply = { 0 };
      if (!proxy_socket_send_request(&ctx->socket, &req, sizeof(req)) ||
          !proxy_socket_receive_reply(&ctx->socket, &reply, sizeof(reply)) ||
          reply.ctx_id != ctx->base.ctx_id || reply.generation != ctx->gpu_jobs_generation ||
          reply.magic != VIRGL_RENDERER_NATIVE_JOB_CAP_MAGIC || reply.ready > 1 ||
          reply.reserved[0] || reply.reserved[1])
         proxy_context_gpu_jobs_failstop(ctx);
      if (reply.ready)
         return; /* native destroy/join completed BEFORE this ACK */
      const struct timespec retry = { .tv_nsec = 1000000 };
      thrd_sleep(&retry, NULL); /* teardown only, no render-path wait variant */
   }
}

static void
proxy_context_destroy(struct virgl_context *base)
{
   struct proxy_context *ctx = (struct proxy_context *)base;

   if (base->gpu_jobs_enabled) {
      /* Actual process mode: native quiescence+fini ACK first, then EOF and
       * parent waitpid ACK. No SIGKILL may bypass the typed native lifetime. */
      proxy_context_drain_gpu_jobs(ctx);
      shutdown(ctx->socket.fd, SHUT_RDWR);
      if (!proxy_client_destroy_protected_context(ctx->client, base->ctx_id,
                                                 ctx->gpu_jobs_generation))
         proxy_context_gpu_jobs_failstop(ctx);
   } else {
      /* Explicit legacy/no-strict-job scope; unchanged legacy teardown. */
      if (!proxy_client_destroy_context(ctx->client, ctx->base.ctx_id))
         proxy_log("failed to destroy ctx %d", ctx->base.ctx_id);
   }

   if (ctx->fence_socket_fd >= 0)
      close(ctx->fence_socket_fd);

   if (ctx->sync_thread.fence_eventfd >= 0) {
      if (ctx->sync_thread.created) {
         ctx->sync_thread.stop = true;
         write_eventfd(ctx->sync_thread.fence_eventfd_write, 1);
         thrd_join(ctx->sync_thread.thread, NULL);
      }

      if (ctx->sync_thread.fence_eventfd_write != ctx->sync_thread.fence_eventfd)
         close(ctx->sync_thread.fence_eventfd_write);
      close(ctx->sync_thread.fence_eventfd);
   }

   if (ctx->shmem.ptr)
      munmap(ctx->shmem.ptr, ctx->shmem.size);
   if (ctx->shmem.fd >= 0)
      close(ctx->shmem.fd);

   if (ctx->timeline_seqnos) {
      for (uint32_t i = 0; i < PROXY_CONTEXT_TIMELINE_COUNT; i++) {
         struct proxy_timeline *timeline = &ctx->timelines[i];
         list_for_each_entry_safe (struct proxy_fence, fence, &timeline->fences, head) {
            /* Never retired, so nothing dropped their registered fds. */
            virgl_fence_retire(fence->fence_id);
            free(fence);
         }
      }
   }
   mtx_destroy(&ctx->timeline_mutex);

   list_for_each_entry_safe (struct proxy_fence, fence, &ctx->free_fences, head)
      free(fence);
   mtx_destroy(&ctx->free_fences_mutex);

   proxy_context_resource_table_fini(ctx);

   proxy_socket_fini(&ctx->socket);

   free(ctx);
}

static void
proxy_context_init_base(struct proxy_context *ctx)
{
   ctx->base.destroy = proxy_context_destroy;
   ctx->base.attach_resource = proxy_context_attach_resource;
   ctx->base.detach_resource = proxy_context_detach_resource;
   ctx->base.transfer_3d = proxy_context_transfer_3d;
   ctx->base.get_blob = proxy_context_get_blob;
   ctx->base.submit_cmd = proxy_context_submit_cmd;

   ctx->base.get_fencing_fd = proxy_context_get_fencing_fd;
   ctx->base.retire_fences = proxy_context_retire_fences;
   ctx->base.submit_fence = proxy_context_submit_fence;
   ctx->base.enable_gpu_jobs = proxy_context_enable_gpu_jobs;
   ctx->base.prepare_gpu_job = proxy_context_prepare_gpu_job;
}

static bool
proxy_context_init_fencing(struct proxy_context *ctx)
{
   /* The render server updates the shmem for the current seqnos and
    * optionally notifies using the eventfd.  That means, when only
    * VIRGL_RENDERER_THREAD_SYNC is set, we just need to set up the eventfd.
    * When VIRGL_RENDERER_ASYNC_FENCE_CB is also set, we need to create a sync
    * thread as well.
    *
    * Fence polling can always check the shmem directly.
    */
   if (!(proxy_renderer.flags & VIRGL_RENDERER_THREAD_SYNC))
      return true;

   if (create_fence_pipe(&ctx->sync_thread.fence_eventfd,
                         &ctx->sync_thread.fence_eventfd_write)) {
      proxy_log("failed to create fence eventfd/pipe");
      return false;
   }

   if (proxy_renderer.flags & VIRGL_RENDERER_ASYNC_FENCE_CB) {
      int ret = thrd_create(&ctx->sync_thread.thread, proxy_context_sync_thread, ctx);
      if (ret != thrd_success) {
         proxy_log("failed to create sync thread");
         return false;
      }
      ctx->sync_thread.created = true;
   }

   return true;
}

static bool
proxy_context_init_timelines(struct proxy_context *ctx)
{
   atomic_uint *timeline_seqnos = ctx->shmem.ptr;
   for (uint32_t i = 0; i < ARRAY_SIZE(ctx->timelines); i++) {
      atomic_init(&timeline_seqnos[i], 0);

      struct proxy_timeline *timeline = &ctx->timelines[i];
      timeline->cur_seqno = 0;
      timeline->next_seqno = 1;
      list_inithead(&timeline->fences);
   }

   ctx->timeline_seqnos = timeline_seqnos;

   return true;
}

static int
alloc_memfd(const char *name, size_t size, void **out_ptr)
{
   int fd = os_create_anonymous_file(size, name);
   if (fd < 0)
      return -1;

   int ret = add_required_seals_to_fd(fd);
   if (ret)
      goto fail;

   if (!out_ptr)
      return fd;

   void *ptr = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
   if (ptr == MAP_FAILED)
      goto fail;

   *out_ptr = ptr;
   return fd;

fail:
   close(fd);
   return -1;
}

static bool
proxy_context_init_shmem(struct proxy_context *ctx)
{
   const size_t shmem_size = sizeof(*ctx->timeline_seqnos) * PROXY_CONTEXT_TIMELINE_COUNT;
   ctx->shmem.fd = alloc_memfd("proxy-ctx", shmem_size, &ctx->shmem.ptr);
   if (ctx->shmem.fd < 0)
      return false;

   ctx->shmem.size = shmem_size;

   return true;
}

static bool
proxy_context_init(struct proxy_context *ctx, uint32_t ctx_flags)
{
   if (!proxy_context_init_shmem(ctx) || !proxy_context_init_timelines(ctx) ||
       !proxy_context_init_fencing(ctx) || !proxy_context_resource_table_init(ctx)) {
      proxy_log("failed to pre-initialize context");
      return false;
   }

   const struct render_context_op_init_request req = {
      .header.op = RENDER_CONTEXT_OP_INIT,
      .flags = ctx_flags,
      .shmem_size = ctx->shmem.size,
   };
   const int req_fds[2] = { ctx->shmem.fd, ctx->sync_thread.fence_eventfd_write };
   const int req_fd_count = req_fds[1] >= 0 ? 2 : 1;
   if (!proxy_socket_send_request_with_fds(&ctx->socket, &req, sizeof(req), req_fds,
                                           req_fd_count)) {
      proxy_log("failed to initialize context");
      return false;
   }

   return true;
}

struct virgl_context *
proxy_context_create(uint32_t ctx_id,
                     uint32_t ctx_flags,
                     size_t debug_len,
                     const char *debug_name)
{
   struct proxy_client *client = proxy_renderer.client;
   struct proxy_context *ctx;

   const uint32_t capset_id = ctx_flags & VIRGL_RENDERER_CONTEXT_FLAG_CAPSET_ID_MASK;

   int ctx_fd;
   if (!proxy_client_create_context(client, ctx_id, capset_id, debug_len, debug_name,
                                    &ctx_fd)) {
      proxy_log("failed to create a context");
      return NULL;
   }

   ctx = calloc(1, sizeof(*ctx));
   if (!ctx) {
      close(ctx_fd);
      return NULL;
   }

   proxy_context_init_base(ctx);
   ctx->client = client;
   proxy_socket_init(&ctx->socket, ctx_fd);
   ctx->capset_id = capset_id;
   ctx->fence_socket_fd = -1;
   ctx->shmem.fd = -1;
   mtx_init(&ctx->timeline_mutex, mtx_plain);
   mtx_init(&ctx->free_fences_mutex, mtx_plain);
   list_inithead(&ctx->free_fences);
   ctx->sync_thread.fence_eventfd = -1;
   ctx->sync_thread.fence_eventfd_write = -1;

   if (!proxy_context_init(ctx, ctx_flags)) {
      proxy_context_destroy(&ctx->base);
      return NULL;
   }

   return &ctx->base;
}
