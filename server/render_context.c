/*
 * Copyright 2021 Google LLC
 * SPDX-License-Identifier: MIT
 */

#include "render_context.h"

#include <errno.h>
#include <poll.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <unistd.h>

#include "util/u_thread.h"
#include "virgl_util.h"

#include "render_socket.h"
#include "render_state.h"

void
render_context_update_timeline(struct render_context *ctx,
                               uint32_t ring_idx,
                               uint32_t seqno)
{
   /* this can be called by the context's main thread and sync threads */
   atomic_store(&ctx->shmem_timelines[ring_idx], seqno);
   if (ctx->fence_eventfd >= 0)
      write_eventfd(ctx->fence_eventfd, 1);
}

static bool
render_context_dispatch_submit_fence(struct render_context *ctx,
                                     const union render_context_op_request *request,
                                     UNUSED const int *fds,
                                     UNUSED int fd_count)
{
   const struct render_context_op_submit_fence_request *req = &request->submit_fence;

   if ((req->flags & ~(VIRGL_RENDERER_FENCE_FLAG_MERGEABLE |
                       VIRGL_RENDERER_FENCE_FLAG_GPU_JOB_COMPLETE)) ||
       req->ring_index >= (uint32_t)ctx->timeline_count)
      return false;
   const bool gpu_job = req->flags & VIRGL_RENDERER_FENCE_FLAG_GPU_JOB_COMPLETE;
   if (gpu_job && (ctx->backend != RENDER_BACKEND_NEPTUNE ||
                   req->ring_index != 0 || !ctx->gpu_jobs_enabled ||
                   !ctx->gpu_job_prepared || req->seqno != ctx->gpu_job_seqno))
      return false;
   if (!gpu_job && req->ring_index == 0 && ctx->gpu_job_prepared)
      return false; /* cannot consume a prepared job with an unrelated fence */
   /* Existing ordinary fences remain mergeable; preserve the new internal
    * bit on the ordered main channel, never the arrival-order-tolerant OOB. */
   const uint32_t flags = gpu_job ? req->flags : VIRGL_RENDERER_FENCE_FLAG_MERGEABLE;
   if (!render_state_submit_fence(ctx->ctx_id, flags,
                                  req->ring_index, req->seqno)) {
      render_log("submit_fence(ring %u) failed", req->ring_index);
      if (ctx->gpu_jobs_enabled && req->ring_index == 0)
         return false; /* close, never acknowledge a failed GPU job */
   }
   if (gpu_job)
      ctx->gpu_job_prepared = false;
   return true;
}

/* Read exactly `size` bytes off a SOCK_STREAM fd.  Returns false on EOF or
 * error (EOF mid-record included: a torn record means the peer died). */
static bool
render_context_fence_read_full(int fd, void *data, size_t size)
{
   uint8_t *cur = data;
   while (size) {
      const ssize_t r = read(fd, cur, size);
      if (r < 0) {
         if (errno == EINTR || errno == EAGAIN)
            continue;
         return false;
      }
      if (r == 0)
         return false;
      cur += r;
      size -= (size_t)r;
   }
   return true;
}

/* Out-of-band fence channel (see render_protocol.h): registers submit_fence
 * records the moment they arrive, independent of the dispatch thread's
 * backlog.  Only arrival-order-tolerant fences travel here. */
static int
render_context_fence_thread(void *arg)
{
   struct render_context *ctx = arg;

   u_thread_setname("virgl-fence");

   for (;;) {
      struct render_context_op_submit_fence_request req;
      if (!render_context_fence_read_full(ctx->fence_socket_fd, &req,
                                          sizeof(req)))
         break;

      if (req.header.op != RENDER_CONTEXT_OP_SUBMIT_FENCE ||
          req.ring_index >= (uint32_t)ctx->timeline_count ||
          (req.flags & ~VIRGL_RENDERER_FENCE_FLAG_MERGEABLE)) {
         render_log("fence channel: bad record (op %u ring %u); closing",
                    req.header.op, req.ring_index);
         if (req.flags & VIRGL_RENDERER_FENCE_FLAG_GPU_JOB_COMPLETE)
            shutdown(ctx->socket.fd, SHUT_RDWR); /* never reroute strict ring0 */
         break;
      }

      if (!render_state_submit_fence(ctx->ctx_id,
                                     VIRGL_RENDERER_FENCE_FLAG_MERGEABLE,
                                     req.ring_index, req.seqno))
         render_log("fence channel: submit_fence(ring %u) failed",
                    req.ring_index);
   }

   return 0;
}

static bool
render_context_dispatch_attach_fence_socket(struct render_context *ctx,
                                            UNUSED const union render_context_op_request *request,
                                            const int *fds,
                                            int fd_count)
{
   if (fd_count != 1) {
      render_log("attach_fence_socket: expected 1 fd, got %d", fd_count);
      return false;
   }
   if (ctx->fence_thread_created) {
      render_log("attach_fence_socket: already attached");
      close(fds[0]);
      return false;
   }

   ctx->fence_socket_fd = fds[0];
   if (thrd_create(&ctx->fence_thread, render_context_fence_thread, ctx) !=
       thrd_success) {
      render_log("attach_fence_socket: thrd_create failed");
      close(ctx->fence_socket_fd);
      ctx->fence_socket_fd = -1;
      return false;
   }
   ctx->fence_thread_created = true;

   return true;
}

static bool
render_context_dispatch_submit_cmd(struct render_context *ctx,
                                   const union render_context_op_request *request,
                                   UNUSED const int *fds,
                                   UNUSED int fd_count)
{
   const struct render_context_op_submit_cmd_request *req = &request->submit_cmd;
   void *cmd = (void *)req->cmd;
   if (req->size > sizeof(req->cmd)) {
      cmd = malloc(req->size);
      if (!cmd)
         return !ctx->gpu_jobs_enabled; /* no silently skipped tracked command */

      const size_t inlined = sizeof(req->cmd);
      const size_t remain = req->size - inlined;

      memcpy(cmd, req->cmd, inlined);
      if (!render_socket_receive_data(&ctx->socket, (char *)cmd + inlined, remain)) {
         free(cmd);
         return false;
      }
   }

   bool ok = render_state_submit_cmd(ctx->ctx_id, cmd, req->size);

   if (cmd != req->cmd)
      free(cmd);

   return ok;
}

static bool
render_context_dispatch_destroy_resource(struct render_context *ctx,
                                         const union render_context_op_request *req,
                                         UNUSED const int *fds,
                                         UNUSED int fd_count)
{
   render_state_destroy_resource(ctx->ctx_id, req->destroy_resource.res_id);
   return true;
}

static bool
render_context_dispatch_import_resource(struct render_context *ctx,
                                        const union render_context_op_request *request,
                                        const int *fds,
                                        int fd_count)
{
   if (fd_count != 1) {
      render_log("failed to attach resource with fd_count %d", fd_count);
      return false;
   }

   /* classic 3d resource with valid size reuses the blob import path here */
   const struct render_context_op_import_resource_request *req =
      &request->import_resource;
   return render_state_import_resource(ctx->ctx_id, req->res_id, req->fd_type, fds[0],
                                       req->size);
}

static bool
render_context_dispatch_import_resource_layout_v1(struct render_context *ctx,
   const union render_context_op_request *request, const int *fds, int fd_count)
{
   const struct render_context_op_import_resource_layout_v1_request *req = &request->import_resource_layout_v1;
   if (fd_count != 1 || req->reserved || req->fd_type != VIRGL_RESOURCE_FD_DMABUF ||
       !virgl_import_layout_valid(&req->layout)) return false;
   int owned = dup(fds[0]);
   bool accepted = owned >= 0 && render_state_import_resource_layout_v1(ctx->ctx_id,
      req->res_id, req->fd_type, owned, req->layout.allocation_size, &req->layout);
   if (owned >= 0 && !accepted) close(owned);
   const struct render_context_op_import_resource_layout_v1_reply reply = {
      .magic=VIRGL_IMPORT_LAYOUT_MAGIC, .version=1, .res_id=req->res_id, .accepted=accepted,
   };
   if (!render_socket_send_reply(&ctx->socket, &reply, sizeof(reply)))
      return false; /* dispatcher closes incoming FD; native keeps its own duplicate */
   close(fds[0]);
   return true;
}

static bool
render_context_dispatch_create_resource(struct render_context *ctx,
                                        const union render_context_op_request *request,
                                        const int *fds,
                                        int fd_count)
{
   const struct render_context_op_create_resource_request *req =
      &request->create_resource;
   /* the fd (if any) stays owned by this function: backends dup what they keep */
   const int shm_fd = req->has_shm_fd && fd_count == 1 ? fds[0] : -1;
   if (req->has_shm_fd && shm_fd < 0) {
      render_log("create_resource: has_shm_fd with fd_count %d", fd_count);
      return false;
   }
   struct render_context_op_create_resource_reply reply = {
      .fd_type = VIRGL_RESOURCE_FD_INVALID,
   };
   int res_fd;
   bool ok = render_state_create_resource(ctx->ctx_id, req->res_id, req->blob_id,
                                          req->blob_size, req->blob_flags, &reply.fd_type,
                                          &res_fd, &reply.map_info, &reply.vulkan_info,
                                          &reply.export_format, shm_fd,
                                          req->shm_fd_offset);
   if (shm_fd >= 0)
      close(shm_fd);
   if (!ok)
      return render_socket_send_reply(&ctx->socket, &reply, sizeof(reply));

   ok =
      render_socket_send_reply_with_fds(&ctx->socket, &reply, sizeof(reply), &res_fd, 1);
   close(res_fd);

   return ok;
}

static bool
render_context_dispatch_init(struct render_context *ctx,
                             const union render_context_op_request *request,
                             const int *fds,
                             int fd_count)
{
   if (fd_count != 1 && fd_count != 2)
      return false;

   const struct render_context_op_init_request *req = &request->init;
   const int timeline_count = req->shmem_size / sizeof(*ctx->shmem_timelines);
   const int shmem_fd = fds[0];
   const int fence_eventfd = fd_count == 2 ? fds[1] : -1;

   void *shmem_ptr = mmap(NULL, req->shmem_size, PROT_WRITE, MAP_SHARED, shmem_fd, 0);
   if (shmem_ptr == MAP_FAILED)
      return false;

   if (!render_state_create_context(ctx, req->flags, ctx->name_len, ctx->name)) {
      munmap(shmem_ptr, req->shmem_size);
      return false;
   }

   ctx->shmem_fd = shmem_fd;
   ctx->shmem_size = req->shmem_size;
   ctx->shmem_ptr = shmem_ptr;
   ctx->shmem_timelines = shmem_ptr;

   for (int i = 0; i < timeline_count; i++)
      atomic_store(&ctx->shmem_timelines[i], 0);

   ctx->timeline_count = timeline_count;

   ctx->fence_eventfd = fence_eventfd;

   return true;
}

static bool
render_context_dispatch_enable_gpu_jobs(struct render_context *ctx,
                                        const union render_context_op_request *request,
                                        UNUSED const int *fds,
                                        UNUSED int fd_count)
{
   const struct render_context_op_enable_gpu_jobs_request *req = &request->enable_gpu_jobs;
   struct render_context_op_enable_gpu_jobs_reply reply = {
      .version = 0, .ctx_id = ctx->ctx_id, .generation = req->generation,
   };
   if (req->version != 0 || req->ctx_id != ctx->ctx_id || !req->generation ||
       !ctx->shmem_timelines || ctx->backend != RENDER_BACKEND_NEPTUNE)
      return false;
   if (ctx->gpu_jobs_enabled && ctx->gpu_jobs_generation != req->generation)
      return false;
   const uint32_t magic = render_state_enable_gpu_jobs(ctx);
   if (magic == VIRGL_RENDERER_NATIVE_JOB_CAP_MAGIC) {
      ctx->gpu_jobs_enabled = true;
      ctx->gpu_jobs_generation = req->generation;
      reply.magic = magic;
      reply.caps = VIRGL_RENDERER_NATIVE_JOB_CAP_GPU_RETIRE;
   }
   return render_socket_send_reply(&ctx->socket, &reply, sizeof(reply));
}

static bool
render_context_dispatch_prepare_gpu_job(struct render_context *ctx,
                                        const union render_context_op_request *request,
                                        UNUSED const int *fds,
                                        UNUSED int fd_count)
{
   const struct render_context_op_prepare_gpu_job_request *req = &request->prepare_gpu_job;
   struct render_context_op_prepare_gpu_job_reply reply = {
      .ctx_id = ctx->ctx_id, .generation = ctx->gpu_jobs_generation,
      .seqno = req->seqno, .front_fence_id = req->front_fence_id,
   };
   if (!ctx->gpu_jobs_enabled || ctx->gpu_job_prepared ||
       req->ctx_id != ctx->ctx_id || req->generation != ctx->gpu_jobs_generation ||
       !req->seqno || req->reserved[0] || req->reserved[1] ||
       !render_state_prepare_gpu_job(ctx, req->seqno))
      return false; /* applied/unknown failure closes; no next-job reuse */
   ctx->gpu_job_prepared = true;
   ctx->gpu_job_seqno = req->seqno;
   ctx->gpu_job_front_fence_id = req->front_fence_id;
   reply.magic = VIRGL_RENDERER_NATIVE_JOB_CAP_MAGIC;
   return render_socket_send_reply(&ctx->socket, &reply, sizeof(reply));
}

static bool
render_context_dispatch_drain_gpu_jobs(struct render_context *ctx,
                                      const union render_context_op_request *request,
                                      UNUSED const int *fds,
                                      UNUSED int fd_count)
{
   const struct render_context_op_drain_gpu_jobs_request *req = &request->drain_gpu_jobs;
   struct render_context_op_drain_gpu_jobs_reply reply = {
      .ctx_id = ctx->ctx_id, .generation = ctx->gpu_jobs_generation,
      .magic = VIRGL_RENDERER_NATIVE_JOB_CAP_MAGIC,
   };
   if (!ctx->gpu_jobs_enabled || req->ctx_id != ctx->ctx_id || req->reserved ||
       req->generation != ctx->gpu_jobs_generation)
      return false;
   /* Stop/join the OOB submitter BEFORE querying quiescence or freeing
    * native context state. Main socket remains alive for drain retries. */
   if (ctx->fence_thread_created) {
      shutdown(ctx->fence_socket_fd, SHUT_RDWR);
      thrd_join(ctx->fence_thread, NULL);
      ctx->fence_thread_created = false;
   }
   if (!ctx->native_destroyed && render_state_gpu_jobs_quiescent(ctx)) {
      /* Positive typed proof, not jobs.empty or process-alive inference. */
      render_state_destroy_context(ctx->ctx_id);
      ctx->native_destroyed = true;
   }
   reply.ready = ctx->native_destroyed;
   return render_socket_send_reply(&ctx->socket, &reply, sizeof(reply));
}

static bool
render_context_dispatch_nop(UNUSED struct render_context *ctx,
                            UNUSED const union render_context_op_request *req,
                            UNUSED const int *fds,
                            UNUSED int fd_count)
{
   return true;
}

struct render_context_dispatch_entry {
   size_t expect_size;
   int max_fd_count;
   bool (*dispatch)(struct render_context *ctx,
                    const union render_context_op_request *req,
                    const int *fds,
                    int fd_count);
};

static const struct render_context_dispatch_entry
   render_context_dispatch_table[RENDER_CONTEXT_OP_COUNT] = {
#define RENDER_CONTEXT_DISPATCH(NAME, name, max_fd)                                      \
   [RENDER_CONTEXT_OP_##                                                                 \
      NAME] = { .expect_size = sizeof(struct render_context_op_##name##_request),        \
                .max_fd_count = (max_fd),                                                \
                .dispatch = render_context_dispatch_##name }
      RENDER_CONTEXT_DISPATCH(NOP, nop, 0),
      RENDER_CONTEXT_DISPATCH(INIT, init, 2),
      RENDER_CONTEXT_DISPATCH(CREATE_RESOURCE, create_resource, 1),
      RENDER_CONTEXT_DISPATCH(IMPORT_RESOURCE, import_resource, 1),
      RENDER_CONTEXT_DISPATCH(IMPORT_RESOURCE_LAYOUT_V1, import_resource_layout_v1, 1),
      RENDER_CONTEXT_DISPATCH(DESTROY_RESOURCE, destroy_resource, 0),
      RENDER_CONTEXT_DISPATCH(SUBMIT_CMD, submit_cmd, 0),
      RENDER_CONTEXT_DISPATCH(SUBMIT_FENCE, submit_fence, 0),
      RENDER_CONTEXT_DISPATCH(ATTACH_FENCE_SOCKET, attach_fence_socket, 1),
      RENDER_CONTEXT_DISPATCH(ENABLE_GPU_JOBS, enable_gpu_jobs, 0),
      RENDER_CONTEXT_DISPATCH(PREPARE_GPU_JOB, prepare_gpu_job, 0),
      RENDER_CONTEXT_DISPATCH(DRAIN_GPU_JOBS, drain_gpu_jobs, 0),
#undef RENDER_CONTEXT_DISPATCH
   };

static bool
render_context_dispatch(struct render_context *ctx)
{
   union render_context_op_request req;
   size_t req_size;
   int req_fds[8];
   int req_fd_count;
   if (!render_socket_receive_request_with_fds(&ctx->socket, &req, sizeof(req), &req_size,
                                               req_fds, ARRAY_SIZE(req_fds),
                                               &req_fd_count))
      return false;

   assert((unsigned int)req_fd_count <= ARRAY_SIZE(req_fds));

   if (req.header.op >= RENDER_CONTEXT_OP_COUNT) {
      render_log("invalid context op %d", req.header.op);
      goto fail;
   }

   const struct render_context_dispatch_entry *entry =
      &render_context_dispatch_table[req.header.op];
   if (entry->expect_size != req_size || entry->max_fd_count < req_fd_count) {
      render_log("invalid request size (%zu) or fd count (%d) for context op %d",
                 req_size, req_fd_count, req.header.op);
      goto fail;
   }

   if (ctx->native_destroyed && req.header.op != RENDER_CONTEXT_OP_DRAIN_GPU_JOBS)
      goto fail;
   const bool ok = entry->dispatch(ctx, &req, req_fds, req_fd_count);
   if (!ok) {
      render_log("failed to dispatch context op %d", req.header.op);
      goto fail;
   }

   return true;

fail:
   for (int i = 0; i < req_fd_count; i++)
      close(req_fds[i]);
   return false;
}

static bool
render_context_run(struct render_context *ctx)
{
   while (true) {
      /* Positive native drain ACK is a graceful terminal protocol state.
       * Exit/fini before parent waitpid ACK; normal cleanup is not EOF error. */
      if (ctx->native_destroyed)
         return true;
      if (!render_context_dispatch(ctx))
         return false;
   }

   return true;
}

static void
render_context_fini(struct render_context *ctx)
{
   /* stop the fence channel before the backend context goes away: its
    * thread calls into render_state and must be joined first */
   if (ctx->fence_thread_created) {
      shutdown(ctx->fence_socket_fd, SHUT_RDWR);
      thrd_join(ctx->fence_thread, NULL);
      ctx->fence_thread_created = false;
   }
   /* EOF/error is not a positive DRAIN. Native destroy returns void and
    * may retain unknown jobs: returning from this worker would nevertheless
    * exit the process and drop its mappings. Keep the whole worker alive.
    * This is not proof that outer VM GuestMemory/pool grants are retained. */
   if (ctx->gpu_jobs_enabled && !ctx->native_destroyed) {
      render_log("protected GPU worker lost protocol before positive DRAIN; retaining, no success");
      for (;;)
         (void)poll(NULL, 0, 1000);
   }
   if (ctx->fence_socket_fd >= 0)
      close(ctx->fence_socket_fd);

   /* A protected normal drain joined native before its ACK. Only legacy
    * contexts can reach native fini here without a positive DRAIN. */
   if (!ctx->native_destroyed)
      render_state_destroy_context(ctx->ctx_id);

   if (ctx->shmem_ptr)
      munmap(ctx->shmem_ptr, ctx->shmem_size);
   if (ctx->shmem_fd >= 0)
      close(ctx->shmem_fd);

   if (ctx->fence_eventfd >= 0)
      close(ctx->fence_eventfd);

   if (ctx->name)
      free(ctx->name);

   render_socket_fini(&ctx->socket);
}

static void
render_context_set_thread_name(uint32_t ctx_id, UNUSED const char *ctx_name)
{
   char thread_name[16];

#pragma GCC diagnostic push
#if !defined(__clang__) || __clang_major__ >= 18
#pragma GCC diagnostic ignored "-Wformat-truncation"
#endif
   /* context name may match guest process name, so use a generic name in
    * production/release builds.
    */
#ifndef NDEBUG
   snprintf(thread_name, ARRAY_SIZE(thread_name), "virgl-%d-%s", ctx_id, ctx_name);
#else
   snprintf(thread_name, ARRAY_SIZE(thread_name), "virgl-%d-gpu_renderer", ctx_id);
#endif
#pragma GCC diagnostic pop

   u_thread_setname(thread_name);
}

static const char *ctx_name_expansions[] = {
   "DOOMEternalx64vk.exe",
};

static bool
render_context_init_name(struct render_context *ctx,
                         uint32_t ctx_id,
                         const char *ctx_name)
{
   /* Linux guests may only pass the first 15 chars of a guest application name,
    * plus a null terminator. In that case, attempt substring matching and
    * expansion to enable proper dri-conf lookup in native mesa drivers.
    */
   static const size_t truncated_name_len = 15;
   size_t name_len = strlen(ctx_name);
   if (name_len == truncated_name_len) {
      for (uint32_t i = 0; i < ARRAY_SIZE(ctx_name_expansions); i++) {
         const char *full = ctx_name_expansions[i];
         if (!strncmp(ctx_name, full, truncated_name_len)) {
            ctx_name = full;
            name_len = strlen(full);
            break;
         }
      }
   }

   ctx->name_len = name_len;
   ctx->name = malloc(name_len + 1);
   if (!ctx->name)
      return false;

   strcpy(ctx->name, ctx_name);

   /* Overrides the executable name used only by mesa to load app-specific
    * driver configuration.
    */
   setenv("MESA_DRICONF_EXECUTABLE_OVERRIDE", ctx->name, 0);

   /* Host Mesa still sees the process name as "virgl_render_server" unless
    * additionally overridden by setenv("MESA_PROCESS_NAME", ...).
    */
   render_context_set_thread_name(ctx_id, ctx->name);

   return true;
}

static bool
render_context_init(struct render_context *ctx, const struct render_context_args *args)
{
   memset(ctx, 0, sizeof(*ctx));
   ctx->ctx_id = args->ctx_id;
   render_socket_init(&ctx->socket, args->ctx_fd);
   ctx->shmem_fd = -1;
   ctx->fence_eventfd = -1;
   ctx->fence_socket_fd = -1;

   if (!render_context_init_name(ctx, args->ctx_id, args->ctx_name))
      return false;

   return true;
}

bool
render_context_main(const struct render_context_args *args)
{
   struct render_context ctx;

   assert(args->valid && args->ctx_id && args->ctx_fd >= 0);

   if (!render_state_init(args->init_flags)) {
      close(args->ctx_fd);
      return false;
   }

   if (!render_context_init(&ctx, args)) {
      render_state_fini();
      close(args->ctx_fd);
      return false;
   }

   const bool ok = render_context_run(&ctx);
   render_context_fini(&ctx);

   render_state_fini();

   return ok;
}
