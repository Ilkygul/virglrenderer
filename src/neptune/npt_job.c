/* C12: actual typed D3D11 tails at Notify decode checkpoints. */
#include "npt_job.h"
#include "npt_job_credit.h"
#include "npt_context.h"
#include "npt_ring.h"
#include "npt_com.h"
#include "npt_profile.h"
#include "d3d11_video_terminal.h"
#include <stdlib.h>
#include <string.h>
#include "neptune-protocol/npt_protocol_defs.h"
#include "neptune-protocol/npt_protocol_host_dispatch_types.h"
#include <poll.h>
#include <unistd.h>
#if defined(__linux__)
#include <pthread.h>
#include <time.h>
#endif
#include <sys/stat.h>

#define JOB_MAX_NATIVE 32u
#define JOB_DEADLINE_NS (30ull * 1000000000ull)

/* C22 diagnostic only: one fresh, empty regular marker activates at most
 * 128 sampled publisher jobs per PROCESS, two lines per completed checkpoint.
 * No file access when unset/invalid, or after activation/budget exhaustion.
 * The marker is metadata-only; removal/replacement never resets the window. */
#define JOB_TRACE_SAMPLES 128u
static atomic_bool job_trace_active;
static atomic_uint job_trace_next;
struct job_trace_record {
   unsigned sample, video_calls, native_calls, resource_calls;
   uint32_t ctx, tail, last_iface, last_method, markers, polls;
   uint64_t prepared, fence, publisher, generation, serial;
   uint64_t notify_ns, checkpoint_ns, signal_ns, flush_ns, marker_end_ns;
   uint64_t first_poll_ns, positive_ns;
};
static unsigned job_trace_take(void)
{
   if (atomic_load_explicit(&job_trace_next, memory_order_relaxed) >= JOB_TRACE_SAMPLES)
      return 0;
   if (!atomic_load_explicit(&job_trace_active, memory_order_acquire)) {
      const char *path = getenv("NPT_NATIVE_JOB_TRACE_FILE");
      if (!path || path[0] != '/') return 0;
      size_t n = 0;
      while (path[n] && n < 512) {
         if ((unsigned char)path[n] < 32 || (unsigned char)path[n] > 126) return 0;
         ++n;
      }
      if (!n || n >= 512) return 0;
      struct stat st;
      if (lstat(path, &st) || !S_ISREG(st.st_mode) || st.st_size != 0) return 0;
      atomic_store_explicit(&job_trace_active, true, memory_order_release);
   }
   const unsigned sample = atomic_fetch_add_explicit(&job_trace_next, 1, memory_order_relaxed);
   return sample < JOB_TRACE_SAMPLES ? sample + 1 : 0;
}
static uint64_t job_trace_delta(uint64_t end, uint64_t start)
{
   return start && end >= start ? end - start : 0;
}
static void job_trace_emit(const char *phase, const struct job_trace_record *t)
{
   if (!t->sample) return;
   npt_log("C22 CHECKPOINT_%s pid=%ld sample=%u ctx=%u prepared=%" PRIu64
      " fence=%" PRIu64 " publisher=%" PRIu64 " generation=%" PRIu64
      " serial=%" PRIu64 " tail=%u markers=%u video=%u native=%u resource=%u"
      " last_iface=%u last_method=%u notify_ns=%" PRIu64 " checkpoint_ns=%" PRIu64
      " marker_end_ns=%" PRIu64 " first_poll_ns=%" PRIu64 " positive_ns=%" PRIu64
      " decode_ns=%" PRIu64 " signal_cpu_ns=%" PRIu64 " flush_cpu_ns=%" PRIu64
      " fifo_ns=%" PRIu64 " poll_lag_ns=%" PRIu64 " total_ns=%" PRIu64
      " polls=%u completion_observed=%u gpu_markers_observed=%u",
      phase, (long)getpid(), t->sample, t->ctx, t->prepared, t->fence,
      t->publisher, t->generation, t->serial, t->tail, t->markers,
      t->video_calls, t->native_calls, t->resource_calls, t->last_iface, t->last_method,
      t->notify_ns, t->checkpoint_ns, t->marker_end_ns, t->first_poll_ns, t->positive_ns,
      job_trace_delta(t->checkpoint_ns, t->notify_ns), t->signal_ns, t->flush_ns,
      job_trace_delta(t->first_poll_ns, t->marker_end_ns),
      job_trace_delta(t->positive_ns, t->first_poll_ns),
      job_trace_delta(t->positive_ns, t->notify_ns), t->polls, t->positive_ns != 0,
      t->positive_ns != 0 && t->markers != 0);
}

struct job_native {
   struct list_head head;
   void *identity, *context4, *device5, *fence;
   mtx_t mutex;
   bool video_terminal_checked;
   uint64_t value, owner_ring, owner_generation;
};
struct job_marker { struct job_native *native; uint64_t value, observed; };
struct native_job {
   struct list_head head;
   mtx_t mutex;
   uint64_t expected_server_id, fence_id, deadline;
   uint64_t publisher_ring, publisher_generation, publisher_serial;
   uint32_t publisher_tail;
   unsigned pending, marker_count;
   bool submitted, assigned, strict;
   struct job_marker markers[JOB_MAX_NATIVE];
   struct job_trace_record trace;
};
struct job_checkpoint {
   struct list_head head;
   struct native_job *job;
   uint32_t target;
};
struct npt_job_ring {
   mtx_t mutex;
   struct list_head checkpoints;
   bool publisher, active;
   struct npt_job_credit credit;
   unsigned touched_count;
   struct job_native *touched[JOB_MAX_NATIVE];
   bool trace_collect;
   unsigned trace_video_calls, trace_native_calls, trace_resource_calls;
   uint32_t trace_last_iface, trace_last_method;
};
struct npt_gpu_jobs {
   struct npt_context *ctx;
   mtx_t mutex;
   cnd_t cond;
   thrd_t thread;
   struct list_head jobs, natives;
   struct native_job *prepared, *building;
   size_t count;
   unsigned native_count, logs;
   atomic_bool poison;
   bool stopping, activated, admission_closed, video_retire;
};
/* A failed worker cannot accumulate unbounded retired contexts. Once an
 * unknown exists, all later enable/prepare attempts fail until VM exit. */
static atomic_bool job_worker_poison;

/* Only Linux/Android changes the jobs condition's clock. cnd_t is the
 * existing pthread_cond_t alias here; cnd_destroy keeps the same lifecycle.
 * A failed monotonic setup rejects activation, never a realtime fallback. */
static int job_cond_init(cnd_t *cond)
{
#if defined(__linux__)
   pthread_condattr_t attr;
   if (pthread_condattr_init(&attr)) return thrd_error;
   int rc = pthread_condattr_setclock(&attr, CLOCK_MONOTONIC);
   if (!rc) rc = pthread_cond_init(cond, &attr);
   pthread_condattr_destroy(&attr);
   return rc ? thrd_error : thrd_success;
#else
   return cnd_init(cond);
#endif
}

/* Caller holds s->mutex. At most 2 ms between completion/deadline checks;
 * CPU checkpoint readiness can still wake the bounded wait early.
 * Other platforms and clock/wait errors retain the old relative poll. */
static void job_wait_ready(struct npt_gpu_jobs *s, uint64_t wait_ns)
{
   (void)wait_ns; /* Non-Linux retains the original relative 2 ms poll. */
#if defined(__linux__)
   struct timespec until;
   if (!clock_gettime(CLOCK_MONOTONIC, &until)) {
      until.tv_nsec += (long)wait_ns;
      if (until.tv_nsec >= 1000000000L) {
         ++until.tv_sec;
         until.tv_nsec -= 1000000000L;
      }
      int rc = cnd_timedwait(&s->cond, &s->mutex, &until);
      if (rc == thrd_success || rc == thrd_busy) return;
   }
#endif
   mtx_unlock(&s->mutex);
   (void)poll(NULL, 0, 2);
   mtx_lock(&s->mutex);
}

static void job_fail(struct npt_gpu_jobs *s, const char *why)
{
   if (!atomic_exchange_explicit(&s->poison, true, memory_order_acq_rel))
      npt_log("C12 GPU_JOB TEST_FAIL ctx=%u %s; no success retirement", s->ctx->ctx_id, why);
   atomic_store_explicit(&job_worker_poison, true, memory_order_release);
}

/* A finite graphics fence is insufficient when an earlier CS operation
 * reported a terminal codec/backend failure without losing the GPU device.
 * GetDeviceRemovedReason is inherited by the strongly held Device5 object.
 * Query after each fence sample so a CS-ordered sticky failure cannot retire
 * as a successful GPU job. No public API or protocol extension is required. */
static bool job_native_healthy(struct npt_gpu_jobs *s, struct job_native *n,
                               const char *why)
{
   if (!n->device5) { job_fail(s, "native device status identity missing"); return false; }
   PFN_ID3D11Device_GetDeviceRemovedReason get_reason = NPT_COM_VTBL_FUNC(
      PFN_ID3D11Device_GetDeviceRemovedReason, npt_com_vtable(n->device5), NPT_VTBL_ID3D11Device_GetDeviceRemovedReason);
   const HRESULT reason = get_reason(n->device5);
   if (reason != 0) { job_fail(s, why); return false; }
   return true;
}

/* New paired private capability. A legacy or mismatched native library
 * cannot pass by returning S_OK without writing this exact zeroed POD. */
static bool job_native_video_capable(struct npt_gpu_jobs *s, struct job_native *n)
{
   mtx_lock(&n->mutex);
   if (n->video_terminal_checked) { mtx_unlock(&n->mutex); return true; }
   struct npt_d3d11_video_terminal_v1 cap = {0};
   PFN_ID3D11Device_CheckFeatureSupport query = NPT_COM_VTBL_FUNC(
      PFN_ID3D11Device_CheckFeatureSupport, npt_com_vtable(n->device5), NPT_VTBL_ID3D11Device_CheckFeatureSupport);
   const HRESULT hr = query(n->device5, (D3D11_FEATURE)NPT_D3D11_FEATURE_VIDEO_TERMINAL_V1,
                            &cap, sizeof(cap));
   bool valid = hr == 0 && cap.magic == NPT_D3D11_VIDEO_TERMINAL_MAGIC &&
      cap.version == NPT_D3D11_VIDEO_TERMINAL_VERSION && cap.size == sizeof(cap) &&
      cap.supported == NPT_D3D11_VIDEO_TERMINAL_SUPPORTED;
   for (unsigned i = 0; i < 4; ++i) valid = valid && cap.reserved[i] == 0;
   n->video_terminal_checked = valid;
   mtx_unlock(&n->mutex);
   if (!valid) job_fail(s, "paired native Video terminal capability unavailable or invalid");
   return valid;
}

static void *job_qi(void *p, const GUID *iid)
{
   void *out = NULL;
   if (!p || npt_com_query_interface(p, iid, &out) != 0) return NULL;
   return out;
}

static struct job_native *job_native_get(struct npt_gpu_jobs *s, struct npt_ring *ring, void *dc)
{
   void *context4 = job_qi(dc, &NPT_IID_ID3D11DeviceContext4);
   if (!context4) { job_fail(s, "typed ID3D11DeviceContext4 unavailable"); return NULL; }
   PFN_ID3D11DeviceContext_GetType get_type = NPT_COM_VTBL_FUNC(
      PFN_ID3D11DeviceContext_GetType, npt_com_vtable(context4), NPT_VTBL_ID3D11DeviceContext_GetType);
   if (get_type(context4) != D3D11_DEVICE_CONTEXT_IMMEDIATE) {
      npt_com_release(context4); return NULL; /* recording is CPU-only */
   }
   void *identity = job_qi(context4, &NPT_IID_IUnknown);
   if (!identity) { npt_com_release(context4); job_fail(s, "context identity QI failed"); return NULL; }
   mtx_lock(&s->mutex);
   list_for_each_entry(struct job_native, n, &s->natives, head) {
      if (n->identity == identity) {
         const bool owner = n->owner_ring == ring->id &&
            n->owner_generation == ring->gpu_jobs->credit.generation;
         mtx_unlock(&s->mutex); npt_com_release(identity); npt_com_release(context4);
         if (!owner) job_fail(s, "native immediate DC used from another publisher ring/generation");
         return owner ? n : NULL;
      }
   }
   if (s->native_count >= JOB_MAX_NATIVE) {
      mtx_unlock(&s->mutex); npt_com_release(identity); npt_com_release(context4);
      job_fail(s, "native device/context bound exceeded"); return NULL;
   }
   struct job_native *n = calloc(1, sizeof(*n));
   if (!n || mtx_init(&n->mutex, mtx_plain) != thrd_success) {
      free(n); mtx_unlock(&s->mutex); npt_com_release(identity); npt_com_release(context4);
      job_fail(s, "native state allocation failed"); return NULL;
   }
   n->identity = identity; n->context4 = context4;
   n->owner_ring = ring->id; n->owner_generation = ring->gpu_jobs->credit.generation;
   ID3D11Device *device = NULL;
   PFN_ID3D11DeviceChild_GetDevice get_device = NPT_COM_VTBL_FUNC(
      PFN_ID3D11DeviceChild_GetDevice, npt_com_vtable(context4), NPT_VTBL_ID3D11DeviceChild_GetDevice);
   get_device(context4, &device);
   n->device5 = job_qi(device, &NPT_IID_ID3D11Device5);
   if (device) npt_com_release(device);
   HRESULT hr = (HRESULT)0x80004002u;
   if (n->device5) {
      PFN_ID3D11Device5_CreateFence create = NPT_COM_VTBL_FUNC(
         PFN_ID3D11Device5_CreateFence, npt_com_vtable(n->device5), NPT_VTBL_ID3D11Device5_CreateFence);
      hr = create(n->device5, 0, (D3D11_FENCE_FLAG)0, &NPT_IID_ID3D11Fence, &n->fence);
   }
   if (hr != 0 || !n->fence) {
      if (n->fence) npt_com_release(n->fence);
      if (n->device5) npt_com_release(n->device5);
      npt_com_release(n->identity); npt_com_release(n->context4);
      mtx_destroy(&n->mutex); free(n); mtx_unlock(&s->mutex);
      job_fail(s, "actual native device fence creation failed"); return NULL;
   }
   list_addtail(&n->head, &s->natives); s->native_count++;
   mtx_unlock(&s->mutex); return n;
}

/* Called only by the ring decoder at an exact publisher boundary. No GPU wait. */
static bool job_checkpoint_signal(struct npt_gpu_jobs *s, struct npt_job_ring *r,
                                  struct native_job *j)
{
   mtx_lock(&j->mutex);
   if (j->trace.sample) {
      j->trace.checkpoint_ns = npt_profile_now_ns();
      j->trace.video_calls = r->trace_video_calls;
      j->trace.native_calls = r->trace_native_calls;
      j->trace.resource_calls = r->trace_resource_calls;
      j->trace.last_iface = r->trace_last_iface;
      j->trace.last_method = r->trace_last_method;
   }
   for (unsigned i = 0; i < r->touched_count; ++i) {
      struct job_native *n = r->touched[i];
      if (j->marker_count >= JOB_MAX_NATIVE) { job_fail(s, "job marker bound exceeded"); break; }
      mtx_lock(&n->mutex);
      if (n->value >= UINT64_MAX - 1) {
         mtx_unlock(&n->mutex); job_fail(s, "native fence counter exhausted"); break;
      }
      const uint64_t value = ++n->value;
      PFN_ID3D11DeviceContext4_Signal signal = NPT_COM_VTBL_FUNC(
         PFN_ID3D11DeviceContext4_Signal, npt_com_vtable(n->context4), NPT_VTBL_ID3D11DeviceContext4_Signal);
      PFN_ID3D11DeviceContext_Flush flush = NPT_COM_VTBL_FUNC(
         PFN_ID3D11DeviceContext_Flush, npt_com_vtable(n->context4), NPT_VTBL_ID3D11DeviceContext_Flush);
      const uint64_t trace_begin = j->trace.sample ? npt_profile_now_ns() : 0;
      const HRESULT hr = signal(n->context4, n->fence, value);
      const uint64_t trace_signaled = j->trace.sample ? npt_profile_now_ns() : 0;
      if (hr == 0) flush(n->context4);
      if (j->trace.sample) {
         const uint64_t trace_flushed = npt_profile_now_ns();
         j->trace.signal_ns += job_trace_delta(trace_signaled, trace_begin);
         j->trace.flush_ns += job_trace_delta(trace_flushed, trace_signaled);
      }
      mtx_unlock(&n->mutex);
      if (hr != 0) { job_fail(s, "actual native Signal failed"); break; }
      j->markers[j->marker_count++] = (struct job_marker){ .native = n, .value = value };
   }
   r->touched_count = 0;
   r->trace_collect = false;
   r->trace_video_calls = r->trace_native_calls = r->trace_resource_calls = 0;
   r->trace_last_iface = r->trace_last_method = 0;
   if (j->trace.sample) {
      j->trace.markers = j->marker_count;
      j->trace.marker_end_ns = npt_profile_now_ns();
   }
   /* Copy diagnostic POD before pending drops: the retirement worker may
    * own/free this job immediately after the existing mutex unlock. */
   const struct job_trace_record trace = j->trace;
   assert(j->pending); --j->pending;
   mtx_unlock(&j->mutex);
   /* Never invert the worker's s->mutex -> j->mutex order. s stays owned
    * by this live ring decoder; no job pointer is used after this unlock. */
   mtx_lock(&s->mutex);
   cnd_signal(&s->cond);
   mtx_unlock(&s->mutex);
   job_trace_emit("DISPATCHED", &trace);
   return !atomic_load_explicit(&s->poison, memory_order_acquire);
}

/* Called only after the unchanged ready/finite/health checks failed.
 * CPU-not-ready/poison keeps 2 ms; only an actual GPU-pending head ramps
 * 250 -> 500 -> 1000 -> 2000 us. No completion is inferred from the delay. */
static uint64_t job_retry_ns(bool cpu_ready, bool healthy, uint64_t *backoff)
{
   if (!cpu_ready || !healthy) {
      *backoff = 250000ull;
      return 2000000ull;
   }
   const uint64_t delay = *backoff;
   if (*backoff < 2000000ull) *backoff *= 2;
   return delay;
}

static int job_thread(void *arg)
{
   struct npt_gpu_jobs *s = arg;
   struct native_job *wait_head = NULL;
   uint64_t gpu_backoff_ns = 250000ull;
   mtx_lock(&s->mutex);
   while (!s->stopping) {
      if (list_is_empty(&s->jobs) || atomic_load_explicit(&s->poison, memory_order_acquire)) {
         cnd_wait(&s->cond, &s->mutex); continue;
      }
      struct native_job *j = list_first_entry(&s->jobs, struct native_job, head);
      if (j != wait_head) {
         wait_head = j;
         gpu_backoff_ns = 250000ull;
      }
      mtx_lock(&j->mutex);
      const bool cpu_ready = j->assigned && j->submitted && j->pending == 0;
      bool done = j->assigned && j->submitted && j->pending == 0;
      if (done && j->trace.sample) {
         if (!j->trace.first_poll_ns) j->trace.first_poll_ns = npt_profile_now_ns();
         ++j->trace.polls;
      }
      for (unsigned i = 0; done && i < j->marker_count; ++i) {
         struct job_marker *m = &j->markers[i];
         PFN_ID3D11Fence_GetCompletedValue get = NPT_COM_VTBL_FUNC(
            PFN_ID3D11Fence_GetCompletedValue, npt_com_vtable(m->native->fence), NPT_VTBL_ID3D11Fence_GetCompletedValue);
         const uint64_t actual = get(m->native->fence);
         m->observed = actual;
         if (actual == UINT64_MAX) { job_fail(s, "native fence returned UINT64_MAX"); done = false; break; }
         if (!job_native_healthy(s, m->native, "native device reported non-S_OK status")) { done = false; break; }
         if (actual < m->value) done = false;
      }
      if (npt_profile_now_ns() >= j->deadline && !done) job_fail(s, "job completion deadline (not success)");
      const bool healthy = !atomic_load_explicit(&s->poison, memory_order_acquire);
      if (done && healthy && j->trace.sample) {
         j->trace.positive_ns = npt_profile_now_ns();
         j->trace.fence = j->fence_id;
      }
      mtx_unlock(&j->mutex);
      if (!done || !healthy) {
         job_wait_ready(s, job_retry_ns(cpu_ready, healthy, &gpu_backoff_ns));
         continue;
      }
      wait_head = NULL; /* Clear before retiring/freeing; no pointer-reuse carryover. */
      gpu_backoff_ns = 250000ull;
      list_del(&j->head); --s->count;
      const uint64_t fence = j->fence_id;
      const unsigned markers = j->marker_count;
      const uint64_t prepared = j->expected_server_id;
      const bool strict = j->strict;
      const unsigned log = ++s->logs;
      const uint64_t first_target = markers ? j->markers[0].value : 0;
      const uint64_t first_actual = markers ? j->markers[0].observed : 0;
      mtx_unlock(&s->mutex);
      job_trace_emit("COMPLETE", &j->trace);
      if (log <= 32 || (log % 512 == 0 && log <= 16384))
         npt_log("C12 %s pid=%ld ctx=%u ring=0 prepared_seq=%" PRIu64 " seq=%" PRIu64 " markers=%u strict=%u publisher=%" PRIu64 " generation=%" PRIu64 " serial=%" PRIu64 " tail=%u target0=%" PRIu64 " actual0=%" PRIu64 " all_reached=1 finite=1",
                 markers ? "GPU_JOB_POSITIVE" : "GPU_JOB_EMPTY", (long)getpid(),
                 s->ctx->ctx_id, prepared, fence, markers, strict,
                 j->publisher_ring, j->publisher_generation, j->publisher_serial,
                 j->publisher_tail, first_target, first_actual);
      /* ALL ring0 responses pass this FIFO, even non-strict CPU entries. */
      s->ctx->retire_fence(s->ctx->ctx_id, 0, fence);
      mtx_destroy(&j->mutex); free(j);
      mtx_lock(&s->mutex);
   }
   mtx_unlock(&s->mutex); return 0;
}

bool npt_job_enable(struct npt_context *ctx)
{
   if (ctx->gpu_jobs) return !ctx->gpu_jobs->admission_closed &&
      !atomic_load_explicit(&ctx->gpu_jobs->poison, memory_order_acquire);
   if (atomic_load_explicit(&job_worker_poison, memory_order_acquire) ||
       ctx->gpu_job_legacy_ring_seen || !list_is_empty(&ctx->rings)) return false;
   struct npt_gpu_jobs *s = calloc(1, sizeof(*s));
   if (!s) return false;
   s->ctx = ctx; list_inithead(&s->jobs); list_inithead(&s->natives);
   const char *video = getenv("NPT_NATIVE_JOB_VIDEO_RETIRE");
   s->video_retire = video && strcmp(video, "1") == 0;
   if (mtx_init(&s->mutex, mtx_plain) != thrd_success) { free(s); return false; }
   if (job_cond_init(&s->cond) != thrd_success) { mtx_destroy(&s->mutex); free(s); return false; }
   if (thrd_create(&s->thread, job_thread, s) != thrd_success) {
      cnd_destroy(&s->cond); mtx_destroy(&s->mutex); free(s); return false;
   }
   ctx->gpu_jobs = s; return true;
}

/* Main-channel caller serializes context control. V2 initialization alone
 * is not a capability ACK: no legacy bytes may ever have auto-consumed. */
bool npt_job_activate(struct npt_context *ctx)
{
   struct npt_gpu_jobs *s = ctx->gpu_jobs;
   if (!s || ctx->gpu_job_legacy_ring_seen || s->admission_closed ||
       atomic_load_explicit(&s->poison, memory_order_acquire)) return false;
   bool valid = true, any = false;
   mtx_lock(&ctx->ring_mutex);
   list_for_each_entry(struct npt_ring, ring, &ctx->rings, head) {
      any = true;
      if (!ring->gpu_jobs || !ring->gpu_jobs->publisher) valid = false;
   }
   mtx_unlock(&ctx->ring_mutex);
   if (valid && any) s->activated = true;
   return valid && any;
}
bool npt_job_active(struct npt_context *ctx)
{
   return ctx->gpu_jobs && ctx->gpu_jobs->activated;
}

/* Close admission, then sample existing work without blocking the control
 * worker. A true result is followed by normal native destroy + thread joins
 * BEFORE the server sends its DRAIN ACK. No legacy or lost-device proof. */
bool npt_job_quiescent(struct npt_context *ctx)
{
   struct npt_gpu_jobs *s = ctx->gpu_jobs;
   if (!s || !s->activated || ctx->gpu_job_legacy_ring_seen) return false;
   mtx_lock(&s->mutex);
   s->admission_closed = true;
   bool ready = !s->prepared && !s->building && list_is_empty(&s->jobs) &&
      !atomic_load_explicit(&s->poison, memory_order_acquire);
   mtx_unlock(&s->mutex);
   if (!ready) return false;
   mtx_lock(&ctx->ring_mutex);
   list_for_each_entry(struct npt_ring, ring, &ctx->rings, head) {
      struct npt_job_ring *r = ring->gpu_jobs;
      if (!r || !r->publisher) { ready = false; break; }
      mtx_lock(&r->mutex);
      const uint32_t limit = atomic_load_explicit(&r->credit.tail, memory_order_acquire);
      if (r->active || !list_is_empty(&r->checkpoints) ||
          npt_ring_load_head(ring) != limit || npt_ring_load_tail(ring) != limit)
         ready = false;
      mtx_unlock(&r->mutex);
      if (!ready) break;
   }
   mtx_unlock(&ctx->ring_mutex);
   if (!ready) return false;
   /* No decoder can authorize or issue additional work now: credit is fixed,
    * all authorized bytes drained and every exact checkpoint has a marker. */
   mtx_lock(&s->mutex);
   list_for_each_entry(struct job_native, n, &s->natives, head) {
      PFN_ID3D11Fence_GetCompletedValue get = NPT_COM_VTBL_FUNC(
         PFN_ID3D11Fence_GetCompletedValue, npt_com_vtable(n->fence), NPT_VTBL_ID3D11Fence_GetCompletedValue);
      const uint64_t actual = get(n->fence);
      if (actual == UINT64_MAX) { job_fail(s, "DRAIN native fence returned UINT64_MAX"); ready = false; break; }
      if (!job_native_healthy(s, n, "DRAIN native device reported non-S_OK status")) { ready = false; break; }
      if (actual < n->value) { ready = false; break; }
   }
   ready = ready && !atomic_load_explicit(&s->poison, memory_order_acquire);
   mtx_unlock(&s->mutex);
   return ready;
}

bool npt_job_prepare(struct npt_context *ctx, uint64_t expected_server_fence_id)
{
   struct npt_gpu_jobs *s = ctx->gpu_jobs;
   if (!s || !expected_server_fence_id || expected_server_fence_id == UINT64_MAX || atomic_load_explicit(&s->poison, memory_order_acquire)) return false;
   struct native_job *j = calloc(1, sizeof(*j));
   if (!j || mtx_init(&j->mutex, mtx_plain) != thrd_success) { free(j); job_fail(s, "job allocation failed"); return false; }
   j->expected_server_id = expected_server_fence_id; j->strict = true;
   j->deadline = npt_profile_now_ns() + JOB_DEADLINE_NS;
   mtx_lock(&s->mutex);
   if (!s->activated || s->admission_closed || s->prepared || s->building || s->stopping) {
      mtx_unlock(&s->mutex); mtx_destroy(&j->mutex); free(j); job_fail(s, "invalid prepare/admission"); return false;
   }
   s->prepared = j; list_addtail(&j->head, &s->jobs); ++s->count;
   mtx_unlock(&s->mutex); return true;
}

bool npt_job_submit_begin(struct npt_context *ctx)
{
   struct npt_gpu_jobs *s = ctx->gpu_jobs;
   if (!s) return true;
   mtx_lock(&s->mutex);
   if (s->admission_closed || s->building || (s->prepared && s->prepared->submitted)) {
      mtx_unlock(&s->mutex); job_fail(s, "prepared job reused by another submit"); return false;
   }
   s->building = s->prepared;
   mtx_unlock(&s->mutex);
   return !atomic_load_explicit(&s->poison, memory_order_acquire);
}
void npt_job_submit_end(struct npt_context *ctx, bool ok)
{
   struct npt_gpu_jobs *s = ctx->gpu_jobs;
   if (!s) return;
   mtx_lock(&s->mutex);
   if (s->building) {
      mtx_lock(&s->building->mutex);
      if (!s->building->publisher_serial) ok = false;
      s->building->submitted = true;
      mtx_unlock(&s->building->mutex);
   }
   s->building = NULL; cnd_signal(&s->cond); mtx_unlock(&s->mutex);
   if (!ok) job_fail(s, "prepared command dispatch failed");
}

bool npt_job_fence(struct npt_context *ctx, uint32_t flags, uint64_t fence_id)
{
   struct npt_gpu_jobs *s = ctx->gpu_jobs;
   if (!s || atomic_load_explicit(&s->poison, memory_order_acquire)) return false;
   mtx_lock(&s->mutex);
   struct native_job *j = s->prepared;
   if (flags & NPT_GPU_JOB_COMPLETE) {
      if (!j || !j->submitted || j->assigned || j->expected_server_id != fence_id) { mtx_unlock(&s->mutex); job_fail(s, "strict fence without exact prepared job"); return false; }
      s->prepared = NULL;
   } else {
      /* Do not pop/convert a requested job with a normal CPU fence. */
      if (j) { mtx_unlock(&s->mutex); job_fail(s, "CPU fence interposed before prepared job fence"); return false; }
      j = calloc(1, sizeof(*j));
      if (!j || mtx_init(&j->mutex, mtx_plain) != thrd_success) { free(j); mtx_unlock(&s->mutex); job_fail(s, "CPU FIFO allocation failed"); return false; }
      j->submitted = true; j->deadline = npt_profile_now_ns() + JOB_DEADLINE_NS;
      list_addtail(&j->head, &s->jobs); ++s->count;
   }
   mtx_lock(&j->mutex); j->fence_id = fence_id; j->assigned = true; mtx_unlock(&j->mutex);
   cnd_signal(&s->cond); mtx_unlock(&s->mutex); return true;
}

bool npt_job_ring_init(struct npt_ring *ring)
{
   if (!ring->context->gpu_jobs) return true;
   struct npt_job_ring *r = calloc(1, sizeof(*r));
   if (!r || mtx_init(&r->mutex, mtx_plain) != thrd_success) { free(r); return false; }
   list_inithead(&r->checkpoints); ring->gpu_jobs = r; return true;
}

static _Thread_local struct npt_ring *job_dispatch_ring;
bool npt_job_ring_before(struct npt_ring *ring, const struct npt_command_header *h)
{
   (void)h;
   job_dispatch_ring = ring;
   struct npt_job_ring *r = ring->gpu_jobs;
   if (!r || !r->publisher) return true;
   if (atomic_load_explicit(&ring->context->gpu_jobs->poison, memory_order_acquire)) {
      job_dispatch_ring = NULL; return false;
   }
   mtx_lock(&r->mutex);
   r->active = true;
   const struct job_checkpoint *trace_p = list_is_empty(&r->checkpoints) ? NULL :
      list_first_entry(&r->checkpoints, struct job_checkpoint, head);
   r->trace_collect = trace_p && trace_p->job->trace.sample;
   mtx_unlock(&r->mutex);
   return true;
}
bool npt_job_note_dispatch(struct npt_context *ctx, struct npt_cs_decoder *dec,
                            const struct npt_command_header *h)
{
   struct npt_ring *ring = job_dispatch_ring;
   if (!ring || ring->context != ctx || !ring->gpu_jobs || !ring->gpu_jobs->publisher) {
      if (ctx->gpu_jobs) {
         const bool native_call = (h->cmd_type >> 24) != 0;
         const bool resource = npt_transport_subgroup(h) == NPT_TRANSPORT_SUBGROUP_RESOURCE;
         const uint32_t method = npt_transport_method(h);
         if (native_call || (resource && method != NPT_TRANSPORT_RESOURCE_BIND_D3D11_BUFFER_SHMEM)) {
            job_fail(ctx->gpu_jobs, "native GPU-capable command outside credited publisher ring");
            return false;
         }
      }
      return true;
   }
   struct npt_job_ring *r = ring->gpu_jobs;
   struct npt_gpu_jobs *s = ctx->gpu_jobs;
   const uint32_t group = h->cmd_type >> 24;
   if (group == 3 || (group == 2 && (h->cmd_type & 0xffffffu) != 0)) {
      job_fail(s, "D3D12/On12/combined-swapchain creation outside D3D11 graphics capability");
      return false;
   }
   if (group == 255) {
      const uint16_t iface = (h->cmd_type >> 8) & 0xffffu;
      void *dc = NULL;
      ID3D11DeviceContext *owned_dc = NULL;
      void *owned_device = NULL;
      const bool is_context = iface == NPT_IFACE_ID_ID3D11DeviceContext || iface == NPT_IFACE_ID_ID3D11DeviceContext1 ||
          iface == NPT_IFACE_ID_ID3D11DeviceContext2 || iface == NPT_IFACE_ID_ID3D11DeviceContext3 || iface == NPT_IFACE_ID_ID3D11DeviceContext4;
      const bool is_device = iface == NPT_IFACE_ID_ID3D11Device || iface == NPT_IFACE_ID_ID3D11Device1 ||
          iface == NPT_IFACE_ID_ID3D11Device2 || iface == NPT_IFACE_ID_ID3D11Device3 ||
          iface == NPT_IFACE_ID_ID3D11Device4 || iface == NPT_IFACE_ID_ID3D11Device5;
      const bool is_video = iface == NPT_IFACE_ID_ID3D11VideoContext || iface == NPT_IFACE_ID_ID3D11VideoContext1 ||
          iface == NPT_IFACE_ID_ID3D11VideoContext2 || iface == NPT_IFACE_ID_ID3D11VideoContext3;
      if (r->trace_collect && (is_context || is_device || is_video)) {
         ++r->trace_native_calls;
         if (is_video) ++r->trace_video_calls;
         r->trace_last_iface = iface;
         r->trace_last_method = h->cmd_type & 0xffu;
      }
      if (is_video && !s->video_retire) {
         /* Default remains graphics-only. Opt-in requires the paired native
          * backend's terminal status and CS-ordered output-copy contract. */
         job_fail(s, "VideoContext completion outside initial D3D11 graphics capability");
         return false;
      }
      if (is_context)
         dc = npt_context_lookup_object(ring->context, dec, h->object_id, NPT_OBJECT_TYPE_ID3D11DEVICECONTEXT);
      else if (is_device || is_video) {
         void *obj = npt_context_lookup_object(ring->context, dec, h->object_id,
            is_device ? NPT_OBJECT_TYPE_ID3D11DEVICE : NPT_OBJECT_TYPE_ID3D11VIDEOCONTEXT);
         if (obj && is_device) owned_device = job_qi(obj, &NPT_IID_ID3D11Device);
         else if (obj) {
            ID3D11Device *device = NULL;
            PFN_ID3D11DeviceChild_GetDevice get_device = NPT_COM_VTBL_FUNC(
               PFN_ID3D11DeviceChild_GetDevice, npt_com_vtable(obj), NPT_VTBL_ID3D11DeviceChild_GetDevice);
            get_device(obj, &device); owned_device = device;
         }
         if (owned_device) {
            PFN_ID3D11Device_GetImmediateContext get_context = NPT_COM_VTBL_FUNC(
               PFN_ID3D11Device_GetImmediateContext, npt_com_vtable(owned_device), NPT_VTBL_ID3D11Device_GetImmediateContext);
            get_context(owned_device, &owned_dc); dc = owned_dc;
         }
      } else if (iface == NPT_IFACE_ID_ID3D12CommandQueue) {
         job_fail(s, "D3D12 queue is outside typed D3D11 GPU-job capability");
      }
      if (is_context || is_device || is_video) {
         if (dc) {
            struct job_native *n = job_native_get(s, ring, dc);
            if (n && is_video && !job_native_video_capable(s, n)) n = NULL;
            if (n) {
               bool seen = false; for (unsigned i = 0; i < r->touched_count; ++i) seen |= r->touched[i] == n;
               if (!seen) {
                  if (r->touched_count >= JOB_MAX_NATIVE) job_fail(s, "touched context bound exceeded");
                  else r->touched[r->touched_count++] = n;
               }
            }
         } else job_fail(s, "typed native device/context identity missing");
      }
      if (owned_dc) npt_com_release(owned_dc);
      if (owned_device) npt_com_release(owned_device);
   }
   if ((h->cmd_type >> 24) == 0 && npt_transport_subgroup(h) == NPT_TRANSPORT_SUBGROUP_RESOURCE) {
      uint64_t context_id = 0;
      uint64_t resource_id = h->object_id;
      const uint32_t method = npt_transport_method(h);
      if (r->trace_collect) ++r->trace_resource_calls;
      const bool update = method == NPT_TRANSPORT_RESOURCE_UPDATE;
      const bool map = method == NPT_TRANSPORT_RESOURCE_MAP || method == NPT_TRANSPORT_RESOURCE_UNMAP;
      if (update) {
         struct npt_cmd_resource_update cmd = {0};
         if (h->cmd_size < sizeof(cmd)) job_fail(s, "short native resource update");
         else { npt_cs_decoder_peek(dec, sizeof(cmd), &cmd, sizeof(cmd)); context_id = cmd.context_id; }
      } else if (map) {
         struct { struct npt_command_header header; uint64_t context_id, resource_id; } cmd = {0};
         if (h->cmd_size < sizeof(cmd)) job_fail(s, "short native map/unmap");
         else { npt_cs_decoder_peek(dec, sizeof(cmd), &cmd, sizeof(cmd)); context_id = cmd.context_id; resource_id = cmd.resource_id; }
      }
      if (update || map) {
         void *dc = NULL;
         ID3D11Device *device = NULL;
         ID3D11DeviceContext *owned_dc = NULL;
         if (context_id) dc = npt_context_lookup_object(ctx, dec, context_id, NPT_OBJECT_TYPE_ID3D11DEVICECONTEXT);
         else if (npt_context_object_is(ctx, resource_id, NPT_OBJECT_TYPE_ID3D11RESOURCE)) {
            void *res = npt_context_lookup_object(ctx, dec, resource_id, NPT_OBJECT_TYPE_ID3D11RESOURCE);
            if (res) {
               PFN_ID3D11DeviceChild_GetDevice get_device = NPT_COM_VTBL_FUNC(
                  PFN_ID3D11DeviceChild_GetDevice, npt_com_vtable(res), NPT_VTBL_ID3D11DeviceChild_GetDevice);
               get_device(res, &device);
               if (device) {
                  PFN_ID3D11Device_GetImmediateContext get_context = NPT_COM_VTBL_FUNC(
                     PFN_ID3D11Device_GetImmediateContext, npt_com_vtable(device), NPT_VTBL_ID3D11Device_GetImmediateContext);
                  get_context(device, &owned_dc); dc = owned_dc;
               }
            }
         }
         if (!dc) job_fail(s, "resource GPU operation has no typed D3D11 context");
         else {
            struct job_native *n = job_native_get(s, ring, dc);
            if (n) {
               bool seen = false; for (unsigned i = 0; i < r->touched_count; ++i) seen |= r->touched[i] == n;
               if (!seen) {
                  if (r->touched_count >= JOB_MAX_NATIVE) job_fail(s, "transport touched-context bound");
                  else r->touched[r->touched_count++] = n;
               }
            }
         }
         if (owned_dc) npt_com_release(owned_dc);
         if (device) npt_com_release(device);
      } else if (method == NPT_TRANSPORT_RESOURCE_CREATE_HEAP_FROM_SHMEM) {
         job_fail(s, "D3D12 heap path outside typed D3D11 GPU jobs");
      }
   }
   return !atomic_load_explicit(&s->poison, memory_order_acquire);
}

/* Constructor commands have no pre-existing object_id. Capture the real
 * returned immediate DC after successful creation, before the boundary tail. */
void npt_job_note_created_device(struct npt_context *ctx, void *device, void *immediate)
{
   if (!ctx->gpu_jobs) return;
   struct npt_ring *ring = job_dispatch_ring;
   struct npt_gpu_jobs *s = ctx->gpu_jobs;
   if (!ring || ring->context != ctx || !ring->gpu_jobs || !ring->gpu_jobs->publisher) {
      job_fail(s, "native device created outside publisher checkpoint"); return;
   }
   ID3D11DeviceContext *owned_dc = NULL;
   if (!immediate && device) {
      PFN_ID3D11Device_GetImmediateContext get = NPT_COM_VTBL_FUNC(
         PFN_ID3D11Device_GetImmediateContext, npt_com_vtable(device), NPT_VTBL_ID3D11Device_GetImmediateContext);
      get(device, &owned_dc); immediate = owned_dc;
   }
   if (!immediate) job_fail(s, "device constructor returned no capturable immediate context");
   else {
      struct job_native *n = job_native_get(s, ring, immediate);
      struct npt_job_ring *r = ring->gpu_jobs;
      if (n) {
         bool seen = false;
         for (unsigned i = 0; i < r->touched_count; ++i) seen |= r->touched[i] == n;
         if (!seen) {
            if (r->touched_count >= JOB_MAX_NATIVE) job_fail(s, "constructor touched-context bound");
            else r->touched[r->touched_count++] = n;
         }
      }
   }
   if (owned_dc) npt_com_release(owned_dc);
}

void npt_job_ring_after(struct npt_ring *ring, uint32_t head, bool ok)
{
   job_dispatch_ring = NULL;
   struct npt_job_ring *r = ring->gpu_jobs;
   if (!r || !r->publisher) return;
   struct npt_gpu_jobs *s = ring->context->gpu_jobs;
   if (!ok) job_fail(s, "ring command dispatch failed");
   if (atomic_load_explicit(&s->poison, memory_order_acquire)) goto idle;
   for (;;) {
      mtx_lock(&r->mutex);
      struct job_checkpoint *p = list_is_empty(&r->checkpoints) ? NULL :
         list_first_entry(&r->checkpoints, struct job_checkpoint, head);
      const int boundary = p ? npt_job_credit_boundary(head, p->target) : 0;
      if (!p || boundary == 0) { mtx_unlock(&r->mutex); break; }
      if (boundary < 0) { mtx_unlock(&r->mutex); job_fail(s, "publisher target splits command"); break; }
      list_del(&p->head); mtx_unlock(&r->mutex);
      /* Only this ring's decode thread issues the tail, outside metadata
       * locks and BEFORE it can dispatch any later command (including Wait).
       * The control worker never waits behind a native Map/Wait/Signal. */
      job_checkpoint_signal(s, r, p->job); free(p);
   }
idle:
   mtx_lock(&r->mutex); r->active = false; mtx_unlock(&r->mutex);
}

bool npt_job_notify(struct npt_context *ctx, struct npt_ring *ring)
{
   struct npt_gpu_jobs *s = ctx->gpu_jobs;
   if (!s) return true;
   mtx_lock(&s->mutex); const bool requested = s->building != NULL; mtx_unlock(&s->mutex);
   if (requested || (ring->gpu_jobs && ring->gpu_jobs->publisher)) {
      job_fail(s, "legacy Notify cannot identify a requested publisher job"); return false;
   }
   return !atomic_load_explicit(&s->poison, memory_order_acquire);
}

bool npt_job_ring_enable_publish(struct npt_ring *ring, uint64_t generation)
{
   struct npt_job_ring *r = ring->gpu_jobs;
   if (!r || !npt_job_credit_start_valid(generation, ring->started,
       npt_ring_load_head(ring), ring->buffer.cur, npt_ring_load_tail(ring),
       ring->buffer.size)) return false;
   npt_job_credit_init(&r->credit, generation);
   r->publisher = true; return true;
}
uint32_t npt_job_ring_limit(struct npt_ring *ring)
{
   struct npt_job_ring *r = ring->gpu_jobs;
   return r && r->publisher ? atomic_load_explicit(&r->credit.tail, memory_order_acquire) : npt_ring_load_tail(ring);
}

bool npt_job_notify_v2(struct npt_context *ctx, struct npt_ring *ring,
                       uint64_t generation, uint64_t serial, uint32_t tail)
{
   struct npt_gpu_jobs *s = ctx->gpu_jobs;
   struct npt_job_ring *r = ring->gpu_jobs;
   if (!s || !s->activated || s->admission_closed || !r || !r->publisher) return false;
   mtx_lock(&s->mutex); struct native_job *j = s->building; mtx_unlock(&s->mutex);
   if (!j) { job_fail(s, "NotifyV2 without requested KMD prepare"); return false; }
   struct job_checkpoint *p = calloc(1, sizeof(*p));
   if (!p) { job_fail(s, "checkpoint allocation failed"); return false; }
   /* Marker metadata lookup is outside all existing job/ring mutexes. */
   const unsigned trace_sample = job_trace_take();
   mtx_lock(&r->mutex);
   /* Snapshot publisher first: decoder may advance head while the publisher
    * refills. An old head paired with a newer tail can falsely exceed size.
    * This main channel has not granted the next credit yet, so head cannot
    * pass the old credit (which is at/below this sampled publication). */
   const uint32_t published = npt_ring_load_tail(ring);
   const uint32_t head = npt_ring_load_head(ring);
   if (!npt_job_credit_valid(&r->credit, generation, serial, tail, head, published, ring->buffer.size)) {
      mtx_unlock(&r->mutex); free(p); job_fail(s, "stale/gapped publisher generation/serial/tail"); return false;
   }
   p->job = j; p->target = tail;
   mtx_lock(&j->mutex);
   if (j->publisher_serial) {
      mtx_unlock(&j->mutex); mtx_unlock(&r->mutex); free(p);
      job_fail(s, "requested job contains more than one publisher NotifyV2"); return false;
   }
   j->publisher_ring = ring->id; j->publisher_generation = generation;
   j->publisher_serial = serial; j->publisher_tail = tail; ++j->pending;
   if (trace_sample) {
      j->trace = (struct job_trace_record){ .sample = trace_sample,
         .ctx = ctx->ctx_id, .prepared = j->expected_server_id,
         .publisher = ring->id, .generation = generation, .serial = serial,
         .tail = tail, .notify_ns = npt_profile_now_ns() };
   }
   mtx_unlock(&j->mutex);
   list_addtail(&p->head, &r->checkpoints);
   /* Publish the exact checkpoint BEFORE authorizing any of its bytes. */
   npt_job_credit_publish(&r->credit, tail);
   mtx_unlock(&r->mutex);
   return !atomic_load_explicit(&s->poison, memory_order_acquire);
}

void npt_job_ring_fini(struct npt_ring *ring)
{
   struct npt_job_ring *r = ring->gpu_jobs;
   if (!r) return;
   if (!list_is_empty(&r->checkpoints)) {
      job_fail(ring->context->gpu_jobs, "ring destroyed before checkpoint");
      /* Keep checkpoint/job refs; context enters failstop quarantine. */
      return;
   }
   mtx_destroy(&r->mutex); free(r); ring->gpu_jobs = NULL;
}

bool npt_job_context_fini(struct npt_context *ctx)
{
   struct npt_gpu_jobs *s = ctx->gpu_jobs;
   if (!s) return true;
   mtx_lock(&s->mutex);
   s->stopping = true; cnd_broadcast(&s->cond); mtx_unlock(&s->mutex);
   thrd_join(s->thread, NULL);
   if (!list_is_empty(&s->jobs) || atomic_load_explicit(&s->poison, memory_order_acquire)) {
      job_fail(s, "context close with pending/unknown GPU jobs; retained until VM exit");
      return false;
   }
   list_for_each_entry_safe(struct job_native, n, &s->natives, head) {
      npt_com_release(n->fence); npt_com_release(n->device5);
      npt_com_release(n->context4); npt_com_release(n->identity);
      list_del(&n->head); mtx_destroy(&n->mutex); free(n);
   }
   cnd_destroy(&s->cond); mtx_destroy(&s->mutex); free(s); ctx->gpu_jobs = NULL; return true;
}
