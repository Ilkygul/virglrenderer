/* C12 internal native GPU-job retirement. No guest COM/wire ABI. */
#ifndef NPT_JOB_H
#define NPT_JOB_H
#include "npt_common.h"
struct npt_command_header;
struct npt_cs_decoder;
#define NPT_GPU_JOB_MAGIC 0x4e4a4331u
#define NPT_GPU_JOB_COMPLETE 0x20000000u
bool npt_job_enable(struct npt_context *ctx);
bool npt_job_activate(struct npt_context *ctx);
bool npt_job_active(struct npt_context *ctx);
bool npt_job_quiescent(struct npt_context *ctx);
bool npt_job_prepare(struct npt_context *ctx, uint64_t frontend_fence_id);
bool npt_job_submit_begin(struct npt_context *ctx);
void npt_job_submit_end(struct npt_context *ctx, bool ok);
bool npt_job_fence(struct npt_context *ctx, uint32_t flags, uint64_t fence_id);
bool npt_job_ring_init(struct npt_ring *ring);
void npt_job_ring_fini(struct npt_ring *ring);
bool npt_job_ring_before(struct npt_ring *ring, const struct npt_command_header *h);
void npt_job_note_created_device(struct npt_context *ctx, void *device, void *immediate);
bool npt_job_note_dispatch(struct npt_context *ctx, struct npt_cs_decoder *dec,
                            const struct npt_command_header *h);
void npt_job_ring_after(struct npt_ring *ring, uint32_t head, bool ok);
bool npt_job_notify(struct npt_context *ctx, struct npt_ring *ring);
bool npt_job_ring_enable_publish(struct npt_ring *ring, uint64_t generation);
uint32_t npt_job_ring_limit(struct npt_ring *ring);
bool npt_job_notify_v2(struct npt_context *ctx, struct npt_ring *ring,
                       uint64_t generation, uint64_t serial, uint32_t tail);
/* False retains the whole context in a bounded failed-worker quarantine;
 * no success callback and no context/object/backing destruction is permitted. */
bool npt_job_context_fini(struct npt_context *ctx);
#endif
