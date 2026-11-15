/* Actual C12 credit predicate, shared with the local boundary unit test. */
#ifndef NPT_JOB_CREDIT_H
#define NPT_JOB_CREDIT_H
#include <stdbool.h>
#include <stdint.h>
#include <stdatomic.h>
struct npt_job_credit {
   uint64_t generation, next_serial;
   atomic_uint tail;
};
/* Shared tail is publisher-owned and may be populated before CREATE reaches
 * the host. Only head/decoder cursor are initially zero; credit stays zero. */
static inline bool npt_job_credit_start_valid(uint64_t generation, bool started,
   uint32_t head, uint32_t cursor, uint32_t published, uint32_t size)
{
   return generation && generation != UINT64_MAX && !started && !head && !cursor &&
      size && size <= 16u * 1024u * 1024u && published <= size;
}
static inline void npt_job_credit_init(struct npt_job_credit *c, uint64_t generation)
{
   c->generation = generation; c->next_serial = 1;
   atomic_init(&c->tail, 0);
}
/* Caller holds metadata mutex. Does not grant bytes or mutate serial. */
static inline bool npt_job_credit_valid(const struct npt_job_credit *c,
   uint64_t generation, uint64_t serial, uint32_t tail,
   uint32_t head, uint32_t published, uint32_t size)
{
   const uint32_t previous = atomic_load_explicit(&c->tail, memory_order_relaxed);
   const uint32_t delta = tail - previous;
   return generation && generation == c->generation && serial == c->next_serial &&
      serial != UINT64_MAX && size && size <= 16u * 1024u * 1024u &&
      delta && delta <= size && (uint32_t)(published - head) <= size &&
      (uint32_t)(tail - head) <= size &&
      (uint32_t)(tail - head) <= (uint32_t)(published - head);
}
/* Call ONLY after the exact checkpoint was inserted. */
static inline void npt_job_credit_publish(struct npt_job_credit *c, uint32_t tail)
{
   ++c->next_serial;
   atomic_store_explicit(&c->tail, tail, memory_order_release);
}
/* 0=pending, 1=exact boundary, -1=invalid command crossing boundary. */
static inline int npt_job_credit_boundary(uint32_t head, uint32_t target)
{
   return head == target ? 1 : ((int32_t)(head - target) < 0 ? 0 : -1);
}
#endif
