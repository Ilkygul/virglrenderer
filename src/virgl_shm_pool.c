/*
 * DroidVM: shared venus_host pool allocator (split out of vkr_context.c so the
 * proxy/Neptune path can use the same boot-shared region).
 * SPDX-License-Identifier: MIT
 */
#include "virgl_shm_pool.h"

#include <inttypes.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <unistd.h>

#include "c11/threads.h"
#include "util/list.h"
#include "virgl_util.h"

static void
pool_log(const char *fmt, ...)
{
   va_list va;
   va_start(va, fmt);
   virgl_prefixed_logv("shm-pool", VIRGL_LOG_LEVEL_INFO, fmt, va);
   va_end(va);
}

/* DroidVM venus_host pool: serve blob_id==0 transport shmems (rings, CS and
 * reply buffers) out of the boot-blessed venus_host region instead of fresh
 * memfds.  The VMM hands the region over via env (VENUS_POOL_FD/_FD_OFFSET/
 * _SIZE, exported before the GPU device starts); vkr keeps one process-
 * lifetime mapping and sub-allocates with a tiny first-fit free list.  The
 * host picks every offset -- the guest maps pool_base+offset from the
 * RESOURCE_MAP_BLOB response (MAP_INFO_POOL) and never chooses a location,
 * and no blob needs a runtime SHARE/accept round trip.  Exhaustion or a
 * missing env falls back to the original memfd path.
 */
struct virgl_shm_pool_hole {
   struct list_head head;
   uint64_t off;
   uint64_t size;
};

static struct {
   mtx_t mutex;
   uint8_t *base;
   uint64_t size;
   int fd;
   uint64_t fd_offset;
   bool init_done;
   struct list_head holes;
} virgl_shm_pool = { .fd = -1 };

static void
virgl_shm_pool_init_locked(void)
{
   if (virgl_shm_pool.init_done)
      return;
   virgl_shm_pool.init_done = true;
   list_inithead(&virgl_shm_pool.holes);

   const char *fd_str = getenv("VENUS_POOL_FD");
   const char *off_str = getenv("VENUS_POOL_FD_OFFSET");
   const char *size_str = getenv("VENUS_POOL_SIZE");
   if (!fd_str || !size_str)
      return;

   const int fd = atoi(fd_str);
   const uint64_t fd_offset = off_str ? strtoull(off_str, NULL, 0) : 0;
   const uint64_t size = strtoull(size_str, NULL, 0);
   if (fd < 0 || !size)
      return;

   /* Use the VMM's own mapping of the window (same process), NOT a private
    * mmap: the VMM recognises pool residency by checking blob.map_ptr against
    * VENUS_POOL_HOST_VA..+SIZE, so the pointers we publish must come from
    * that exact mapping.  A second mapping of the same pages has a different
    * VA and every probe misses -- the first deployment did exactly that, the
    * guest never got MAP_INFO_POOL and every ring came up unusable.
    */
   void *base = NULL;
   const char *va_str = getenv("VENUS_POOL_HOST_VA");
   if (va_str)
      base = (void *)(uintptr_t)strtoull(va_str, NULL, 0);
   if (!base) {
      pool_log("venus_host pool: VENUS_POOL_HOST_VA missing, pool disabled");
      return;
   }

   struct virgl_shm_pool_hole *hole = malloc(sizeof(*hole));
   if (!hole) {
      munmap(base, size);
      return;
   }
   hole->off = 0;
   hole->size = size;
   list_addtail(&hole->head, &virgl_shm_pool.holes);

   virgl_shm_pool.base = base;
   virgl_shm_pool.size = size;
   virgl_shm_pool.fd = fd;
   virgl_shm_pool.fd_offset = fd_offset;
   pool_log("venus_host pool: %" PRIu64 " MiB at %p (fd %d off 0x%" PRIx64 ")",
           size >> 20, base, fd, fd_offset);
}

static once_flag virgl_shm_pool_once = ONCE_FLAG_INIT;

static void
virgl_shm_pool_once_init(void)
{
   mtx_init(&virgl_shm_pool.mutex, mtx_plain);
}

void *
virgl_shm_pool_alloc(uint64_t size, uint64_t *out_off)
{
   call_once(&virgl_shm_pool_once, virgl_shm_pool_once_init);
   void *ptr = NULL;

   mtx_lock(&virgl_shm_pool.mutex);
   virgl_shm_pool_init_locked();
   if (virgl_shm_pool.base) {
      list_for_each_entry (struct virgl_shm_pool_hole, hole, &virgl_shm_pool.holes,
                           head) {
         if (hole->size < size)
            continue;
         *out_off = hole->off;
         ptr = virgl_shm_pool.base + hole->off;
         hole->off += size;
         hole->size -= size;
         if (!hole->size) {
            list_del(&hole->head);
            free(hole);
         }
         break;
      }
      if (!ptr)
         pool_log("venus_host pool: exhausted (%" PRIu64 " wanted), memfd fallback",
                 size);
   }
   mtx_unlock(&virgl_shm_pool.mutex);
   return ptr;
}

/* returns false if ptr is not pool memory */
bool
virgl_shm_pool_free(void *ptr, uint64_t size)
{
   if (!virgl_shm_pool.base || (uint8_t *)ptr < virgl_shm_pool.base ||
       (uint8_t *)ptr >= virgl_shm_pool.base + virgl_shm_pool.size)
      return false;

   const uint64_t off = (uint8_t *)ptr - virgl_shm_pool.base;

   mtx_lock(&virgl_shm_pool.mutex);
   /* insert sorted, coalescing with neighbours */
   struct virgl_shm_pool_hole *next = NULL;
   list_for_each_entry (struct virgl_shm_pool_hole, hole, &virgl_shm_pool.holes,
                        head) {
      if (hole->off > off) {
         next = hole;
         break;
      }
   }
   struct list_head *insert_before = next ? &next->head : &virgl_shm_pool.holes;
   struct virgl_shm_pool_hole *prev = NULL;
   if (insert_before->prev != &virgl_shm_pool.holes)
      prev = list_container_of(insert_before->prev, prev, head);

   if (prev && prev->off + prev->size == off) {
      prev->size += size;
      if (next && prev->off + prev->size == next->off) {
         prev->size += next->size;
         list_del(&next->head);
         free(next);
      }
   } else if (next && off + size == next->off) {
      next->off = off;
      next->size += size;
   } else {
      struct virgl_shm_pool_hole *hole = malloc(sizeof(*hole));
      if (hole) {
         hole->off = off;
         hole->size = size;
         list_add(&hole->head, insert_before->prev);
      }
   }
   mtx_unlock(&virgl_shm_pool.mutex);
   return true;
}

int
virgl_shm_pool_dup_fd(void)
{
   if (virgl_shm_pool.fd < 0)
      return -1;
   return fcntl(virgl_shm_pool.fd, F_DUPFD_CLOEXEC, 0);
}

uint64_t
virgl_shm_pool_fd_offset(void)
{
   return virgl_shm_pool.fd_offset;
}

bool
virgl_shm_pool_available(void)
{
   call_once(&virgl_shm_pool_once, virgl_shm_pool_once_init);
   mtx_lock(&virgl_shm_pool.mutex);
   virgl_shm_pool_init_locked();
   const bool ok = virgl_shm_pool.base != NULL;
   mtx_unlock(&virgl_shm_pool.mutex);
   return ok;
}
