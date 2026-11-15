/*
 * DroidVM: shared venus_host pool allocator.
 * SPDX-License-Identifier: MIT
 */
#ifndef VIRGL_SHM_POOL_H
#define VIRGL_SHM_POOL_H

#include <stdbool.h>
#include <stdint.h>

/* True if VENUS_POOL_* env describes a usable pool in this process. */
bool virgl_shm_pool_available(void);

/* Sub-allocate size bytes (page multiple); returns the host VA inside the
 * VMM mapping at VENUS_POOL_HOST_VA, or NULL (no pool / exhausted). */
void *virgl_shm_pool_alloc(uint64_t size, uint64_t *out_off);

/* Returns false if ptr is not pool memory. */
bool virgl_shm_pool_free(void *ptr, uint64_t size);

/* New CLOEXEC dup of the pool fd, or -1. */
int virgl_shm_pool_dup_fd(void);

/* VENUS_POOL_FD_OFFSET: file offset of pool offset 0. */
uint64_t virgl_shm_pool_fd_offset(void);

#endif
