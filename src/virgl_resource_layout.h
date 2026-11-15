/* SPDX-License-Identifier: MIT
 * NEW internal protocol v1. Metadata for the exact FD exported by the
 * classic pipe-resource exporter. Not an ownership grant or GPU fence.
 */
#ifndef VIRGL_RESOURCE_LAYOUT_H
#define VIRGL_RESOURCE_LAYOUT_H
#include <stdint.h>
#define VIRGL_IMPORT_LAYOUT_MAGIC 0x314c4956u /* VIL1 */
#define VIRGL_IMPORT_LAYOUT_VERSION 1u
#define VIRGL_IMPORT_LAYOUT_CLASSIC 1u
#define VIRGL_IMPORT_LAYOUT_INVALID_MODIFIER UINT64_C(0x00ffffffffffffff)
#define VIRGL_IMPORT_FOURCC(a,b,c,d) ((uint32_t)(a) | ((uint32_t)(b)<<8) | ((uint32_t)(c)<<16) | ((uint32_t)(d)<<24))
struct virgl_import_layout_v1 {
   uint32_t magic, version, size, flags;
   uint32_t fourcc, plane_count, reserved0, reserved1;
   uint64_t modifier, allocation_size;
   struct { uint64_t offset, pitch; } planes[4];
};
#ifdef __cplusplus
static_assert(sizeof(struct virgl_import_layout_v1) == 112, "layout v1 wire");
#else
_Static_assert(sizeof(struct virgl_import_layout_v1) == 112, "layout v1 wire");
#endif
/* Initial bounded route: one FD, one plane, three directly representable TYPE_3D RGB32
 * formats. Multi-plane/unknown formats fail explicitly; never synthesize.
 */
static inline uint32_t
virgl_import_layout_dxgi_format(const struct virgl_import_layout_v1 *l)
{
   switch (l->fourcc) {
   case VIRGL_IMPORT_FOURCC('A','R','2','4'): return 87; /* BGRA8 */
   case VIRGL_IMPORT_FOURCC('X','R','2','4'): return 88; /* BGRX8 */
   case VIRGL_IMPORT_FOURCC('A','B','2','4'): return 28; /* RGBA8 */
   default: return 0;
   }
}
static inline int
virgl_import_layout_valid(const struct virgl_import_layout_v1 *l)
{
   if (!l || l->magic != VIRGL_IMPORT_LAYOUT_MAGIC || l->version != 1 ||
       l->size != sizeof(*l) || l->flags != VIRGL_IMPORT_LAYOUT_CLASSIC ||
       l->plane_count != 1 || l->reserved0 || l->reserved1 ||
       !l->allocation_size || !l->planes[0].pitch ||
       l->planes[0].offset >= l->allocation_size ||
       l->modifier == VIRGL_IMPORT_LAYOUT_INVALID_MODIFIER ||
       !virgl_import_layout_dxgi_format(l)) return 0;
   for (unsigned i=1;i<4;i++)
      if (l->planes[i].offset || l->planes[i].pitch) return 0;
   return 1;
}
static inline int
virgl_import_layout_matches(const struct virgl_import_layout_v1 *l,
                            uint32_t width, uint32_t height, uint32_t dxgi_format)
{
   if (!virgl_import_layout_valid(l) || !width || !height ||
       virgl_import_layout_dxgi_format(l) != dxgi_format) return 0;
   const uint64_t row=(uint64_t)width*4u, pitch=l->planes[0].pitch;
   const uint64_t offset=l->planes[0].offset;
   if (row>pitch) return 0;
   if (l->modifier == 0 &&
       (row>l->allocation_size-offset ||
        (uint64_t)(height-1)>(l->allocation_size-offset-row)/pitch)) return 0;
   /* Non-linear modifiers: only structural bounds above. Native Vulkan
    * validates modifier-specific image/memory requirements; no linear
    * height*pitch claim for tiled allocations. */
   return 1;
}
#endif
