/* Private, versioned DXVK/Neptune feature response. Not a Windows SDK feature.
 * New implementation writes every byte; callers start with a zero POD and
 * require HRESULT S_OK plus all exact fields. Not a pixel-correctness claim.
 */
#ifndef NPT_D3D11_VIDEO_TERMINAL_H
#define NPT_D3D11_VIDEO_TERMINAL_H
#include <stdint.h>
#define NPT_D3D11_FEATURE_VIDEO_TERMINAL_V1 UINT32_C(0x4e505654)
#define NPT_D3D11_VIDEO_TERMINAL_MAGIC UINT32_C(0x3154564e)
#define NPT_D3D11_VIDEO_TERMINAL_VERSION UINT32_C(1)
#define NPT_D3D11_VIDEO_TERMINAL_SUPPORTED UINT32_C(1)
struct npt_d3d11_video_terminal_v1 {
   uint32_t magic, version, size, supported;
   uint32_t reserved[4];
};
#ifdef __cplusplus
static_assert(sizeof(struct npt_d3d11_video_terminal_v1) == 32, "video terminal v1 wire");
#else
_Static_assert(sizeof(struct npt_d3d11_video_terminal_v1) == 32, "video terminal v1 wire");
#endif
/* SUPPORTED v1: MediaCodec enqueue/expected-output/copy/backend failures
 * persist through the actual device's GetDeviceRemovedReason, and successful
 * target copies use the same ordered immediate-context CS/graphics prefix.
 * Hidden/no-output and malformed API input retain their existing semantics.
 * This is not decoder pixel validation, a new VPU fence, or a DMA ownership
 * grant. Unknown failures must not be retired as successful/empty GPU jobs.
 */
#endif
