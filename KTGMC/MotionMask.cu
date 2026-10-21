#define _CRT_SECURE_NO_WARNINGS
#include "avisynth.h"

#define NOMINMAX
#include <windows.h>

#include <algorithm>
#include <cmath>
#include <mutex>
#include <vector>

#include <cuda_runtime_api.h>

#include "CommonFunctions.h"
#include "KMV.h"
#include "CudaKernelBase.h"

// ---------------------------------------------------------------------------
//  KMotionMask
//
//  CUDA implementation of the "MMask" filter of MVTools.
//
//  The mask is built from the motion vectors that KMAnalyse produced:
//
//    kind = 0 : the length of the motion vector
//    kind = 1 : the SAD of the block (sampled at an intermediate time)
//    kind = 3 : the x component of the motion vector
//    kind = 4 : the y component of the motion vector
//
//  The block sized mask is afterwards enlarged to the size of the picture
//  with the same bilinear filter that MVTools uses (SimpleResize of Fizick).
//  The formulas are taken from MVMask.cpp / MaskFun.cpp of MVTools so that
//  the result matches the CPU implementation of QTGMC.
// ---------------------------------------------------------------------------

namespace {

// The plugin keeps the vectors of all levels in one frame, the finest level
// comes last.  This is the offset of the finest level inside that frame.
inline int FinestLevelOffset(const KMVParam* params)
{
  int offset = 0;
  for (int i = 1; i < params->nAnalyzeLevels; ++i) {
    offset += params->levelInfo[i].nBlkX * params->levelInfo[i].nBlkY;
  }
  return offset;
}

// The SCD1 threshold, normalized like MVTools does it in MVClip::MVClip.
inline int NormalizedSCD1(const KMVParam* params, int thSCD1)
{
  int nSCD1 = std::min(thSCD1, 8 * 8 * 255); // max for 8 bit, normalized to 8x8 block size
  if (params->nPixelSize == 2) {
    nSCD1 = int(nSCD1 / 255.0 * ((1 << params->nBitsPerPixel) - 1));
  }
  nSCD1 = int((uint64_t)nSCD1 * (params->nBlkSizeX * params->nBlkSizeY) / (8 * 8));
  if (params->chroma) {
    // base: YV12, the chroma adds half of the luma area again
    int chromaScale = params->chromaSADScale;
    int sadChroma = (chromaScale == 0) ? nSCD1 * 2
                   : (chromaScale > 0) ? (nSCD1 * 2) >> chromaScale
                   : (nSCD1 * 2) << (-chromaScale);
    nSCD1 += sadChroma / 4;
  }
  return nSCD1;
}

// ---------------------------------------------------------------------------
//  kernels
// ---------------------------------------------------------------------------

// kind = 0: the length of the motion vector
__global__ void kl_mask_length(const VECTOR* mv, unsigned char* mask,
  int nBlkCount, int nPel, double normFactor2, double halfGamma, int scdThreshold, int* scCount)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= nBlkCount) return;

  int x = mv[i].x;
  int y = mv[i].y;
  double norme = double(x * x + y * y) / (double(nPel) * double(nPel));
  double l = 255.0 * ((halfGamma == 0.5) ? sqrt(norme * normFactor2)
                                        : pow(norme * normFactor2, halfGamma));
  mask[i] = (unsigned char)(l > 255.0 ? 255.0 : l);

  if (mv[i].sad > scdThreshold) {
    atomicAdd(scCount, 1);
  }
}

// kind = 3 and 4: the x or the y component of the motion vector
__global__ void kl_mask_vectorxy(const VECTOR* mv, unsigned char* mask,
  int nBlkCount, double normFactor, int useY, int scdThreshold, int* scCount)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= nBlkCount) return;

  double v = useY ? double(mv[i].y) : double(mv[i].x);
  double l = v * normFactor * 100 + 128;
  if (l < 0) l = 0;
  if (l > 255) l = 255;
  mask[i] = (unsigned char)l;

  if (mv[i].sad > scdThreshold) {
    atomicAdd(scCount, 1);
  }
}

// kind = 1: the SAD of the block, taken at an intermediate time
__global__ void kl_mask_sad(const VECTOR* mv, unsigned char* mask,
  int nBlkX, int nBlkY, int time4096X, int time4096Y, double factor, double gamma,
  int scdThreshold, int* scCount)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int nBlkCount = nBlkX * nBlkY;
  if (i >= nBlkCount) return;

  int bx = i % nBlkX;
  int by = i / nBlkX;
  int vx = mv[i].x;
  int vy = mv[i].y;
  int bxi = bx - vx * time4096X / 4096;
  int byi = by - vy * time4096Y / 4096;
  if (bxi < 0 || bxi >= nBlkX || byi < 0 || byi >= nBlkY) {
    bxi = bx;
    byi = by;
  }
  int sad = mv[bxi + byi * nBlkX].sad;
  double l = 255.0 * ((gamma == 1.0) ? (sad * factor) : pow(sad * factor, gamma));
  mask[i] = (unsigned char)(l > 255.0 ? 255.0 : l);

  if (mv[i].sad > scdThreshold) {
    atomicAdd(scCount, 1);
  }
}

// Enlarge the block sized mask to the size of the picture.  The weights are
// the ones of the SimpleResize filter of MVTools, the intermediate values
// are truncated the same way so that the result is bit exact.
template <typename pixel_t>
__global__ void kl_mask_scale(const unsigned char* srcMask, pixel_t* dst,
  int dstPitch, int width, int height,
  int nBlkX, int nBlkY, int nWidthB, int nHeightB,
  int bitsPerPixel, int maxPixelValue,
  const int* scCount, int scdLimit, int sceneChangeValue, int forceSceneChange)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  if (x >= width || y >= height) return;

  unsigned char val;
  if (forceSceneChange || (scCount != nullptr && *scCount > scdLimit)) {
    val = (unsigned char)sceneChangeValue;
  }
  else {
    // the area that is not covered by the blocks is filled with the last row/column
    int cx = (x < nWidthB) ? x : nWidthB - 1;
    int cy = (y < nHeightB) ? y : nHeightB - 1;

    // vertical
    int j = cy * 256 * nBlkY / nHeightB + (128 * nBlkY) / nHeightB - 128;
    if (j < 0) j = 0;
    int ky = j >> 8;
    int w2y;
    if (ky > nBlkY - 2) {
      ky = nBlkY - 1;
      w2y = 0;
    }
    else {
      w2y = j - (ky << 8);
    }
    int w1y = 256 - w2y;
    int ky1 = (ky + 1 < nBlkY) ? ky + 1 : nBlkY - 1;

    // horizontal
    int j2 = cx * 256 * nBlkX / nWidthB + (128 * nBlkX) / nWidthB - 128;
    if (j2 < 0) j2 = 0;
    int kx = j2 >> 8;
    int w2x;
    if (kx > nBlkX - 2) {
      kx = nBlkX - 1;
      w2x = 0;
    }
    else {
      w2x = j2 - (kx << 8);
    }
    int w1x = 256 - w2x;
    int kx1 = (kx + 1 < nBlkX) ? kx + 1 : nBlkX - 1;

    int v0 = (srcMask[ky * nBlkX + kx] * w1y + srcMask[ky1 * nBlkX + kx] * w2y + 128) >> 8;
    int v1 = (srcMask[ky * nBlkX + kx1] * w1y + srcMask[ky1 * nBlkX + kx1] * w2y + 128) >> 8;
    val = (unsigned char)((v0 * w1x + v1 * w2x + 128) >> 8);
  }

  if (sizeof(pixel_t) == 1) {
    dst[x + y * dstPitch] = (pixel_t)val;
  }
  else {
    dst[x + y * dstPitch] = (pixel_t)(val == 255 ? maxPixelValue : (val << (bitsPerPixel - 8)));
  }
}

} // namespace

// ---------------------------------------------------------------------------
//  the filter
// ---------------------------------------------------------------------------

class KMotionMask : public GenericVideoFilter
{
private:
  PClip mvClip;
  const KMVParam* params;

  int kind;
  double ml;
  double gamma;
  double fMaskNormFactor;
  double fGamma;
  double fMaskNormFactor2;
  double fSADMaskNormFactor;
  double fHalfGamma;
  double sadFactor;
  int time256;
  int time4096X, time4096Y;

  int nSceneChangeValue;

  int nBlkX, nBlkY, nBlkCount;
  int nPel;
  int nWidthB, nHeightB;
  int nLevelOffset;

  int nSCD1;      // sad threshold, normalized to 8x8 blocks
  int nSCD2Limit; // number of blocks that may exceed nSCD1

  int logUVx, logUVy;

  std::mutex mutex;
  void* scratch = nullptr;
  size_t scratchSize = 0;
  int scratchDev = -1;

  void* GetScratch(size_t bytes, PNeoEnv env)
  {
    std::lock_guard<std::mutex> lock(mutex);
    int dev = env->GetDeviceId();
    if (scratch == nullptr || scratchSize < bytes || scratchDev != dev) {
      if (scratch != nullptr) {
        cudaFree(scratch);
        scratch = nullptr;
        scratchSize = 0;
      }
      CUDA_CHECK(cudaMalloc(&scratch, bytes));
      scratchSize = bytes;
      scratchDev = dev;
    }
    return scratch;
  }

  // The CPU version of the mask.  It produces the same values as the CUDA
  // kernels above, it is used when the clip is not evaluated on the device.
  void MakeSmallMaskCPU(const VECTOR* mv, unsigned char* smallMask, int& scCount) const
  {
    scCount = 0;
    for (int i = 0; i < nBlkCount; ++i) {
      double l;
      if (kind == 3 || kind == 4) {
        double v = (kind == 4) ? double(mv[i].y) : double(mv[i].x);
        l = v * fMaskNormFactor * 100 + 128;
        if (l < 0) l = 0;
        if (l > 255) l = 255;
      }
      else if (kind == 0) {
        int x = mv[i].x;
        int y = mv[i].y;
        double norme = double(x * x + y * y) / (double(nPel) * double(nPel));
        l = 255.0 * ((fHalfGamma == 0.5) ? sqrt(norme * fMaskNormFactor2)
                                        : pow(norme * fMaskNormFactor2, fHalfGamma));
      }
      else {
        int bx = i % nBlkX;
        int by = i / nBlkX;
        int bxi = bx - mv[i].x * time4096X / 4096;
        int byi = by - mv[i].y * time4096Y / 4096;
        if (bxi < 0 || bxi >= nBlkX || byi < 0 || byi >= nBlkY) {
          bxi = bx;
          byi = by;
        }
        int sad = mv[bxi + byi * nBlkX].sad;
        l = 255.0 * ((gamma == 1.0) ? (sad * sadFactor) : pow(sad * sadFactor, gamma));
      }
      smallMask[i] = (unsigned char)(l > 255.0 ? 255.0 : l);
      if (mv[i].sad > nSCD1) {
        scCount++;
      }
    }
  }

  // The value of the mask at the given position.  Same bilinear filter and
  // same rounding as MVTools (SimpleResize).
  unsigned char MaskValueCPU(const unsigned char* smallMask, int x, int y, int widthB, int heightB) const
  {
    int cx = (x < widthB) ? x : widthB - 1;
    int cy = (y < heightB) ? y : heightB - 1;

    int j = cy * 256 * nBlkY / heightB + (128 * nBlkY) / heightB - 128;
    if (j < 0) j = 0;
    int ky = j >> 8;
    int w2y;
    if (ky > nBlkY - 2) { ky = nBlkY - 1; w2y = 0; }
    else { w2y = j - (ky << 8); }
    int w1y = 256 - w2y;
    int ky1 = (ky + 1 < nBlkY) ? ky + 1 : nBlkY - 1;

    int j2 = cx * 256 * nBlkX / widthB + (128 * nBlkX) / widthB - 128;
    if (j2 < 0) j2 = 0;
    int kx = j2 >> 8;
    int w2x;
    if (kx > nBlkX - 2) { kx = nBlkX - 1; w2x = 0; }
    else { w2x = j2 - (kx << 8); }
    int w1x = 256 - w2x;
    int kx1 = (kx + 1 < nBlkX) ? kx + 1 : nBlkX - 1;

    int v0 = (smallMask[ky * nBlkX + kx] * w1y + smallMask[ky1 * nBlkX + kx] * w2y + 128) >> 8;
    int v1 = (smallMask[ky * nBlkX + kx1] * w1y + smallMask[ky1 * nBlkX + kx1] * w2y + 128) >> 8;
    return (unsigned char)((v0 * w1x + v1 * w2x + 128) >> 8);
  }

  template <typename pixel_t>
  void WritePlaneCPU(pixel_t* dst, int pitch, int width, int height,
    const unsigned char* smallMask, int widthB, int heightB, bool sceneChange,
    int sceneChangeValue, int bitsPerPixel, int maxPixelValue) const
  {
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        unsigned char v = sceneChange ? (unsigned char)sceneChangeValue
                                      : MaskValueCPU(smallMask, x, y, widthB, heightB);
        if (sizeof(pixel_t) == 1) {
          dst[x + y * pitch] = (pixel_t)v;
        }
        else {
          dst[x + y * pitch] = (pixel_t)(v == 255 ? maxPixelValue : (v << (bitsPerPixel - 8)));
        }
      }
    }
  }

  bool IsSceneChange(int scCount, bool isValid) const
  {
    return !isValid || (scCount > nSCD2Limit);
  }

  PVideoFrame GetFrameCPU(int n, PNeoEnv env)
  {
    PVideoFrame mvFrame = mvClip->GetFrame(n, env);
    PVideoFrame dst = env->NewVideoFrame(vi);

    const VECTOR* mv = reinterpret_cast<const VECTOR*>(mvFrame->GetReadPtr()) + nLevelOffset;

    std::vector<unsigned char> smallMask(nBlkX * nBlkY);
    int scCount = 0;
    MakeSmallMaskCPU(mv, smallMask.data(), scCount);

    bool isValid = true;
    const AVSMapValue* prop = mvFrame->GetProperty("KMVIsValid");
    if (prop != nullptr) {
      isValid = prop->GetInt() != 0;
    }
    bool sceneChange = IsSceneChange(scCount, isValid);

    int bitsPerPixel = vi.BitsPerComponent();
    int maxPixelValue = (1 << bitsPerPixel) - 1;
    int widthUV = vi.width >> logUVx;
    int heightUV = vi.height >> logUVy;
    int widthBUV = nWidthB >> logUVx;
    int heightBUV = nHeightB >> logUVy;

    if (vi.ComponentSize() == 1) {
      WritePlaneCPU(reinterpret_cast<uint8_t*>(dst->GetWritePtr(PLANAR_Y)), dst->GetPitch(PLANAR_Y),
        vi.width, vi.height, smallMask.data(), nWidthB, nHeightB, sceneChange, nSceneChangeValue,
        bitsPerPixel, maxPixelValue);
      if (!vi.IsY()) {
        WritePlaneCPU(reinterpret_cast<uint8_t*>(dst->GetWritePtr(PLANAR_U)), dst->GetPitch(PLANAR_U),
          widthUV, heightUV, smallMask.data(), widthBUV, heightBUV, sceneChange, nSceneChangeValue,
          bitsPerPixel, maxPixelValue);
        WritePlaneCPU(reinterpret_cast<uint8_t*>(dst->GetWritePtr(PLANAR_V)), dst->GetPitch(PLANAR_V),
          widthUV, heightUV, smallMask.data(), widthBUV, heightBUV, sceneChange, nSceneChangeValue,
          bitsPerPixel, maxPixelValue);
      }
    }
    else {
      WritePlaneCPU(reinterpret_cast<uint16_t*>(dst->GetWritePtr(PLANAR_Y)), dst->GetPitch(PLANAR_Y) / 2,
        vi.width, vi.height, smallMask.data(), nWidthB, nHeightB, sceneChange, nSceneChangeValue,
        bitsPerPixel, maxPixelValue);
      if (!vi.IsY()) {
        WritePlaneCPU(reinterpret_cast<uint16_t*>(dst->GetWritePtr(PLANAR_U)), dst->GetPitch(PLANAR_U) / 2,
          widthUV, heightUV, smallMask.data(), widthBUV, heightBUV, sceneChange, nSceneChangeValue,
          bitsPerPixel, maxPixelValue);
        WritePlaneCPU(reinterpret_cast<uint16_t*>(dst->GetWritePtr(PLANAR_V)), dst->GetPitch(PLANAR_V) / 2,
          widthUV, heightUV, smallMask.data(), widthBUV, heightBUV, sceneChange, nSceneChangeValue,
          bitsPerPixel, maxPixelValue);
      }
    }

    return dst;
  }

public:
  KMotionMask(PClip src, PClip mv, double _ml, double _gamma, int _kind, double _time100,
    int Ysc, int thSCD1, int thSCD2, PNeoEnv env)
    : GenericVideoFilter(src)
    , mvClip(mv)
    , kind(_kind)
    , ml(_ml)
    , gamma(_gamma)
    , time256(0)
    , nSceneChangeValue(0)
  {
    VideoInfo mvVi = mv->GetVideoInfo();
    params = KMVParam::GetParam(mvVi, env);

    if (ml <= 0) {
      env->ThrowError("KMotionMask: ML must be > 0.0");
    }
    if (kind != 0 && kind != 1 && kind != 3 && kind != 4) {
      env->ThrowError("KMotionMask: kind must be 0, 1, 3 or 4");
    }
    if (_time100 < 0 || _time100 > 100) {
      env->ThrowError("KMotionMask: time must be 0.0 to 100.0");
    }

    fMaskNormFactor = 1.0 / ml;
    fMaskNormFactor2 = fMaskNormFactor * fMaskNormFactor;
    fGamma = (_gamma < 0) ? 1.0 : _gamma;
    fHalfGamma = fGamma * 0.5;
    time256 = int(256 * _time100 / 100);

    nSceneChangeValue = (Ysc < 0) ? 0 : ((Ysc > 255) ? 255 : Ysc);

    nBlkX = params->levelInfo[0].nBlkX;
    nBlkY = params->levelInfo[0].nBlkY;
    nBlkCount = nBlkX * nBlkY;
    nPel = params->nPel;
    nLevelOffset = FinestLevelOffset(params);

    nWidthB = nBlkX * (params->nBlkSizeX - params->nOverlapX) + params->nOverlapX;
    nHeightB = nBlkY * (params->nBlkSizeY - params->nOverlapY) + params->nOverlapY;

    logUVx = vi.GetPlaneWidthSubsampling(PLANAR_U);
    logUVy = vi.GetPlaneHeightSubsampling(PLANAR_U);

    if (vi.width < nWidthB) {
      env->ThrowError("KMotionMask: input clip width is too small");
    }
    if (vi.height < nHeightB) {
      env->ThrowError("KMotionMask: input clip height is too small");
    }

    // the thresholds of the scene change detection, same as in MVTools
    nSCD1 = NormalizedSCD1(params, thSCD1);
    nSCD2Limit = thSCD2 * nBlkCount / 256;

    fSADMaskNormFactor = double(nSCD1) / thSCD1 / 3 * 8 * 8 * 2;
    sadFactor = 4.0 * fMaskNormFactor / fSADMaskNormFactor;

    int stepX = params->nBlkSizeX - params->nOverlapX;
    int stepY = params->nBlkSizeY - params->nOverlapY;
    time4096X = (256 - time256) * 16 / (stepX * nPel);
    time4096Y = (256 - time256) * 16 / (stepY * nPel);
  }

  ~KMotionMask()
  {
    if (scratch != nullptr) {
      cudaFree(scratch);
      scratch = nullptr;
    }
  }

  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env_)
  {
    PNeoEnv env = env_;
    if (!IS_CUDA) {
      return GetFrameCPU(n, env);
    }
    cudaStream_t stream = static_cast<cudaStream_t>(env->GetDeviceStream());

    PVideoFrame mvFrame = mvClip->GetFrame(n, env);
    PVideoFrame dst = env->NewVideoFrame(vi);

    // the vectors of the finest level
    const VECTOR* mvSrc = reinterpret_cast<const VECTOR*>(mvFrame->GetReadPtr()) + nLevelOffset;

    // one scratch buffer for the counter, the small mask and the vectors
    size_t offSmall = 16;
    size_t offMv = offSmall + ((nBlkX * nBlkY + 15) & ~size_t(15));
    size_t bytes = offMv + sizeof(VECTOR) * nBlkCount;
    char* buf = static_cast<char*>(GetScratch(bytes, env));
    int* d_sccount = reinterpret_cast<int*>(buf);
    unsigned char* d_small = reinterpret_cast<unsigned char*>(buf + offSmall);
    VECTOR* d_mv = reinterpret_cast<VECTOR*>(buf + offMv);

    // the vectors may live on the device already
    // the vectors may live on the device already
    cudaPointerAttributes attr;
    cudaError_t attrErr = cudaPointerGetAttributes(&attr, mvSrc);
    bool mvOnDevice = false;
#if CUDART_VERSION >= 11020
    mvOnDevice = (attr.type == cudaMemoryTypeDevice);
#else
    mvOnDevice = (attr.memoryType == cudaMemoryTypeDevice);
#endif
    if (attrErr != cudaSuccess) {
      cudaGetLastError(); // the attribute query fails for host memory, ignore it
    }
    cudaMemcpyKind copyKind = mvOnDevice ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice;
    CUDA_CHECK(cudaMemcpy(d_mv, mvSrc, sizeof(VECTOR) * nBlkCount, copyKind));
    CUDA_CHECK(cudaMemsetAsync(d_sccount, 0, sizeof(int), stream));

    int threads = 256;
    int blocks = nblocks(nBlkCount, threads);
    if (kind == 0) {
      kl_mask_length<<<blocks, threads, 0, stream>>>(
        d_mv, d_small, nBlkCount, nPel, fMaskNormFactor2, fHalfGamma, nSCD1, d_sccount);
    }
    else if (kind == 3 || kind == 4) {
      kl_mask_vectorxy<<<blocks, threads, 0, stream>>>(
        d_mv, d_small, nBlkCount, fMaskNormFactor, kind == 4 ? 1 : 0, nSCD1, d_sccount);
    }
    else {
      kl_mask_sad<<<blocks, threads, 0, stream>>>(
        d_mv, d_small, nBlkX, nBlkY, time4096X, time4096Y, sadFactor, gamma, nSCD1, d_sccount);
    }

    // a frame without valid vectors gets the scene change value
    bool isValid = true;
    const AVSMapValue* prop = mvFrame->GetProperty("KMVIsValid");
    if (prop != nullptr) {
      isValid = prop->GetInt() != 0;
    }

    int bitsPerPixel = vi.BitsPerComponent();
    int maxPixelValue = (1 << bitsPerPixel) - 1;

    dim3 threads2(32, 16);
    dim3 blocks2(nblocks(vi.width, threads2.x), nblocks(vi.height, threads2.y));

    if (vi.ComponentSize() == 1) {
      kl_mask_scale<uint8_t><<<blocks2, threads2, 0, stream>>>(
        d_small, reinterpret_cast<uint8_t*>(dst->GetWritePtr(PLANAR_Y)), dst->GetPitch(PLANAR_Y),
        vi.width, vi.height, nBlkX, nBlkY, nWidthB, nHeightB,
        bitsPerPixel, maxPixelValue, d_sccount, nSCD2Limit, nSceneChangeValue, isValid ? 0 : 1);
    }
    else {
      kl_mask_scale<uint16_t><<<blocks2, threads2, 0, stream>>>(
        d_small, reinterpret_cast<uint16_t*>(dst->GetWritePtr(PLANAR_Y)), dst->GetPitch(PLANAR_Y) / 2,
        vi.width, vi.height, nBlkX, nBlkY, nWidthB, nHeightB,
        bitsPerPixel, maxPixelValue, d_sccount, nSCD2Limit, nSceneChangeValue, isValid ? 0 : 1);
    }

    if (!vi.IsY()) {
      int widthUV = vi.width >> logUVx;
      int heightUV = vi.height >> logUVy;
      int widthBUV = nWidthB >> logUVx;
      int heightBUV = nHeightB >> logUVy;
      dim3 blocksUV(nblocks(widthUV, threads2.x), nblocks(heightUV, threads2.y));

      // NOTE: PLANAR_Y/U/V are bit masks, not plane numbers, so the planes
      // have to be handled one by one.  Iterating from PLANAR_U to PLANAR_V
      // would use the invalid plane index 3, which returns the luma plane.
      if (vi.ComponentSize() == 1) {
        kl_mask_scale<uint8_t><<<blocksUV, threads2, 0, stream>>>(
          d_small, reinterpret_cast<uint8_t*>(dst->GetWritePtr(PLANAR_U)), dst->GetPitch(PLANAR_U),
          widthUV, heightUV, nBlkX, nBlkY, widthBUV, heightBUV,
          bitsPerPixel, maxPixelValue, d_sccount, nSCD2Limit, nSceneChangeValue, isValid ? 0 : 1);
        kl_mask_scale<uint8_t><<<blocksUV, threads2, 0, stream>>>(
          d_small, reinterpret_cast<uint8_t*>(dst->GetWritePtr(PLANAR_V)), dst->GetPitch(PLANAR_V),
          widthUV, heightUV, nBlkX, nBlkY, widthBUV, heightBUV,
          bitsPerPixel, maxPixelValue, d_sccount, nSCD2Limit, nSceneChangeValue, isValid ? 0 : 1);
      }
      else {
        kl_mask_scale<uint16_t><<<blocksUV, threads2, 0, stream>>>(
          d_small, reinterpret_cast<uint16_t*>(dst->GetWritePtr(PLANAR_U)), dst->GetPitch(PLANAR_U) / 2,
          widthUV, heightUV, nBlkX, nBlkY, widthBUV, heightBUV,
          bitsPerPixel, maxPixelValue, d_sccount, nSCD2Limit, nSceneChangeValue, isValid ? 0 : 1);
        kl_mask_scale<uint16_t><<<blocksUV, threads2, 0, stream>>>(
          d_small, reinterpret_cast<uint16_t*>(dst->GetWritePtr(PLANAR_V)), dst->GetPitch(PLANAR_V) / 2,
          widthUV, heightUV, nBlkX, nBlkY, widthBUV, heightBUV,
          bitsPerPixel, maxPixelValue, d_sccount, nSCD2Limit, nSceneChangeValue, isValid ? 0 : 1);
      }
    }

    DEBUG_SYNC;

    return dst;
  }

  int __stdcall SetCacheHints(int cachehints, int frame_range)
  {
    if (cachehints == CACHE_GET_DEV_TYPE) {
      return DEV_TYPE_CPU | DEV_TYPE_CUDA;
    }
    else if (cachehints == CACHE_GET_MTMODE) {
      return MT_NICE_FILTER;
    }
    return 0;
  }

  static AVSValue __cdecl Create(AVSValue args, void* user_data, IScriptEnvironment* env)
  {
    return new KMotionMask(
      args[0].AsClip(),  // source, only used for the format of the mask
      args[1].AsClip(),  // motion vectors
      args[2].AsFloat(100), // ml
      args[3].AsFloat(1),   // gamma
      args[4].AsInt(0),     // kind
      args[5].AsFloat(100), // time in percent
      args[6].AsInt(0),     // Ysc
      args[7].AsInt(400),   // thSCD1
      args[8].AsInt(130),   // thSCD2
      env);
  }
};

void AddFuncMotionMask(IScriptEnvironment* env)
{
  env->AddFunction("KMotionMask",
    "cc[ml]f[gamma]f[kind]i[time]f[Ysc]i[thSCD1]i[thSCD2]i",
    KMotionMask::Create, 0);
}
