#pragma once
#include "avisynth.h"

#define NOMINMAX
#include <windows.h>

#include "CommonFunctions.h"

// CUDAƒJ[ƒlƒ‹ŽÀ‘•‚Ì‹¤’Êˆ—
class CudaKernelBase
{
protected:
  PNeoEnv env;
  cudaStream_t stream;
public:

  void SetEnv(PNeoEnv env)
  {
    this->env = env;
    stream = static_cast<cudaStream_t>(env->GetDeviceStream());
  }

  void VerifyCUDAPointer(void* ptr)
  {
#ifndef NDEBUG
    cudaPointerAttributes attr;
    CUDA_CHECK(cudaPointerGetAttributes(&attr, ptr));
#if CUDART_VERSION >= 11020
    // cudaPointerAttributes::memoryType was deprecated in CUDA 11.2 and removed in 12.0
    if (attr.type != cudaMemoryTypeDevice) {
#else
    if (attr.memoryType != cudaMemoryTypeDevice) {
#endif
      env->ThrowError("[CUDA Error] Not valid devicce pointer");
    }
#endif
  }
};

