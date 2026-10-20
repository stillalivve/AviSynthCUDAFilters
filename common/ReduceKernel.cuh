#pragma once

#include <cuda_runtime_api.h>
#include <cuda_device_runtime_api.h>

template <typename T>
struct AddReducer {
	__device__ void operator()(T& v, T o) { v += o; }
};

template <typename T>
struct MaxReducer {
  __device__ void operator()(T& v, T o) { v = max(v, o); }
};

template <typename T>
struct MaxIndexReducer {
	__device__ void operator()(T& cnt, int& idx, T ocnt, int oidx) {
    if (ocnt > cnt) {
      cnt = ocnt;
      idx = oidx;
    }
  }
};

// MAX��<=32����2�ׂ��̂ݑΉ�
// The lanes that take part in a warp shuffle are the MAX lanes of the group
// the thread belongs to.  Naming all 32 lanes in the mask is wrong as soon as
// only some of them reach the shuffle: on Volta and later the hardware waits
// for every thread named in the mask, so the kernel deadlocks when a thread
// that is named never arrives.  That is what happens in the motion search,
// where the SAD reduction runs for 7 of the 8 thread groups only.
__device__ __forceinline__ unsigned int dev_warp_reduce_mask(int max_lanes) {
  const int width = (max_lanes < 32) ? max_lanes : 32;
  const int lane = threadIdx.x & 31;
  const int group = (lane / width) * width;
  // 1u << 32 is undefined behaviour, so the full width has to be special cased
  const unsigned int lanes = (width >= 32) ? 0xffffffffu : ((1u << width) - 1u);
  return lanes << group;
}
template <typename T, int MAX, typename REDUCER>
__device__ void dev_reduce_warp(int tid, T& value)
{
  REDUCER red;
  const unsigned int mask = dev_warp_reduce_mask(MAX);
  // warp shuffle��reduce
#if CUDART_VERSION >= 9000
  if (MAX >= 32) red(value, __shfl_down_sync(mask, value, 16));
  if (MAX >= 16) red(value, __shfl_down_sync(mask, value, 8));
  if (MAX >= 8) red(value, __shfl_down_sync(mask, value, 4));
  if (MAX >= 4) red(value, __shfl_down_sync(mask, value, 2));
  if (MAX >= 2) red(value, __shfl_down_sync(mask, value, 1));
#else
  if (MAX >= 32) red(value, __shfl_down(value, 16));
  if (MAX >= 16) red(value, __shfl_down(value, 8));
  if (MAX >= 8) red(value, __shfl_down(value, 4));
  if (MAX >= 4) red(value, __shfl_down(value, 2));
  if (MAX >= 2) red(value, __shfl_down(value, 1));
#endif
}

// MAX��2�ׂ��̂ݑΉ�
// buf��shared memory����
template <typename T, int MAX, typename REDUCER>
__device__ void dev_reduce(int tid, T& value, T* buf)
{
  REDUCER red;
  if (MAX >= 64) {
    buf[tid] = value;
    __syncthreads();
    if (MAX >= 1024) {
      if (tid < 512) {
        red(buf[tid], buf[tid + 512]);
      }
      __syncthreads();
    }
    if (MAX >= 512) {
      if (tid < 256) {
        red(buf[tid], buf[tid + 256]);
      }
      __syncthreads();
    }
    if (MAX >= 256) {
      if (tid < 128) {
        red(buf[tid], buf[tid + 128]);
      }
      __syncthreads();
    }
    if (MAX >= 128) {
      if (tid < 64) {
        red(buf[tid], buf[tid + 64]);
      }
      __syncthreads();
    }
    if (MAX >= 64) {
      if (tid < 32) {
        red(buf[tid], buf[tid + 32]);
      }
      __syncthreads();
    }
    value = buf[tid];
  }
  if (tid < 32) {
    dev_reduce_warp<T, MAX, REDUCER>(tid, value);
  }
}

// MAX��<=32����2�ׂ��̂ݑΉ�
template <typename T, int N, int MAX, typename REDUCER>
__device__ void dev_reduceN_warp(int tid, T value[N])
{
  REDUCER red;
  const unsigned int mask = dev_warp_reduce_mask(MAX);
  // warp shuffle��reduce
#if CUDART_VERSION >= 9000
  if (MAX >= 32) for(int i = 0; i < N; ++i) red(value[i], __shfl_down_sync(mask, value[i], 16));
  if (MAX >= 16) for (int i = 0; i < N; ++i) red(value[i], __shfl_down_sync(mask, value[i], 8));
  if (MAX >= 8) for (int i = 0; i < N; ++i) red(value[i], __shfl_down_sync(mask, value[i], 4));
  if (MAX >= 4) for (int i = 0; i < N; ++i) red(value[i], __shfl_down_sync(mask, value[i], 2));
  if (MAX >= 2) for (int i = 0; i < N; ++i) red(value[i], __shfl_down_sync(mask, value[i], 1));
#else
  if (MAX >= 32) for (int i = 0; i < N; ++i) red(value[i], __shfl_down(value[i], 16));
  if (MAX >= 16) for (int i = 0; i < N; ++i) red(value[i], __shfl_down(value[i], 8));
  if (MAX >= 8) for (int i = 0; i < N; ++i) red(value[i], __shfl_down(value[i], 4));
  if (MAX >= 4) for (int i = 0; i < N; ++i) red(value[i], __shfl_down(value[i], 2));
  if (MAX >= 2) for (int i = 0; i < N; ++i) red(value[i], __shfl_down(value[i], 1));
#endif
}

// MAX��2�ׂ��̂ݑΉ�
// buf��shared memory����
template <typename T, int N, int MAX, typename REDUCER>
__device__ void dev_reduceN(int tid, T value[N], T* buf)
{
  REDUCER red;
  if (MAX >= 64) {
    for (int i = 0; i < N; ++i) buf[i * MAX + tid] = value[i];
    __syncthreads();
    if (MAX >= 1024) {
      if (tid < 512) {
        for (int i = 0; i < N; ++i) red(buf[i * MAX + tid], buf[i * MAX + tid + 512]);
      }
      __syncthreads();
    }
    if (MAX >= 512) {
      if (tid < 256) {
        for (int i = 0; i < N; ++i) red(buf[i * MAX + tid], buf[i * MAX + tid + 256]);
      }
      __syncthreads();
    }
    if (MAX >= 256) {
      if (tid < 128) {
        for (int i = 0; i < N; ++i) red(buf[i * MAX + tid], buf[i * MAX + tid + 128]);
      }
      __syncthreads();
    }
    if (MAX >= 128) {
      if (tid < 64) {
        for (int i = 0; i < N; ++i) red(buf[i * MAX + tid], buf[i * MAX + tid + 64]);
      }
      __syncthreads();
    }
    if (MAX >= 64) {
      if (tid < 32) {
        for (int i = 0; i < N; ++i) red(buf[i * MAX + tid], buf[i * MAX + tid + 32]);
      }
      __syncthreads();
    }
    for (int i = 0; i < N; ++i) value[i] = buf[i * MAX + tid];
  }
  if (tid < 32) {
    dev_reduceN_warp<T, N, MAX, REDUCER>(tid, value);
  }
}

// MAX��<=32����2�ׂ��̂ݑΉ�
template <typename K, typename V, int MAX, typename REDUCER>
__device__ void dev_reduce2_warp(int tid, K& key, V& value)
{
  REDUCER red;
  if (MAX >= 32) {
#if CUDART_VERSION >= 9000
    K okey = __shfl_down_sync(0xffffffff, key, 16);
    V ovalue = __shfl_down_sync(0xffffffff, value, 16);
#else
    K okey = __shfl_down(key, 16);
    V ovalue = __shfl_down(value, 16);
#endif
    red(key, value, okey, ovalue);
  }
  if (MAX >= 16) {
#if CUDART_VERSION >= 9000
    K okey = __shfl_down_sync(0xffffffff, key, 8);
    V ovalue = __shfl_down_sync(0xffffffff, value, 8);
#else
    K okey = __shfl_down(key, 8);
    V ovalue = __shfl_down(value, 8);
#endif
    red(key, value, okey, ovalue);
  }
  if (MAX >= 8) {
#if CUDART_VERSION >= 9000
    K okey = __shfl_down_sync(0xffffffff, key, 4);
    V ovalue = __shfl_down_sync(0xffffffff, value, 4);
#else
    K okey = __shfl_down(key, 4);
    V ovalue = __shfl_down(value, 4);
#endif
    red(key, value, okey, ovalue);
  }
  if (MAX >= 4) {
#if CUDART_VERSION >= 9000
    K okey = __shfl_down_sync(0xffffffff, key, 2);
    V ovalue = __shfl_down_sync(0xffffffff, value, 2);
#else
    K okey = __shfl_down(key, 2);
    V ovalue = __shfl_down(value, 2);
#endif
    red(key, value, okey, ovalue);
  }
  if (MAX >= 2) {
#if CUDART_VERSION >= 9000
    K okey = __shfl_down_sync(0xffffffff, key, 1);
    V ovalue = __shfl_down_sync(0xffffffff, value, 1);
#else
    K okey = __shfl_down(key, 1);
    V ovalue = __shfl_down(value, 1);
#endif
    red(key, value, okey, ovalue);
  }
}

// MAX��2�ׂ��̂ݑΉ�
template <typename K, typename V, int MAX, typename REDUCER>
__device__ void dev_reduce2(int tid, K& key, V& value, K* kbuf, V* vbuf)
{
  REDUCER red;
  if (MAX >= 64) {
    kbuf[tid] = key;
    vbuf[tid] = value;
    __syncthreads();
    if (MAX >= 1024) {
      if (tid < 512) {
        red(kbuf[tid], vbuf[tid], kbuf[tid + 512], vbuf[tid + 512]);
      }
      __syncthreads();
    }
    if (MAX >= 512) {
      if (tid < 256) {
        red(kbuf[tid], vbuf[tid], kbuf[tid + 256], vbuf[tid + 256]);
      }
      __syncthreads();
    }
    if (MAX >= 256) {
      if (tid < 128) {
        red(kbuf[tid], vbuf[tid], kbuf[tid + 128], vbuf[tid + 128]);
      }
      __syncthreads();
    }
    if (MAX >= 128) {
      if (tid < 64) {
        red(kbuf[tid], vbuf[tid], kbuf[tid + 64], vbuf[tid + 64]);
      }
      __syncthreads();
    }
    if (MAX >= 64) {
      if (tid < 32) {
        red(kbuf[tid], vbuf[tid], kbuf[tid + 32], vbuf[tid + 32]);
      }
      __syncthreads();
    }
    key = kbuf[tid];
    value = vbuf[tid];
  }
  if (tid < 32) {
    dev_reduce2_warp<K, V, MAX, REDUCER>(tid, key, value);
  }
}

// MAX��<=32����2�ׂ��̂ݑΉ�
template <typename T, int MAX, typename REDUCER>
__device__ void dev_scan_warp(int tid, T& value)
{
  REDUCER red;
  // warp shuffle��scan
  if (MAX >= 2) {
#if CUDART_VERSION >= 9000
    T tmp = __shfl_up_sync(0xffffffff, value, 1);
#else
    T tmp = __shfl_up(value, 1);
#endif
    if (tid >= 1) red(value, tmp);
  }
  if (MAX >= 4) {
#if CUDART_VERSION >= 9000
    T tmp = __shfl_up_sync(0xffffffff, value, 2);
#else
    T tmp = __shfl_up(value, 2);
#endif
    if (tid >= 2) red(value, tmp);
  }
  if (MAX >= 8) {
#if CUDART_VERSION >= 9000
    T tmp = __shfl_up_sync(0xffffffff, value, 4);
#else
    T tmp = __shfl_up(value, 4);
#endif
    if (tid >= 4) red(value, tmp);
  }
  if (MAX >= 16) {
#if CUDART_VERSION >= 9000
    T tmp = __shfl_up_sync(0xffffffff, value, 8);
#else
    T tmp = __shfl_up(value, 8);
#endif
    if (tid >= 8) red(value, tmp);
  }
  if (MAX >= 32) {
#if CUDART_VERSION >= 9000
    T tmp = __shfl_up_sync(0xffffffff, value, 16);
#else
    T tmp = __shfl_up(value, 16);
#endif
    if (tid >= 16) red(value, tmp);
  }
}

// MAX��2�ׂ��̂ݑΉ�
// buf��shared memory���� ����: MAX/32
template <typename T, int MAX, typename REDUCER>
__device__ void dev_scan(int tid, T& value, T* buf)
{
  REDUCER red;
  int wid = tid & 31;
  // �܂�warp����scan
  dev_scan_warp<T, MAX, REDUCER>(wid, value);
  if (MAX >= 64) {
    // warp���Ƃ̌��ʂ�shared����������ďW��
    if (wid == 31) buf[tid >> 5] = value;
    __syncthreads();
    if (tid < MAX / 32) {
      // warp���Ƃ̌��ʂ�warp���ł����scan
      T v2 = buf[tid];
      dev_scan_warp<T, MAX / 32, REDUCER>(wid, v2);
      // shared����������ĕ��z
      buf[tid] = v2;
    }
    __syncthreads();
    // warp���Ƃ�scan���ʂ𑫂�
    if(tid >= 32) red(value, buf[(tid >> 5) - 1]);
    __syncthreads();
  }
}

