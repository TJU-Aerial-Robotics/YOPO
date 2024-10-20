/***********************************************************************
  Copyright (C) 2020 Hironori Fujimoto

  This program is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.
  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
***********************************************************************/

#ifndef SGM_GPU__UTIL_H_
#define SGM_GPU__UTIL_H_

#include <iostream>
#include <dirent.h>
#include <stdio.h>

#define FERMI false

#define GPU_THREADS_PER_BLOCK_FERMI 256
#define GPU_THREADS_PER_BLOCK_MAXWELL 64

/* Defines related to GPU Architecture */
#if FERMI
  #define GPU_THREADS_PER_BLOCK   GPU_THREADS_PER_BLOCK_FERMI
#else
  #define GPU_THREADS_PER_BLOCK   GPU_THREADS_PER_BLOCK_MAXWELL
#endif

#define WARP_SIZE		32

namespace sgm_gpu
{

static void CheckCudaErrorAux (const char *, unsigned, const char *, cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)

/**
* Check the return value of the CUDA runtime API call and exit
* the application if the call has failed.
*/
static void CheckCudaErrorAux (const char *file, unsigned line, const char *statement, cudaError_t err) {
  if (err == cudaSuccess)
    return;
  std::cerr << statement<<" returned " << cudaGetErrorString(err) << "("<<err<< ") at "<<file<<":"<<line << std::endl;
  exit (1);
}

/*************************************
GPU Side defines (ASM instructions)
**************************************/

// output temporal carry in internal register
#define UADD__CARRY_OUT(c, a, b) \
  asm volatile("add.cc.u32 %0, %1, %2;" : "=r"(c) : "r"(a) , "r"(b))

// add & output with temporal carry of internal register
#define UADD__IN_CARRY_OUT(c, a, b) \
  asm volatile("addc.cc.u32 %0, %1, %2;" : "=r"(c) : "r"(a) , "r"(b))

// add with temporal carry of internal register
#define UADD__IN_CARRY(c, a, b) \
  asm volatile("addc.u32 %0, %1, %2;" : "=r"(c) : "r"(a) , "r"(b))

// packing and unpacking: from uint64_t to uint2
#define V2S_B64(v,s) \
  asm("mov.b64 %0, {%1,%2};" : "=l"(s) : "r"(v.x), "r"(v.y))

// packing and unpacking: from uint2 to uint64_t
#define S2V_B64(s,v) \
  asm("mov.b64 {%0,%1}, %2;" : "=r"(v.x), "=r"(v.y) : "l"(s))


/*************************************
DEVICE side basic block primitives
**************************************/

#if FERMI
  #define LDG(ptr)  (* ptr)
#else
  #define LDG(ptr)  __ldg(ptr)
#endif

#if FERMI
__shared__ int interBuff[GPU_THREADS_PER_BLOCK];
__inline__ __device__ int __emulated_shfl(const int scalarValue, const uint32_t source_lane) {
  const int warpIdx = threadIdx.x / WARP_SIZE;
  const int laneIdx = threadIdx.x % WARP_SIZE;
  volatile int *interShuffle = interBuff + (warpIdx * WARP_SIZE);
  interShuffle[laneIdx] = scalarValue;
  return(interShuffle[source_lane % WARP_SIZE]);
}
#endif

__inline__ __device__ int shfl_32(int scalarValue, const int lane) {
  #if FERMI
    return __emulated_shfl(scalarValue, (uint32_t)lane);
  #else
    return __shfl_sync(0xffffffff, scalarValue, lane);
  #endif
}

__inline__ __device__ int shfl_up_32(int scalarValue, const int n) {
  #if FERMI
    int lane = threadIdx.x % WARP_SIZE;
    lane -= n;
    return shfl_32(scalarValue, lane);
  #else
    return __shfl_up_sync(0xffffffff, scalarValue, n);
  #endif
}

__inline__ __device__ int shfl_down_32(int scalarValue, const int n) {
  #if FERMI
    int lane = threadIdx.x % WARP_SIZE;
    lane += n;
    return shfl_32(scalarValue, lane);
  #else
    return __shfl_down_sync(0xffffffff, scalarValue, n);
  #endif
}

__inline__ __device__ int shfl_xor_32(int scalarValue, const int n) {
  #if FERMI
    int lane = threadIdx.x % WARP_SIZE;
    lane = lane ^ n;
    return shfl_32(scalarValue, lane);
  #else
    return __shfl_xor_sync(0xffffffff, scalarValue, n);
  #endif
}

__device__ __forceinline__ uint32_t ld_gbl_ca(const __restrict__ uint32_t *addr) {
  uint32_t return_value;
  asm("ld.global.ca.u32 %0, [%1];" : "=r"(return_value) : "l"(addr));
  return return_value;
}

__device__ __forceinline__ uint32_t ld_gbl_cs(const __restrict__ uint32_t *addr) {
  uint32_t return_value;
  asm("ld.global.cs.u32 %0, [%1];" : "=r"(return_value) : "l"(addr));
  return return_value;
}

__device__ __forceinline__ void st_gbl_wt(const __restrict__ uint32_t *addr, const uint32_t value) {
  asm("st.global.wt.u32 [%0], %1;" :: "l"(addr), "r"(value));
}

__device__ __forceinline__ void st_gbl_cs(const __restrict__ uint32_t *addr, const uint32_t value) {
  asm("st.global.cs.u32 [%0], %1;" :: "l"(addr), "r"(value));
}

__device__ __forceinline__ uint32_t gpu_get_sm_idx(){
  uint32_t smid;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
  return(smid);
}

__device__ __forceinline__ void uint32_to_uchars(const uint32_t s, int *u1, int *u2, int *u3, int *u4) {
  //*u1 = s & 0xff;
  *u1 = __byte_perm(s, 0, 0x4440);
  //*u2 = (s>>8) & 0xff;
  *u2 = __byte_perm(s, 0, 0x4441);
  //*u3 = (s>>16) & 0xff;
  *u3 = __byte_perm(s, 0, 0x4442);
  //*u4 = s>>24;
  *u4 = __byte_perm(s, 0, 0x4443);
}

__device__ __forceinline__ uint32_t uchars_to_uint32(int u1, int u2, int u3, int u4) {
  //return u1 | (u2<<8) | (u3<<16) | (u4<<24);
  //return __byte_perm(u1, u2, 0x7740) + __byte_perm(u3, u4, 0x4077);
  return u1 | (u2<<8) | __byte_perm(u3, u4, 0x4077);
}

__device__ __forceinline__ uint32_t uchar_to_uint32(int u1) {
  return __byte_perm(u1, u1, 0x0);
}

__device__ __forceinline__ unsigned int vcmpgeu4(unsigned int a, unsigned int b) {
    unsigned int r, c;
    c = a-b;
    asm ("prmt.b32 %0,%1,0,0xba98;" : "=r"(r) : "r"(c));// build mask from msbs
    return r;           // byte-wise unsigned gt-eq comparison with mask result
}

__device__ __forceinline__ unsigned int vminu4(unsigned int a, unsigned int b) {
    unsigned int r, s;
    s = vcmpgeu4 (b, a);// mask = 0xff if a >= b
    r = a & s;          // select a when b >= a
    s = b & ~s;         // select b when b < a
    r = r | s;          // combine byte selections
    return r;
}

__device__ __forceinline__ void print_uchars(const char* str, const uint32_t s) {
  int u1, u2, u3, u4;
  uint32_to_uchars(s, &u1, &u2, &u3, &u4);
  printf("%s: %d %d %d %d\n", str, u1, u2, u3, u4);
}

template<class T>
__device__ __forceinline__ int popcount(T n) {
#if CSCT or CSCT_RECOMPUTE
  return __popc(n);
#else
  return __popcll(n);
#endif
}

__inline__ __device__ uint8_t minu8_index4(int *min_idx, const uint8_t val1, const int dis, const uint8_t val2, const int dis2, const uint8_t val3, const int dis3, const uint8_t val4, const int dis4) {
  int min_idx1 = dis;
  uint8_t min1 = val1;
  if(val1 > val2) {
    min1 = val2;
    min_idx1 = dis2;
  }

  int min_idx2 = dis3;
  uint8_t min2 = val3;
  if(val3 > val4) {
    min2 = val4;
    min_idx2 = dis4;
  }

  uint8_t minval = min1;
  *min_idx = min_idx1;
  if(min1 > min2) {
    minval = min2;
    *min_idx = min_idx2;
  }
  return minval;
}

__inline__ __device__ uint8_t minu8_index8(int *min_idx, const uint8_t val1, const int dis, const uint8_t val2, const int dis2, const uint8_t val3, const int dis3, const uint8_t val4, const int dis4, const uint8_t val5, const int dis5, const uint8_t val6, const int dis6, const uint8_t val7, const int dis7, const uint8_t val8, const int dis8) {
  int min_idx1, min_idx2;
  uint8_t minval1, minval2;

  minval1 = minu8_index4(&min_idx1, val1, dis, val2, dis2, val3, dis3, val4, dis4);
  minval2 = minu8_index4(&min_idx2, val5, dis5, val6, dis6, val7, dis7, val8, dis8);

  *min_idx = min_idx1;
  uint8_t minval = minval1;
  if(minval1 > minval2) {
    *min_idx = min_idx2;
    minval = minval2;
  }
  return minval;
}

__inline__ __device__ int warpReduceMinIndex2(int *val, int idx) {
  for(int d = 1; d < WARP_SIZE; d *= 2) {
    int tmp = shfl_xor_32(*val, d);
    int tmp_idx = shfl_xor_32(idx, d);
    if(*val > tmp) {
      *val = tmp;
      idx = tmp_idx;
    }
  }
  return idx;
}

__inline__ __device__ int warpReduceMinIndex(int val, int idx) {
  for(int d = 1; d < WARP_SIZE; d *= 2) {
    int tmp = shfl_xor_32(val, d);
    int tmp_idx = shfl_xor_32(idx, d);
    if(val > tmp) {
      val = tmp;
      idx = tmp_idx;
    }
  }
  return idx;
}

__inline__ __device__ int warpReduceMin(int val) {
  val = min(val, shfl_xor_32(val, 1));
  val = min(val, shfl_xor_32(val, 2));
  val = min(val, shfl_xor_32(val, 4));
  val = min(val, shfl_xor_32(val, 8));
  val = min(val, shfl_xor_32(val, 16));
  return val;
}

__inline__ __device__ int blockReduceMin(int val) {
  static __shared__ int shared[WARP_SIZE]; // Shared mem for WARP_SIZE partial sums
  const int lane = threadIdx.x % WARP_SIZE;
  const int wid = threadIdx.x / WARP_SIZE;

  val = warpReduceMin(val);     // Each warp performs partial reduction

  if (lane==0) shared[wid]=val; // Write reduced value to shared memory

  __syncthreads();              // Wait for all partial reductions

  //read from shared memory only if that warp existed
  val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : INT_MAX;

  if (wid==0) val = warpReduceMin(val); //Final reduce within first warp

  return val;
}

__inline__ __device__ int blockReduceMinIndex(int val, int idx) {
  static __shared__ int shared_val[WARP_SIZE]; // Shared mem for WARP_SIZE partial mins
  static __shared__ int shared_idx[WARP_SIZE]; // Shared mem for WARP_SIZE indexes
  const int lane = threadIdx.x % WARP_SIZE;
  const int wid = threadIdx.x / WARP_SIZE;

  idx = warpReduceMinIndex2(&val, idx);     // Each warp performs partial reduction

  if (lane==0) {
    shared_val[wid]=val;
    shared_idx[wid]=idx;
  }

  __syncthreads();              // Wait for all partial reductions

  //read from shared memory only if that warp existed
  val = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared_val[lane] : INT_MAX;
  idx = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared_idx[lane] : INT_MAX;

  if (wid==0) {
    idx = warpReduceMinIndex2(&val, idx); //Final reduce within first warp
  }

  return idx;
}


__inline__ __device__ bool blockAny(bool local_condition) {
  __shared__ bool conditions[WARP_SIZE];
  const int lane = threadIdx.x % WARP_SIZE;
  const int wid = threadIdx.x / WARP_SIZE;

  local_condition = __any_sync(0xffffffff, local_condition);     // Each warp performs __any

  if (lane==0) {
    conditions[wid]=local_condition;
  }

  __syncthreads();              // Wait for all partial __any

  //read from shared memory only if that warp existed
  local_condition = (threadIdx.x < blockDim.x / WARP_SIZE) ? conditions[lane] : false;

  if (wid==0) {
    local_condition = __any_sync(0xffffffff, local_condition); //Final __any within first warp
  }

  return local_condition;
}

} //namespace sgm_gpu

#endif // SGM_GPU__UTIL_H_

