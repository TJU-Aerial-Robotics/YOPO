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

#ifndef SGM_GPU__MEDIAN_FILTER_H_
#define SGM_GPU__MEDIAN_FILTER_H_

#include <stdint.h>

namespace sgm_gpu
{

__global__ void MedianFilter3x3(const uint8_t* __restrict__ d_input, uint8_t* __restrict__ d_out, const uint32_t rows, const uint32_t cols);

template<int n, typename T>
__inline__ __device__ void MedianFilter(const T* __restrict__ d_input, T* __restrict__ d_out, const uint32_t rows, const uint32_t cols) {
  const uint32_t idx = blockIdx.x*blockDim.x+threadIdx.x;
  const uint32_t row = idx / cols;
  const uint32_t col = idx % cols;
  T window[n*n];
  int half = n/2;

  if(row >= half && col >= half && row < rows-half && col < cols-half) {
    for(uint32_t i = 0; i < n; i++) {
      for(uint32_t j = 0; j < n; j++) {
        window[i*n+j] = d_input[(row-half+i)*cols+col-half+j];
      }
    }

    for(uint32_t i = 0; i < (n*n/2)+1; i++) {
      uint32_t min_idx = i;
      for(uint32_t j = i+1; j < n*n; j++) {
        if(window[j] < window[min_idx]) {
          min_idx = j;
        }
      }
      const T tmp = window[i];
      window[i] = window[min_idx];
      window[min_idx] = tmp;
    }
    d_out[idx] = window[n*n/2];
  } else if(row < rows && col < cols) {
    d_out[idx] = d_input[idx];
  }
}

} // namespace sgm_gpu

#endif // SGM_GPU__MEDIAN_FILTER_H_

