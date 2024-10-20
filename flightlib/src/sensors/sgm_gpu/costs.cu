/***********************************************************************
  Copyright (C) 2019 Hironori Fujimoto

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

#include "flightlib/sensors/sgm_gpu/costs.h"
#include <stdio.h>

namespace sgm_gpu
{

__global__ void 
__launch_bounds__(1024, 2)
CenterSymmetricCensusKernelSM2(const uint8_t *im, const uint8_t *im2, cost_t *transform, cost_t *transform2, const uint32_t rows, const uint32_t cols) {
  const int idx = blockIdx.x*blockDim.x+threadIdx.x;
  const int idy = blockIdx.y*blockDim.y+threadIdx.y;

  const int win_cols = (32+LEFT*2); // 32+4*2 = 40
  const int win_rows = (32+TOP*2); // 32+3*2 = 38

  __shared__ uint8_t window[win_cols*win_rows];
  __shared__ uint8_t window2[win_cols*win_rows];

  const int id = threadIdx.y*blockDim.x+threadIdx.x;
  const int sm_row = id / win_cols;
  const int sm_col = id % win_cols;

  const int im_row = blockIdx.y*blockDim.y+sm_row-TOP;
  const int im_col = blockIdx.x*blockDim.x+sm_col-LEFT;
  const bool boundaries = (im_row >= 0 && im_col >= 0 && im_row < rows && im_col < cols);
  window[sm_row*win_cols+sm_col] = boundaries ? im[im_row*cols+im_col] : 0;
  window2[sm_row*win_cols+sm_col] = boundaries ? im2[im_row*cols+im_col] : 0;

  // Not enough threads to fill window and window2
  const int block_size = blockDim.x*blockDim.y;
  if(id < (win_cols*win_rows-block_size)) {
    const int id = threadIdx.y*blockDim.x+threadIdx.x+block_size;
    const int sm_row = id / win_cols;
    const int sm_col = id % win_cols;

    const int im_row = blockIdx.y*blockDim.y+sm_row-TOP;
    const int im_col = blockIdx.x*blockDim.x+sm_col-LEFT;
    const bool boundaries = (im_row >= 0 && im_col >= 0 && im_row < rows && im_col < cols);
    window[sm_row*win_cols+sm_col] = boundaries ? im[im_row*cols+im_col] : 0;
    window2[sm_row*win_cols+sm_col] = boundaries ? im2[im_row*cols+im_col] : 0;
  }

  __syncthreads();
  cost_t census = 0;
  cost_t census2 = 0;
  if(idy < rows && idx < cols) {
      for(int k = 0; k < CENSUS_HEIGHT/2; k++) {
        for(int m = 0; m < CENSUS_WIDTH; m++) {
          const uint8_t e1 = window[(threadIdx.y+k)*win_cols+threadIdx.x+m];
          const uint8_t e2 = window[(threadIdx.y+2*TOP-k)*win_cols+threadIdx.x+2*LEFT-m];
          const uint8_t i1 = window2[(threadIdx.y+k)*win_cols+threadIdx.x+m];
          const uint8_t i2 = window2[(threadIdx.y+2*TOP-k)*win_cols+threadIdx.x+2*LEFT-m];

          const int shft = k*CENSUS_WIDTH+m;
          // Compare to the center
          cost_t tmp = (e1 >= e2);
          // Shift to the desired position
          tmp <<= shft;
          // Add it to its place
          census |= tmp;
          // Compare to the center
          cost_t tmp2 = (i1 >= i2);
          // Shift to the desired position
          tmp2 <<= shft;
          // Add it to its place
          census2 |= tmp2;
        }
      }
      if(CENSUS_HEIGHT % 2 != 0) {
        const int k = CENSUS_HEIGHT/2;
        for(int m = 0; m < CENSUS_WIDTH/2; m++) {
          const uint8_t e1 = window[(threadIdx.y+k)*win_cols+threadIdx.x+m];
          const uint8_t e2 = window[(threadIdx.y+2*TOP-k)*win_cols+threadIdx.x+2*LEFT-m];
          const uint8_t i1 = window2[(threadIdx.y+k)*win_cols+threadIdx.x+m];
          const uint8_t i2 = window2[(threadIdx.y+2*TOP-k)*win_cols+threadIdx.x+2*LEFT-m];

          const int shft = k*CENSUS_WIDTH+m;
          // Compare to the center
          cost_t tmp = (e1 >= e2);
          // Shift to the desired position
          tmp <<= shft;
          // Add it to its place
          census |= tmp;
          // Compare to the center
          cost_t tmp2 = (i1 >= i2);
          // Shift to the desired position
          tmp2 <<= shft;
          // Add it to its place
          census2 |= tmp2;
        }
      }

    transform[idy*cols+idx] = census;
    transform2[idy*cols+idx] = census2;
  }
}

} // namespace sgm_gpu

