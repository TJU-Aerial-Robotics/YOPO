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

#include "flightlib/sensors/sgm_gpu/left_right_consistency.h"
#include "flightlib/sensors/sgm_gpu/configuration.h"

namespace sgm_gpu
{

__global__ void ChooseRightDisparity(uint8_t *right_disparity, const uint16_t *smoothed_cost, const uint32_t rows, const uint32_t cols) {
  const int x = blockIdx.x*blockDim.x+threadIdx.x;
  const int y = blockIdx.y*blockDim.y+threadIdx.y;
  
  if (x >= cols || y >= rows)
    return;
  
  int min_cost_disparity = 0;
  uint16_t min_cost = smoothed_cost[(y*cols + x)*MAX_DISPARITY + min_cost_disparity];
  
  for (int d = 1; d < MAX_DISPARITY; d++) {
    if (x + d >= cols)
      break;
    uint16_t tmp_cost = smoothed_cost[(y*cols + (x+d))*MAX_DISPARITY + d];
    if (tmp_cost < min_cost) {
      min_cost = tmp_cost;
      min_cost_disparity = d;
    }
  }
  
  right_disparity[y*cols+x] = min_cost_disparity;
}

__global__ void LeftRightConsistencyCheck(uint8_t* disparity, const uint8_t* disparity_right, uint32_t rows, uint32_t cols)
{
  const int x = blockIdx.x*blockDim.x+threadIdx.x;
  const int y = blockIdx.y*blockDim.y+threadIdx.y;
  
  if (x >= cols || y >= rows)
    return;
    
  const int x_right = x - disparity[y*cols + x];
  
  if (x_right < 0) {
    disparity[y*cols + x] = 255;
    return;
  }
  
  int diff = disparity[y*cols + x] - disparity_right[y*cols + x_right];
  diff = diff < 0 ? diff * -1 : diff;
  if (diff > 1) {
    disparity[y*cols + x] = 255;
  }
}

} // namespace sgm_gpu

