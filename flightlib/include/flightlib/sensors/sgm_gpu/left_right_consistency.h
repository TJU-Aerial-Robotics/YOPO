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

#ifndef SGM_GPU__LEFT_RIGHT_CONSISTENCY_H_
#define SGM_GPU__LEFT_RIGHT_CONSISTENCY_H_

#include <stdint.h>

namespace sgm_gpu
{

__global__ void ChooseRightDisparity(uint8_t *right_disparity, const uint16_t *smoothed_cost, const uint32_t rows, const uint32_t cols);
__global__ void LeftRightConsistencyCheck(uint8_t *disparity, const uint8_t *disparity_right, const uint32_t rows, const uint32_t cols);

}

#endif // SGM_GPU__LEFT_RIGHT_CONSISTENCY_H_

