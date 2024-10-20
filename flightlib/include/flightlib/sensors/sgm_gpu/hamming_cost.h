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

#ifndef SGM_GPU__HAMMING_COST_H_
#define SGM_GPU__HAMMING_COST_H_

#include "flightlib/sensors/sgm_gpu/configuration.h"
#include "flightlib/sensors/sgm_gpu/util.h"
#include <stdint.h>

namespace sgm_gpu
{

__global__ void
HammingDistanceCostKernel(const cost_t *d_transform0, const cost_t *d_transform1, uint8_t *d_cost, const int rows, const int cols );

}

#endif // SGM_GPU__HAMMING_COST_H_ 

