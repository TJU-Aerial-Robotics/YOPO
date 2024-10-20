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

#ifndef SGM_GPU__CONFIGURATION_H_
#define SGM_GPU__CONFIGURATION_H_

#include <stdint.h>

#define	MAX_DISPARITY 128
#define CENSUS_WIDTH  9
#define CENSUS_HEIGHT 7

#define TOP  (CENSUS_HEIGHT-1)/2
#define LEFT (CENSUS_WIDTH-1)/2

namespace sgm_gpu
{

typedef uint32_t cost_t;

}

#define COSTAGG_BLOCKSIZE       GPU_THREADS_PER_BLOCK
#define COSTAGG_BLOCKSIZE_HORIZ GPU_THREADS_PER_BLOCK

#endif // SGM_GPU__CONFIGURATION_H_
