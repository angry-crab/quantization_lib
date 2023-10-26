#ifndef CENTERPOINT_PREPROCESS_KERNEL_HPP
#define CENTERPOINT_PREPROCESS_KERNEL_HPP

#include <cuda.h>
#include <cuda_runtime_api.h>

namespace centerpoint
{
cudaError_t generateFeatures_launch(
  const float * voxel_features, const float * voxel_num_points, const int * coords,
  const std::size_t num_voxels, const std::size_t max_voxel_size, const float voxel_size_x,
  const float voxel_size_y, const float voxel_size_z, const float range_min_x,
  const float range_min_y, const float range_min_z, float * features, cudaStream_t stream);

}  // namespace centerpoint

#endif  // CENTERPOINT_PREPROCESS_KERNEL_HPP
