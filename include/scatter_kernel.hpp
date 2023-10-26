#ifndef CENTERPOINT_SCATTER_KERNEL_HPP
#define CENTERPOINT_SCATTER_KERNEL_HPP

#include <cuda.h>
#include <cuda_runtime_api.h>

namespace centerpoint
{
cudaError_t scatterFeatures_launch(
  const float * pillar_features, const int * coords, const std::size_t num_pillars,
  const std::size_t max_voxel_size, const std::size_t encoder_out_feature_size,
  const std::size_t grid_size_x, const std::size_t grid_size_y, float * scattered_features,
  cudaStream_t stream);

}  // namespace centerpoint

#endif  // CENTERPOINT_SCATTER_KERNEL_HPP
