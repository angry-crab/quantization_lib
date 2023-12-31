#include "scatter_kernel.hpp"

#include "utils.hpp"

namespace
{
const std::size_t THREADS_PER_BLOCK = 32;
}  // namespace

namespace centerpoint
{
__global__ void scatterFeatures_kernel(
  const float * pillar_features, const int * coords, const std::size_t num_pillars,
  const std::size_t pillar_feature_size, const std::size_t grid_size_x,
  const std::size_t grid_size_y, float * scattered_features)
{
  // pillar_features: shape of (max_num_pillars, pillar_feature_size)
  // coords: shape of (max_num_pillars, 3)
  // scattered_features: shape of (num_pillars, grid_size_y, grid_size_x)
  const auto pillar_i = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;
  const auto feature_i = blockIdx.y * THREADS_PER_BLOCK + threadIdx.y;

  if (pillar_i >= num_pillars || feature_i >= pillar_feature_size) {
    return;
  }

  const int3 coord = ((int3 *)coords)[pillar_i];  // zyx
  if (coord.x < 0) {
    return;
  }

  const auto feature = pillar_features[pillar_feature_size * pillar_i + feature_i];
  scattered_features[grid_size_y * grid_size_x * feature_i + grid_size_x * coord.y + coord.z] =
    feature;
}

cudaError_t scatterFeatures_launch(
  const float * pillar_features, const int * coords, const std::size_t num_pillars,
  const std::size_t max_voxel_size, const std::size_t encoder_out_feature_size,
  const std::size_t grid_size_x, const std::size_t grid_size_y, float * scattered_features,
  cudaStream_t stream)
{
  dim3 blocks(
    divup(max_voxel_size, THREADS_PER_BLOCK), divup(encoder_out_feature_size, THREADS_PER_BLOCK));
  dim3 threads(THREADS_PER_BLOCK, THREADS_PER_BLOCK);
  scatterFeatures_kernel<<<blocks, threads, 0, stream>>>(
    pillar_features, coords, num_pillars, encoder_out_feature_size, grid_size_x, grid_size_y,
    scattered_features);

  return cudaGetLastError();
}

}  // namespace centerpoint
