#ifndef CENTERPOINT_POSTPROCESS_KERNEL_HPP
#define CENTERPOINT_POSTPROCESS_KERNEL_HPP

#include "centerpoint_config.hpp"
#include "utils.hpp"

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <thrust/device_vector.h>

#include <vector>

namespace centerpoint
{
struct Box3D
{
  int label;
  float score;
  float x;
  float y;
  float z;
  float length;
  float width;
  float height;
  float yaw;
  float vel_x;
  float vel_y;
};

class PostProcessCUDA
{
public:
  explicit PostProcessCUDA(const CenterPointConfig & config);

  cudaError_t generateDetectedBoxes3D_launch(
    const float * out_heatmap, const float * out_offset, const float * out_z, const float * out_dim,
    const float * out_rot, const float * out_vel, std::vector<Box3D> & det_boxes3d,
    cudaStream_t stream);

private:
  CenterPointConfig config_;
  thrust::device_vector<Box3D> boxes3d_d_;
  thrust::device_vector<float> yaw_norm_thresholds_d_;
};

}  // namespace centerpoint

#endif  // CENTERPOINT_POSTPROCESS_KERNEL_HPP
