#ifndef CENTERPOINT_CIRCLE_NMS_KERNEL_HPP
#define CENTERPOINT_CIRCLE_NMS_KERNEL_HPP

#include "utils.hpp"

#include <thrust/device_vector.h>

namespace centerpoint
{
// Non-maximum suppression (NMS) uses the distance on the xy plane instead of
// intersection over union (IoU) to suppress overlapped objects.
std::size_t circleNMS(
  thrust::device_vector<Box3D> & boxes3d, const float distance_threshold,
  thrust::device_vector<bool> & keep_mask, cudaStream_t stream);

}  // namespace centerpoint

#endif  // CENTERPOINT_CIRCLE_NMS_KERNEL_HPP
