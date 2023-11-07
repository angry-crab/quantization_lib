#ifndef CENTERPOINT_NETWORK_TRT_HPP
#define CENTERPOINT_NETWORK_TRT_HPP

#include "centerpoint_config.hpp"
#include "tensorrt_wrapper.hpp"

#include <vector>

namespace centerpoint
{
class VoxelEncoderTRT : public TensorRTWrapper
{
public:
  using TensorRTWrapper::TensorRTWrapper;

protected:
  bool setProfile(
    nvinfer1::IBuilder & builder, nvinfer1::INetworkDefinition & network,
    nvinfer1::IBuilderConfig & config) override;
};

class HeadTRT : public TensorRTWrapper
{
public:
  using TensorRTWrapper::TensorRTWrapper;

  HeadTRT(const std::vector<std::size_t> & out_channel_sizes, const CenterPointConfig & config);

protected:
  bool setProfile(
    nvinfer1::IBuilder & builder, nvinfer1::INetworkDefinition & network,
    nvinfer1::IBuilderConfig & config) override;

  std::vector<std::size_t> out_channel_sizes_;
};

}  // namespace centerpoint

#endif  // CENTERPOINT_NETWORK_TRT_HPP
