#ifndef CENTERPOINT_VOXEL_GENERATOR_HPP
#define CENTERPOINT_VOXEL_GENERATOR_HPP

#include "centerpoint_config.hpp"

#include <memory>
#include <vector>

namespace centerpoint
{
class VoxelGeneratorTemplate
{
public:
  explicit VoxelGeneratorTemplate(const CenterPointConfig & config);

  virtual std::size_t pointsToVoxels(
    std::vector<float> & data,
    std::vector<float> & voxels, std::vector<int> & coordinates,
    std::vector<float> & num_points_per_voxel) = 0;

  // bool enqueuePointCloud(
  //   const sensor_msgs::msg::PointCloud2 & input_pointcloud_msg, const tf2_ros::Buffer & tf_buffer);

protected:

  CenterPointConfig config_;
  std::array<float, 6> range_;
  std::array<int, 3> grid_size_;
  std::array<float, 3> recip_voxel_size_;
};

class VoxelGenerator : public VoxelGeneratorTemplate
{
public:
  using VoxelGeneratorTemplate::VoxelGeneratorTemplate;

  std::size_t pointsToVoxels(
    std::vector<float> & data,
    std::vector<float> & voxels, std::vector<int> & coordinates,
    std::vector<float> & num_points_per_voxel) override;
};

}  // namespace centerpoint

#endif  // CENTERPOINT_VOXEL_GENERATOR_HPP
