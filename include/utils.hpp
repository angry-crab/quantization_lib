#ifndef CENTERPOINT_UTILS_HPP
#define CENTERPOINT_UTILS_HPP

#include <cstddef>

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

std::size_t divup(const std::size_t a, const std::size_t b);

}  // namespace centerpoint

#endif  // CENTERPOINT_UTILS_HPP
