#ifndef CENTERPOINT_UTILS_HPP
#define CENTERPOINT_UTILS_HPP

#include <cstddef>
#include <cstdlib>
#include <ctime>
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

std::size_t divup(const std::size_t a, const std::size_t b);

void init_time();

void random_input(std::vector<float>& input);

}  // namespace centerpoint

#endif  // CENTERPOINT_UTILS_HPP
