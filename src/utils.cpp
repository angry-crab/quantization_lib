#include "utils.hpp"

#include <stdexcept>

namespace centerpoint
{
std::size_t divup(const std::size_t a, const std::size_t b)
{
  if (a == 0) {
    throw std::runtime_error("A dividend of divup isn't positive.");
  }
  if (b == 0) {
    throw std::runtime_error("A divisor of divup isn't positive.");
  }

  return (a + b - 1) / b;
}

}  // namespace centerpoint
