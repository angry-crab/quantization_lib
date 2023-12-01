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

void init_time() {
    srand(static_cast <unsigned> (time(0)));
}

void random_input(std::vector<float>& input, float min, float max) {

    for(int i = 0; i < input.size(); ++i) {
      if(i == 0) {
        input[i] = min;
        continue;
      }
      if(i == 1) {
        input[i] = max;
        continue;
      }
      // input[i] = (rand() / ( RAND_MAX / (100.0) ));
      float random = ((float) rand()) / (float) RAND_MAX;
      float diff = max - min;
      float r = random * diff;
      input[i] = min + r;
    }
}

}  // namespace centerpoint
