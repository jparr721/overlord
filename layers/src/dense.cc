#include <iostream>
#include <layers/dense.h>

namespace cerebrum {
  void Dense::build() {
    std::cout << "Initializing..." << std::endl;
  }

  void Dense::forward(Eigen::VectorXf& input) {
    auto output = (input * weights);
  }
} // namespace cerebrum
