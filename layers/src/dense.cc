#include <layers/dense.h>
#include <initializers/initializers.h>

#include <memory>
#include <iostream>

namespace cerebrum {
  void Dense::build() {
    auto k_initializer = std::make_shared<Initializers>(kernel_initializer_, weights);

    if (bias_) {
      auto b_initializers = std::make_shared<Initializers>(bias_initializer_, biases);
    }
  }

  Eigen::VectorXf Dense::forward(Eigen::VectorXf& input) {
    auto output = (input * weights) + biases;
    return input;
  }
} // namespace cerebrum
