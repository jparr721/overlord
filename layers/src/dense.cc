#include <memory>
#include <initiailizers/initializers.h>
#include <iostream>
#include <layers/dense.h>

namespace cerebrum {
  void Dense::build() {
    auto k_initializer = std::make_shared<Initializers>(kernel_initializer, weights);

    if (bias_) {
      auto b_initializers = std::make_shared<Initializers>(bias_iniializer, biases);
    }
  }

  // Consider switching to template-based architecture
  Eigen::VectorXf Dense::forward(Eigen::VectorXf& input) {
    auto output = (input * weights) + biases;
  }
} // namespace cerebrum
