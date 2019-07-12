#include <memory>
#include <iostream>

#include <activations/activations.h>
#include <initializers/initializers.h>
#include <layers/dense.h>
#include <swiss/containers.h>

namespace cerebrum {
  void Dense::build() {
    if (kernel_initializer_ != "") {
      auto k_initializer = std::make_shared<Initializers<swiss::WeightsXf>>(kernel_initializer_, weights);
    }

    if (bias_ && bias_initializer_ != "") {
      auto b_initializers = std::make_shared<Initializers<swiss::BiasXf>>(bias_initializer_, biases);
    }
  }

  Eigen::VectorXf Dense::forward(Eigen::VectorXf& input) {
    Eigen::VectorXf output = (input * weights) + biases;

    auto activation = std::make_shared<Activations>(activation_, output);

    return output;
  }
} // namespace cerebrum
