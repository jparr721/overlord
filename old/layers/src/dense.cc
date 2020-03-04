#include <cassert>
#include <iostream>
#include <memory>

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
    assert(input.rows() == weights.cols());

    // First take our input by the weights, then add bias
    Eigen::VectorXf output = (input * weights) + biases;

    // Then, take our activation and apply it to the output
    auto activation = std::make_shared<Activations>(activation_, output);

    // Get new output size
    outputs_ = output.rows() * output.cols();

    return output;
  }

  /// Our backprop algorithm. The epsilon is calculated within the engine
  /// class with the provided target values. This is used as our upstream
  /// gradient to multiply by our weights in the matrix.
  void Dense::backward(const Eigen::VectorXf& epsilon) {
    assert(epsilon.size() == weights.size());

    // Our empty vector of deltas
    // this helps us determine weight direction change
    Eigen::VectorXf deltas = Eigen::VectorXf::Zero(epsilon.rows(), epsilon.cols());


  }
} // namespace cerebrum
