#include <cmath>
#include <activations/activations.h>

namespace cerebrum {
  Activations::Activations(
      std::string& activation,
      Eigen::VectorXf& input_layer) {
    std::transform(activation.begin(), activation.end(),
        activation.begin(), ::tolower);

    // Get the activation by name
    auto activation_ = functions.at(activation);

    // Execute on the input layer
    activation_(input_layer);
  }

  // TODO(jparr721) - Docs here
  void Activations::ReLu(Eigen::VectorXf& input_layer) {

  }

  // TODO(jparr721) - Docs here
  void Activations::Sigmoid(Eigen::VectorXf& input_layer) {
    for (auto& value : input_layer) {
      value = 1 / 1 + std::exp(-value);
    }
  }

  // TODO(jparr721) - Docs here
  void Activations::Tanh(Eigen::VectorXf& input_layer) {

  }
} // namespace cerebrum
