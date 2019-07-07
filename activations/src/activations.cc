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
    for (auto i = 0; i < input_layer.size(); ++i) {
      float val = input_layer(i);
      input_layer(i) = std::max(val, 0.0f);
    }
  }

  // TODO(jparr721) - Docs here
  void Activations::Sigmoid(Eigen::VectorXf& input_layer) {
    for (size_t i = 0; i < input_layer.size(); ++i) {
      input_layer(i) = 1 / 1 + std::exp(input_layer(i));
    }
  }

  // TODO(jparr721) - Docs here
  void Activations::Tanh(Eigen::VectorXf& input_layer) {

  }
} // namespace cerebrum
