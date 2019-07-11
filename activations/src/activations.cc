#include <cmath>
#include <activations/activations.h>

namespace cerebrum {
  Activations::Activations(
      std::string& activation,
      WeightsXf& input_weights) {
    std::transform(activation.begin(), activation.end(),
        activation.begin(), ::tolower);

    // Get the activation by name
    auto activation_ = functions.at(activation);

    // Execute on the input layer
    activation_(input_weights);
  }

  // TODO(jparr721) - Docs here
  void Activations::ReLu(WeightsXf& input_weights) {
    for (auto i = 0; i < input_weights.size(); ++i) {
      float val = input_weights(i);
      input_weights(i) = std::max(val, 0.0f);
    }
  }

  // TODO(jparr721) - Docs here
  void Activations::Sigmoid(WeightsXf& input_weights) {
    for (size_t i = 0; i < input_weights.size(); ++i) {
      input_weights(i) = 1 / 1 + std::exp(input_layer(i));
    }
  }

  // TODO(jparr721) - Docs here
  void Activations::Tanh(WeightsXf& input_weights) {

  }
} // namespace cerebrum
