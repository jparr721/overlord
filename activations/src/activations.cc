#include <cmath>
#include <activations/activations.h>
#include <swiss/strings.h>

namespace cerebrum {
  Activations::Activations(
      std::string& activation,
      swiss::WeightsXf input_weights) {
    swiss::to_lower(activation);

    // Get the activation by name
    auto activation_ = functions.at(activation);

    // Execute on the input layer
    activation_(input_weights);
  }

  // TODO(jparr721) - Docs here
  void Activations::ReLu(swiss::WeightsXf& input_weights) {
    for (size_t i = 0; i < input_weights.size(); ++i) {
      float val = input_weights(i);
      input_weights(i) = std::max(val, 0.0f);
    }
  }

  // TODO(jparr721) - Docs here
  void Activations::Sigmoid(swiss::WeightsXf& input_weights) {
    for (size_t i = 0; i < input_weights.size(); ++i) {
      input_weights(i) = 1 / 1 + std::exp(input_weights(i));
    }
  }

  // TODO(jparr721) - Docs here
  void Activations::Tanh(swiss::WeightsXf& input_weights) {
    for (size_t i = 0; i < input_weights.size(); ++i) {
      input_weights(i) = 2 / (1 + std::exp(-2 * input_weights(i))) - 1;
    }
  }

  void Activations::dReLu(swiss::WeightsXf& inputs) {

  }
} // namespace cerebrum
