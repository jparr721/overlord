#ifndef ACTIVATIONS_H_
#define ACTIVATIONS_H_

#include <eigen3/Eigen/Dense>
#include <functional>
#include <string>
#include <unordered_map>

#include <layers/base.h>

namespace cerebrum {
  class Activations {
    public:
      Activations(
          std::string& activation,
          WeightsXf& input_layer);

      static void ReLu(WeightsXf& input_weights);
      static void Sigmoid(WeightsXf& input_weights);
      static void Tanh(WeightsXf& input_weights);
    private:
      const std::unordered_map<
        std::string,
        std::function<void(WeightsXf& input_weights)>> functions {
          { "relu", ReLu },
          { "sigmoid", Sigmoid },
          { "tanh", Tanh },
        };
  };
} // namespace cerebrum

#endif // ACTIVATIONS_H_
