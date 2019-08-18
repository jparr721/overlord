#ifndef ACTIVATIONS_H_
#define ACTIVATIONS_H_

#include <eigen3/Eigen/Dense>
#include <functional>
#include <string>
#include <unordered_map>

#include <swiss/containers.h>

namespace cerebrum {
  class Activations {
    public:
      Activations(
          std::string& activation,
          swiss::WeightsXf input_layer);

      /// ------------------------------------- Activations
      static void ReLu(swiss::WeightsXf& input_weights);
      static void Sigmoid(swiss::WeightsXf& input_weights);
      static void Tanh(swiss::WeightsXf& input_weights);

      /// ------------------------------------- Activation Derivatives
      static void dReLu(swiss::WeightsXf& input_weights);
      static void dSigmoid(swiss::WeightsXf& input_weights);
      static void dTanh(swiss::WeightsXf& input_weights);
    private:
      const std::unordered_map<
        std::string,
        std::function<void(swiss::WeightsXf& input_weights)>> functions {
          { "relu", ReLu },
          { "sigmoid", Sigmoid },
          { "tanh", Tanh },
          { "drelu", dReLu },
          { "dsigmoid", dSigmoid },
          { "dtanh", dTanh },
        };
  };
} // namespace cerebrum

#endif // ACTIVATIONS_H_
