#ifndef ACTIVATIONS_H_
#define ACTIVATIONS_H_

#include <eigen3/Eigen/Dense>
#include <functional>
#include <string>
#include <unordered_map>

namespace cerebrum {
  class Activations {
    public:
      Activations(
          std::string& activation,
          Eigen::VectorXf& input_layer);

      static void ReLu(Eigen::VectorXf& input_layer);
      static void Sigmoid(Eigen::VectorXf& input_layer);
      static void Tanh(Eigen::VectorXf& input_layer);
    private:
      const std::unordered_map<
        std::string,
        std::function<void(Eigen::VectorXf& input_layers)>> functions {
          { "relu", ReLu },
          { "sigmoid", Sigmoid },
          { "tanh", Tanh },
        };
  };
} // namespace cerebrum

#endif // ACTIVATIONS_H_
