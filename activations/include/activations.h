#ifndef ACTIVATIONS_H_
#define ACTIVATIONS_H_

#include <Eigen/Dense>
#include <string>

namespace cerebrum {
  class Activations {
    public:
      Activations(
          std::string& activation,
          Eigen::VectorXf& input_layer);

      static void ReLu(Eigen::VectorXf& input_layer);
      static void Sigmoid(Eigen::VectorXf& input_layer);
      static void Tanh(Eigen::VectorXf& input_layer);
  }
} // namespace cerebrum

#endif // ACTIVATIONS_H_
