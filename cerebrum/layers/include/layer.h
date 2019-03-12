#ifndef LAYER_H_
#define LAYER_H_

#include <Eigen/Dense>
#include <functional>
#include <stdexcept>

namespace layer {
  class Layer {
    public:
      template<int dim>
      Eigen::Vector<float, dim> feed_forward(
          double input,
          Eigen::MatrixXd weights,
          Eigen::Vector<float, dim> bias,
          std::function<double(double)> activation) {
        if (weights.rows() != bias.size()) {
          const std::string err_string =
            "Mismatched dimensions for weights and biases (weight: " + weights.rows() + " bias: " + bias.size()";
          throw new std::invalid_argument(err_string);
        }

        for (size_t i = 0; i < weights.rows(); ++i) {
          input = activation(weights(i).dot(input) + bias(i));
        }

        return input;
      }
  };

} // namespace layer

#endif // LAYER_H_
