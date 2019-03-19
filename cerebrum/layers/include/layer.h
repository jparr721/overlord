#ifndef LAYER_H_
#define LAYER_H_

#include <Eigen/Dense>
#include <functional>
#include <stdexcept>
#include <layers/layer_types.h>
#include <linalg/tensor.h>

namespace layer {
  template<class LT>
  class Layer {
    public:
      const LayerType type = LT;
      linalg::Tensor<float> input_gradient;
      linalg::Tensor<float> input;
      linalg::Tensor<float> output;

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
