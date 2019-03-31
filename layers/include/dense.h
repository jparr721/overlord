#ifndef DENSE_H_
#define DENSE_H_

#include <armadillo>
#include <cmath>
#include <vector>

namespace layer {
  class Dense {
    public:
      arma::mat weights;
      arma::vec biases;

      arma::cube gradient_input;
      arma::mat gradient_weights;
      arma::vec gradient_biases;

      /// Dense layers take the inputs and construct a dense layer for
      /// a neural network architecture
      /// height {size_t} - The number of nodes in the total model
      /// width {size_t} - The number of nodes in the given layer
      /// depth {size_t} - The number of layers in the neural net
      Dense(size_t height, size_t width, size_t depth, size_t n_output):
        height(height), width(width), depth(depth), num_outputs(n_output) {
        weights = arma::zeros(num_outputs, height * width * depth);

        // Normalize our weights into the proper range
        weights.imbue([&]() { return _get_truncated_norm_dist_value(0.0, 1.0); });

        // Use a bias of zero unless otherwise specified
        biases = arma::zeros(num_outputs);
      }

      void forward(arma::cube& input, arma::vec& output) {
        // Flatten our input data
        arma::vec flat_input = arma::vectorise(input);

        // Then just multiply our weights by our flattened layer
        output = (weights * flat_input) + biases;

        // TODO (jparr721): Add an activation function option
        this->input = input;
        this->output = output;
      }

      /// Backpropagate at each node and multiply our weights by our
      /// selected gradient value.
      ///
      /// upstream_gradient {arma::vec} - Our error function calculation
      void backward(arma::vec& upstream_gradient) {
        arma::vec gradient_input_vec = arma::zeros(height * width * depth);
        for (size_t i = 0; i < (height * width * depth); ++i) {
          gradient_input_vec[i] = arma::dot(weights.col(i), upstream_gradient);
        }

        arma::cube tmp((height * width * depth), 1, 1);
        tmp.slice(0).col(0) = gradient_input_vec;
        gradient_input = arma::reshape(tmp, height, width, depth);

        accumulated_gradient_input += gradient_input;

        gradient_weights = arma::zeros(arma::size(weights));
        for (size_t i = 0; i < gradient_weights.n_rows; ++i) {
          gradient_weights.row(i) = vectorise(input).t() * upstream_gradient[i];
        }

        accumulated_gradient_weights += gradient_weights;
        gradient_biases = upstream_gradient;
        accumulated_gradient_biases += gradient_biases;
      }

      /// This applies our gradients to each of the neurons in our
      /// network which were accumulated during backpropagation.
      ///
      /// batch_size {size_t} - The number of training examples in our epoch
      /// learning_rate {double} - The learning rate of our descent (into madness)
      void apply_gradients_at_each_neuron(size_t batch_size, double learning_rate) {
        // Update our weight matrix
        weights -= learning_rate * accumulated_gradient_weights / batch_size;
        // Update our biases
        biases = biases - learning_rate * accumulated_gradient_biases / batch_size;

        // We must reset our gradients since its based on the backward pass anyway.
        _reset_accumulated_gradients();
      }

    private:
      size_t height;
      size_t width;
      size_t depth;
      arma::cube input;

      size_t num_outputs;
      arma::vec output;

      arma::cube accumulated_gradient_input;
      arma::mat accumulated_gradient_weights;
      arma::vec accumulated_gradient_biases;

      /// Truncating our normal distribution value keeps us from having
      /// any neurons that go out of the range of 0 to 1
      double _get_truncated_norm_dist_value(double mean, double variance) {
        double stddev = sqrt(variance);
        arma::mat candidate = {3.0 * stddev};
        while (std::abs(candidate[0] - mean) > 2.0 * stddev)
          candidate.randn(1, 1);
        return candidate[0];
      }

      /// Reset our gradient weight values at each node
      void _reset_accumulated_gradients() {
        accumulated_gradient_input = arma::zeros(height, width, depth);
        accumulated_gradient_weights = arma::zeros(num_outputs, height * width * depth);
        accumulated_gradient_biases = arma::zeros(num_outputs);
      }
  };
} // namespace layer

#endif // DENSE_H_
