#ifndef BASE_H_
#define BASE_H_

#include <armadillo>

namespace layer {
  class Base {
    public:

      arma::mat weights;
      arma::vec biases;

      arma::cube gradient_input;
      arma::mat gradient_weights;
      arma::vec gradient_biases;

      /// The default layer class houses the baseline code for makin a neural
      /// network layer. This abstract class can be used as a jumping-off
      /// point to start the other network layers
      Base(size_t height, size_t width, size_t depth) :
        height(height), width(width), depth(depth) {};

      virtual void forward(arma::cube& input, arma::vec& output) = 0;
      virtual void backward(arma::vec& upstream_gradient_weights) = 0;
      virtual void apply_gradients_at_each_neuron(size_t batch_height, double learning_rate) = 0;
    private:
      size_t height;
      size_t width;
      size_t depth;

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
  };
}
