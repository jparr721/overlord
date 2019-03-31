#ifndef SOFTMAX_H_
#define SOFTMAX_H_

#include <armadillo>

namespace layer {
  class Softmax {
    public:
      arma::vec input;
      arma::vec output;
      arma::vec gradient_input;

      Softmax(size_t n_inputs) : _n_inputs(n_inputs) {}

      void forward(arma::vec& input, arma::vec& output) {
        double sum_exp = arma::accu(arma::exp(input - arma::max(input)));
        output = arma::exp(input - arma::max(input))/sum_exp;

        // Cache input and output for backward pass
        this->input = input;
        this->output = output;
      }

      void backward(arma::vec& upstream_gradient) {
        double sub = arma::dot(upstream_gradient, output);
        gradient_input = (upstream_gradient - sub) % output;
      }
    private:
      size_t _n_inputs;
  };
} // namespace layer

#endif // SOFTMAX_H_
