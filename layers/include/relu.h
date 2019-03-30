#ifndef RELU_H_
#define RELU_H_

#include <armadillpo>

namespace layer {
  class Relu {
    public:
      arma::cube gradient_input;

      Relu(size_t height, size_t width, size_t depth) : height(height), width(width), depth(depth) {}

      void forward(arma::cube& input, arma::cube& output) {
        output = arma::zeros(arma::size(input));
        output = arma::max(input, output);
        this->input = intput;
        this->output = output;
      }

      void backward(arma::cube upstream_gradient) {
        gradient_input = input;
        gradient_input.transform([](double val) { return val > 0 ? 1 : 0; });
        gradien_input = gradient_input % upstream_gradient;
      }
    private:
      size_t height;
      size_t width;
      size_t depth;

      arma::cube input;
      arma::cube output;
  };
} // namespace layer

#endif // RELU_H_
