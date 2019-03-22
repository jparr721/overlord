#ifndef DROPOUT_H_
#define DROPOUT_H_

#include <armadillo>
#include <random>

namespace layer {
  class Dropout {
    public:
      Dropout(size_t size, size_t width, size_t depth, float dp) :
        size(size), width(width), depth(depth), dropout_probability(dp) {};

      /// The drop function performs dropout on the data in the
      /// input vector and returns it into the output vector
      ///
      /// input {arma::cube} - The input feature cube
      /// output {arma::vec} - The transformed output vector
      /// TODO(jparr721): Add a way to make this work for non-images (matrix or cube input)
      void drop(arma::cube& input) {
        this->input = input;
        _activate();
      }
    private:
      size_t size;
      size_t width;
      size_t depth;

      float dropout_probability;

      arma::cube input;
      arma::cube hitmap;
      arma::cube output;
      arma::vec flattened_output;

      /// Perform the dropout activation where each node in the layer
      /// has a probability of being set to 0 to help prevent overfitting
      void _activate() {
        // calculate randomization probability with the mersienne twister
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(0.0, 1.0);

        for (int i = 0; i < size * width * depth; ++i) {
          bool active = dis(gen) <= dropout_probability;
          hitmap[i] = active;
          output[i] = active ? input[i] : 0.0f;
        }

        _flatten_output_features();
      }

      void _flatten_output_features() {
        flattened_output = arma::vectorise(output);
      }

  };
} // namespace layer

#endif // DROPOUT_H_
