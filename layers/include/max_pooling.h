#ifndef MAX_POOLING_H_
#define MAX_POOLING_H_

#include <armadillo>
#include <cassert>
#include <iostream>

namespace layer {
  class MaxPooling {
    public:
      arma::cube gradient_input;

      MaxPooling(size_t height,
                 size_t width,
                 size_t depth,
                 size_t pooling_window_height,
                 size_t pooling_window_width,
                 size_t vertical_stride,
                 size_t horizontal_stride) :
        height(height),
        width(width),
        depth(depth),
        pooling_window_height(pooling_window_height),
        pooling_window_width(pooling_window_width),
        vertical_stride(vertical_stride),
        horizontal_stride(horizontal_stride) {};

      void forward(arma::cube& input, arma::cube& output) {
        assert((height - pooling_window_height) % vertical_stride == 0);
        assert((width - pooling_window_width) % horizontal_stride == 0);

        output = arma::zeros((height - pooling_window_height) / vertical_stride + 1,
                             (width - pooling_window_width) / horizontal_stride + 1,
                             depth);

        for (size_t layer = 0; layer < depth; ++layer) {
          for (size_t row = 0; row <= height - pooling_window_height; row += vertical_stride) {
            for (size_t col = 0; col <= width - pooling_window_width; col += horizontal_stride) {
              output.slice(layer)(row/vertical_stride, col/horizontal_stride) = input.slice(layer)
                .submat(row, col, row + pooling_window_height - 1, col + pooling_window_width - 1).max();

            }
          }
        }

        this->input = input;
        this->output = output;
      }

      void backward(arma::cube& upstream_gradient) {
        assert(upstream_gradient.n_slices == output.n_slices);
        assert(upstream_gradient.n_rows == output.n_rows);
        assert(upstream_gradient.n_cols == output.n_cols);

        gradient_input = arma::zeros(height, width, depth);

        for (size_t i = 0; i < depth; ++i) {
          for (size_t row = 0; row + pooling_window_height <= height; row += vertical_stride) {
            for (size_t col = 0; col + pooling_window_width <= width; col += horizontal_stride) {
              arma::mat tmp(pooling_window_height,
                            pooling_window_width,
                            arma::fill::zeros);
              tmp(input.slice(i).submat(row, col,
                    row + pooling_window_height - 1, col + pooling_window_width - 1)
                  .index_max()) = upstream_gradient.slice(i)(row/vertical_stride,
                    col/horizontal_stride);

              gradient_input.slice(i).submat(row, col,
                  row + pooling_window_height - 1, col + pooling_window_width - 1) += tmp;
            }
          }
        }
      }

      private:
        size_t height;
        size_t width;
        size_t depth;
        size_t pooling_window_height;
        size_t pooling_window_width;
        size_t vertical_stride;
        size_t horizontal_stride;

        arma::cube input;
        arma::cube output;
  };
} // namespace layer

#endif // MAX_POOLING_H_
