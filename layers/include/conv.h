#ifndef CONV_H_
#define CONV_H_

#include <cassert>
#include <armadillo>
#include <iostream>
#include <vector>

namespace layer {
  class Conv {
    public:
      arma::cube accumulated_gradient_input;
      arma::cube gradient_input;

      std::vector<arma::cube> filters;
      std::vector<arma::cube> gradient_filters;

      Conv(
          size_t height,
          size_t width,
          size_t depth,
          size_t filter_height,
          size_t filter_width,
          size_t horizontal_stride,
          size_t vertical_stride,
          size_t num_filters) :
        height(height),
        width(width),
        depth(depth),
        filter_height(filter_height),
        filter_width(filter_width),
        horizontal_stride(horizontal_stride),
        vertical_stride(vertical_stride),
        num_filters(num_filters) {

        filters.resize(num_filters);
        for (size_t i = 0; i < num_filters; ++i) {
          filters[i] = arma::zeros(filter_height, filter_width, depth);
          filters[i].imbue([&]() { return _get_truncated_norm_dist_value(0.0, 1.0); });
        }

        _reset_accumulated_gradients();
      }

      void forward(arma::cube& input , arma::cube& output) {
        // TODO(jparr721) conver to custom error
        assert((height - filter_height) % vertical_stride == 0);
        assert((width - filter_width) % horizontal_stride == 0);

        output = arma::zeros((height - filter_height)/vertical_stride + 1,
                            (width - filter_width)/horizontal_stride + 1,
                            num_filters);

        for (size_t filterx = 0; filterx < num_filters; ++filterx) {
          for (size_t i = 0; i <= height - filter_height; i+= vertical_stride) {
            for (size_t j = 0; j <= width - filter_width; j += horizontal_stride) {
              output((i/vertical_stride), (j/horizontal_stride), filterx) = arma::dot(
                  arma::vectorise(
                      input.subcube(i, j, 0,
                                    i+filter_height-1, j+filter_width-1, depth-1)
                    ),
                  arma::vectorise(filters[filterx]));
            }
          }
        }

        this->input = input;
        this->output = output;
      }

      void backward(arma::cube& upstream_gradient) {
        // TODO(jparr721) convert to custom error
        assert(upstream_gradient.n_slices == num_filters);
        assert(upstream_gradient.n_rows == output.n_rows);
        assert(upstream_gradient.n_cols == output.n_cols);

        // Initialize the gradient input. Dims are same as input
        gradient_input = arma::zeros(arma::size(input));

        for (size_t filterx = 0; filterx < num_filters; ++filterx) {
          for (size_t row = 0; row < output.n_rows; ++row) {
            for (size_t col = 0; col < output.n_cols; ++col) {
              arma::cube tmp(arma::size(input), arma::fill::zeros);
              tmp.subcube(row * vertical_stride,
                          col * horizontal_stride,
                          0,
                          (row * vertical_stride) + filter_height - 1,
                          (col * horizontal_stride) + filter_width - 1,
                          depth - 1) = filters[filterx];
              gradient_input += upstream_gradient.slice(filterx)(row, col) * tmp;
            }
          }
        }

        // Update our accumulated gradient
        accumulated_gradient_input += gradient_input;

        // Initialize our gradient filters
        gradient_filters.clear();
        gradient_filters.resize(num_filters);

        for (size_t i = 0; i < num_filters; ++i) {
          gradient_filters[i] = arma::zeros(filter_height, filter_width, depth);
        }
        for (size_t filterx = 0; filterx < num_filters; ++filterx) {
          for (size_t row = 0; row < output.n_rows; ++row) {
            for (size_t col = 0; col < output.n_cols; ++col) {
              arma::cube tmp(arma::size(filters[filterx]), arma::fill::zeros);
              tmp = input.subcube(row * vertical_stride,
                                  col * horizontal_stride,
                                  0,
                                  (row * vertical_stride) + filter_height - 1,
                                  (col * horizontal_stride) + filter_width - 1,
                                  depth - 1);
              gradient_filters[filterx] += upstream_gradient.slice(filterx)(row, col) * tmp;
            }
          }
        }

        for (size_t i = 0; i < num_filters; ++i) {
          accumulated_gradient_filters[i] += gradient_filters[i];
        }
      }

      void update_filter_weights(size_t batch_size, double learning_rate) {
        for (size_t i = 0; i < num_filters; ++i) {
          filters[i] -= learning_rate * (accumulated_gradient_filters[i]/batch_size);
        }
        _reset_accumulated_gradients();
      }

    private:
      size_t height;
      size_t width;
      size_t depth;
      size_t filter_height;
      size_t filter_width;
      size_t horizontal_stride;
      size_t vertical_stride;
      size_t num_filters;

      arma::cube input;
      arma::cube output;
      std::vector<arma::cube> accumulated_gradient_filters;

      double _get_truncated_norm_dist_value(double mean, double variance) {
        double stddev = sqrt(variance);
        arma::mat candidate = {3.0 * stddev};
        while (std::abs(candidate[0] - mean) > 2.0 * stddev)
          candidate.randn(1, 1);
        return candidate[0];
      }

      void _reset_accumulated_gradients() {
        accumulated_gradient_filters.clear();
        accumulated_gradient_filters.resize(num_filters);
        for (size_t i = 0; i < num_filters; ++i) {
          accumulated_gradient_filters[i] = arma::zeros(
              filter_height, filter_width, depth);
          accumulated_gradient_input = arma::zeros(height, width, depth);
        }
      }
  };
} // namespace layer

#endif // CONV_H_
