#ifndef CONV_H_
#define CONV_H_

#include <cmath>
#include <cassert>
#include <armadillo>
#include <vector>

namespace layer {
  class Conv {
    public:
      Conv(
          size_t size,
          size_t width,
          size_t depth,
          size_t filter_height,
          size_t filter_width,
          size_t horizontal_stride,
          size_t vertical_stride,
          size_t num_filters) :
        size(size),
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

    private:
      size_t size;
      size_t width;
      size_t depth;
      size_t filter_size;
      size_t filter_width;
      size_t horizontal_stride;
      size_t vertical_stride;
      size_t num_filters;

      std::vector<arma::cube> filters;

      arma::cube input;
      arma::cube output;
      arma::cube gradient_input;
      arma::cube accumulated_gradient_input;
      std::vector<arma::cube> gradient_filters;
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
          accumulated_gradient_filters[i] = arma::
        }
      }
  };
} // namespace layer

#endif // CONV_H_
