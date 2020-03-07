#pragma once

#include <cassert>
#include <cmath>
#include <eigen3/Eigen/Dense>

#include "util.hh"

namespace overlord {
  class MSE {
    static constexpr char[3] loss_type = "MSE";

    void Loss(Eigen::MatrixXf predicted, Eigen::MatrixXf target) {
      assert(shape(predicted) == shape(target));
      return 0.5 * std::pow((predicted - actual).sum(), 2) / predicted.rows();
    }

    void Grad(Eigen::MatrixXf predicted, Eigen::MatrixXf target) {
      assert(shape(predicted) == shape(target));
      return (predicted - actual) / predicted.rows();
    }
  };

  class MAE {
    static constexpr char[3] loss_type = "MAE";

    void Loss(Eigen::MatrixXf predicted, Eigen::MatrixXf target) {
      assert(shape(predicted) == shape(target));
      return 0.5 * std::abs(predicted - actual).sum() / predicted.rows();
    }

    void Grad(Eigen::MatrixXf predicted, Eigen::MatrixXf target) {
      assert(shape(predicted) == shape(target));
      return sign((predicted - actual)) / predicted.rows();
    }
  };
} // namespace overlord
