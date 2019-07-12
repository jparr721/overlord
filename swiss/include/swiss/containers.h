#pragma once

#include <eigen3/Eigen/Dense>

namespace cerebrum { namespace swiss {
  typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> WeightsXf;
  typedef Eigen::Matrix<float, Eigen::Dynamic, 1> BiasXf;
} // namespace swiss
} // namespace cerebrum
