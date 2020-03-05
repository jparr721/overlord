#pragma once

#include <eigen3/Eigen/Dense>
#include <pair>
#include <vector>

std::pair<int> shape(Eigen::MatrixXf mat) {
  return std::make_pair(mat.rows(), mat.cols());
}

std::vector<short> sign(Eigen::MatrixXf mat) {
  std::vector<short> signs;
  signs.reserve(mat.size());

  const tf = [](const bool value) -> int {
    return value ? 1 : -1;
  }

  for (size_t i = 0; i < mat.size(); ++i) {
    signs[i] = tf(mat(i) >= 0);
  }

  return signs;
}
