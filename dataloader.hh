/**
 * DataLoader loads csvs with no frills or other bullshit.
 */

#pragma once

#include <sstream>
#include <string>
#include <vector>

#include <eigen3/Eigen/Dense>

namespace overlord {
  class DataLoader {
    Eigen::MatrixXf targets;
    Eigen::MatrixXf data;

    DataLoader(
      const std::string& csv,
      const std::string& target,
      const char delim=','
    ) {
      std::string line;

      while (std::getline(csv, line, delim)) {
      }
    }
  };
} // namespace overlord
