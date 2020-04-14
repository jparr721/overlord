/**
 * DataLoader loads csvs with no frills or other bullshit.
 */

#pragma once

#include <algorithm>
#include <cctype>
#include <sstream>
#include <string>
#include <vector>

#include <eigen3/Eigen/Dense>

namespace overlord {
  class DataLoader {
    public:
      Eigen::MatrixXf targets;
      Eigen::MatrixXf data;

      DataLoader(
        const std::string& csv,
        const std::string& target,
        const char delim=',',
        const bool with_headers = true,
      ) delim_(delim) {

      }

    private:
      const char delim_;

      std::vector<std::string> ReadNextLine_(std::istream& str, const char delim=',') {
        std::vector<std::string> result;
        std::string line;
        std::stringstream line_stream(line);
        std::string cell;

        while (std::getline(line_stream, cell, delim)) {
          result.push_back(rstrip(cell));
        }
      }

      std::string rstrip(std::string& value) {
        return value.erase(std::remove_if(value.begin(), value.end(), std::isspace), value.end());
      }
  };
} // namespace overlord
