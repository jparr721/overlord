#include <cassert>
#include "./include/file_reader.h"

namespace util {
  FileReader::FileReader(
      const std::string& train_data,
      const std::string& test_data,
      double split_ratio,
      bool with_validation) : _ratio(split_ratio), _with_validation(with_validation) {
    if (test_data == "") {
      auto loaded_data = train_test_split(train_data, split_ratio);
      this->train_data = loaded_data.first;
      this->test_data = loaded_data.second;
    } else {
      this->train_data.load(train_data, arma::csv_ascii);
      this->test_data.load(test_data, arma::csv_ascii);

      if (with_validation) {
        auto loaded_data = train_test_split(train_data, split_ratio);
        this->train_data = loaded_data.first;
        this->validation_data = loaded_data.second;
      }
    }
  }

  std::pair<arma::mat, arma::mat>
  FileReader::train_test_split(const std::string& train_data, double ratio, bool shuffle) {
    // First, load from the file
    this->train_data.load(train_data, arma::csv_ascii);

    // Then, if checked, shuffle the data
    if (shuffle) {
      this->train_data = arma::shuffle(this->train_data);
    }

    // Finally, slice by our ratio and return our split pairs
    return std::make_pair(slice(this->train_data, ratio), slice(this->train_data, 1 - ratio, true));
  }

  std::pair<arma::mat, arma::mat>
  FileReader::train_test_split(const arma::mat& train_data, double ratio, bool shuffle) {
    arma::mat new_data(train_data.n_rows, train_data.n_cols, arma::fill::zeros);
    if (shuffle) {
      new_data = arma::shuffle(train_data);
    }

    return std::make_pair(slice(new_data, ratio), slice(new_data, 1 - ratio, true));
  }

  arma::mat FileReader::slice(const arma::mat& input, double ratio, bool bottom) {
    assert(_ratio >= 0 && _ratio <= 1.0);
    double end_row{(ratio * input.n_rows) - 1};

    arma::mat retmat;

    // Conditional for bottom up or top-down iterators
    const arma::mat::const_row_iterator row_it_begin = bottom ? input.begin_row(end_row) : input.begin_row(0);
    const arma::mat::const_row_iterator row_it_end = bottom ? input.end_row(input.n_rows - end_row) : input.end_row(end_row);

    // Load up the matrix
    retmat.imbue([&]() {
      for (auto it = row_it_begin; it != row_it_end; ++it) {
        return *it;
      }
    });

    // Fire it off
    return retmat;
  }
} // namespace util
