#ifndef READ_CSV_H_
#define READ_CSV_H_

#include <armadillo>
#include <string>
#include <utility>

namespace util {
  struct FileReader {
    arma::mat train_data;
    arma::mat test_data;
    arma::mat validation_data;

    FileReader(
        const std::string& train_data,
        const std::string& test_data="",
        double split_ratio=0.7,
        bool with_validation=true);

    private:
      std::pair<arma::mat, arma::mat>
      train_test_split(const std::string& train_data, double ratio=0.7, bool shuffle=true);

      std::pair<arma::mat, arma::mat>
      train_test_split(const arma::mat& train_data, double ratio=0.7, bool shuffle=true);
      arma::mat slice(const arma::mat& input, double ratio=0.7, bool bottom=false);

      const double _ratio;
      const bool _with_validation;
  };
}

#endif // READ_CSV_H_
