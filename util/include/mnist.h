#ifndef MNIST_H_
#define MNIST_H_

#include <armadillo>
#include <cassert>
#include <string>
#include <vector>

namespace util {
  struct Mnist {
    std::vector<arma::cube> train_data;
    std::vector<arma::cube> validation_data;
    std::vector<arma::cube> test_data;

    std::vector<arma::vec> train_labels;
    std::vector<arma::vec> validation_labels;

    Mnist(const std::string& data_dir, double ratio = 0.9) : _data_dir(data_dir), _ratio(ratio) {
      assert(ratio <= 1.0 && ratio >= 0.0);
      const std::string train_file = data_dir + "/mnist_train.csv";
      const std::string test_file = data_dir + "/mnist_test.csv";

      arma::mat train_data_raw;

      train_data_raw.load(train_file, arma::csv_ascii);
      train_data_raw = train_data_raw.submat(1, 0, train_data_raw.n_rows - 1, train_data_raw.n_cols - 1);

      int n_examples = train_data_raw.n_rows;

      std::vector<arma::cube> train_data_all;
      std::vector<arma::vec> train_labels_all;

      for (size_t i = 0; i < train_data_raw.n_rows; ++i) {
        int label = (int)train_data_raw.row(i)(0);
        arma::cube img(28, 28, 1, arma::fill::zeros);
        for (size_t row = 0; row < 28; ++row) {
          img.slice(0).row(row) = train_data_raw.row(i).subvec(28 * row + 1, 28 * row + 28);
        }

        img.slice(0) = arma::normalise(img.slice(0));
        train_data_all.push_back(img);
        arma::vec labels(10, arma::fill::zeros);
        labels[label] += 1.0;
        train_labels_all.push_back(labels);
      }

      // Split train and test and validation
      train_data = std::vector<arma::cube>(train_data_all.begin(), train_data_all.begin() + n_examples * _ratio);
      train_labels = std::vector<arma::vec>(train_labels_all.begin(), train_labels_all.begin() + n_examples * _ratio);
      validation_data = std::vector<arma::cube>(train_data_all.begin() + n_examples * _ratio, train_data_all.end());
      validation_labels = std::vector<arma::vec>(train_labels_all.begin() + n_examples * _ratio, train_labels_all.end());

      arma::mat test_data_raw;
      test_data_raw.load(test_file, arma::csv_ascii);
      test_data_raw = test_data_raw.submat(1, 0, test_data_raw.n_rows - 1, test_data_raw.n_cols - 1);

      for (size_t i = 0; i < test_data_raw.n_rows; ++i) {
        arma::cube img(28, 28, 1, arma::fill::zeros);
        for (size_t row = 0; row < 28; ++row) {
          img.slice(0).row(row) = test_data_raw.row(i).subvec(28 * row, 28 * row + 27);
        }

        img.slice(0) /= 255.0;
        test_data.push_back(img);
      }
    }

    private:
      const std::string _data_dir;

      const double _ratio;
  };
} // namespace util

#endif // MNIST_H_
