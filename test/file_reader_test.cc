#define BOOST_TEST_MODULE FileReaderTests
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include <iostream>
#include <string>
#include "../util/include/file_reader.h"

using namespace util;

constexpr auto DEBUG = false;
constexpr auto DEBUG_PREFIX = "[FILE READER TESTS]\t";
constexpr auto train_path = "../../data/mnist_train.csv";
constexpr auto test_path = "../../data/mnist_test.csv";

void _print_cached_members(const FileReader& f) {
  std::cout << DEBUG_PREFIX << std::endl;
  std::cout << "TestData--------------------------" << std::endl;
  std::cout << f.test_data.n_rows << std::endl;
  std::cout << f.test_data.n_cols << std::endl;
  std::cout << "ValidationData--------------------------" << std::endl;
  std::cout << f.validation_data.n_rows << std::endl;
  std::cout << f.validation_data.n_cols << std::endl;
  std::cout << "TrainData--------------------------" << std::endl;
  std::cout << f.train_data.n_rows << std::endl;
  std::cout << f.train_data.n_cols << std::endl;
}

BOOST_AUTO_TEST_CASE(ConstructorWithTestData) {
  FileReader f = FileReader(train_path, test_path, 0.7, false);
  _print_cached_members(f);
}

BOOST_AUTO_TEST_CASE(ConstructorWithoutTestData) {
  FileReader f = FileReader(train_path);
  _print_cached_members(f);
}
