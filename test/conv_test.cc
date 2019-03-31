#define BOOST_TEST_MODULE ConvTests
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include "../layers/include/conv.h"

constexpr auto DEBUG = false;
constexpr auto DEBUG_PREFIX = "[DENSE LAYERS TESTS]\t";

BOOST_AUTO_TEST_CASE(Constructor) {
  layer::Conv c(
      5, // Input size
      5, // Input width
      3, // Input depth
      2, // Filter height
      3, // Filter width
      1, // Horizontal string
      1, // Vertical stride
      3); // Number of filters
}

BOOST_AUTO_TEST_CASE(Forward) {
  arma::cube input(3, 3, 1, arma::fill::zeros);
  input.slice(0) = {{1, 2, 3}, {2, 3, 4}, {3, 4, 5}};

  arma::cube filter1(2, 2, 1, arma::fill::zeros);
  filter1.slice(0) = {{1, 0}, {0, 1}};

  arma::cube filter2(2, 2, 1, arma::fill::zeros);
}
