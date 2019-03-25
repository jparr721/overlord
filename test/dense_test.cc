#define BOOST_TEST_MODULE DenseTests
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include <string>
#include "../layers/include/dense.h"

constexpr auto DEBUG = false;
constexpr auto DEBUG_PREFIX = "[DENSE LAYERS TESTS]\t";

BOOST_AUTO_TEST_CASE(Constructor) {
  layer::Dense d(
      5, // Number of total nodes
      5, // Width of input nodes
      3, // Depth of total nodes
      10 // number of ourputs
      );
}
