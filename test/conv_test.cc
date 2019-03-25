#define BOOST_TEST_MODULE ConvTests
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include "../layers/include/conv.h"

#define DEBUG false
#define DEBUG_PREFIX "[CONV LAYER TESTS ]\t"

BOOST_AUTO_CASE(ConstructorTest) {
  Conv c(
      5, // Input size
      5, // Input width
      3, // Input depth
      2, // Filter height
      3, // Filter width
      1, // Horizontal string
      1, // Vertical stride
      3); // Number of filters
}
