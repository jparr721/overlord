#define BOOST_TEST_MODULE FullNetworkTests
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include <string>
#include "../util/include/mnist.h"
#include "../layers/include/cross_entropy.h"
#include "../layers/include/conv.h"
#include "../layers/include/dense.h"
#include "../layers/include/dropout.h"
#include "../layers/include/max_pooling.h"
#include "../layers/include/relu.h"
#include "../layers/include/softmax.h"

using namespace util;
using namespace layer;

constexpr auto DEBUG = false;
constexpr auto DEBUG_PREFIX = "[FULL NETWORK TEST]\t";

BOOST_AUTO_TEST_CASE(ConvMnist) {
  // Load in our csv data
  Mnist mn("../data");

  auto train_data = mn.train_data;
  auto train_labels = mn.train_labels;

  auto validation_data = mn.validation_data;
  auto validation_labels = mn.validation_labels;

  auto test_data = mn.test_data;

  const size_t TRAIN_DATA_SIZE = train_data.size();
  const size_t VALIDATION_DATA_SIZE = validation_data.size();
  const size_t TEST_DATA_SIZE = test_data.size();
  const float learning_rate = 0.05;
  const size_t epochs = 100;
  const size_t batch_size = 64;
  const size_t NUM_BATCHES = TRAIN_DATA_SIZE / BATCH_SIZE;

  // Layers (TODO) Make a engine class and cache these
  // Input is a 28x28x1 image
  Conv c1(28, 28, 1, 5, 5, 1, 1, 6);
  // Output is 24x24x6

  // Relu 1
  Relu r1(24, 24, 6);
  // Ouput is 24x24x6

  // Input is 24x24x6
  MaxPooling mp1(24, 24, 5, 2, 2, 2, 2);
  // Output is 12x12x6

  // Input is 12x12x6
  Conv c2(12, 12, 6, 5, 5, 1, 1, 16);
  // Output is 8x8x16

  // Relu 2
  Relu r2(8, 8, 16);

  // Output is 8x8x16

  // Input is 8x8x16
  MaxPooling mp2(8, 8, 16, 2, 2, 2, 2);
  // Output is 4x4x16

  // Input is 4x4x16
  // Dense layer will flatten output vector
  Dense d(4, 4, 16, 10);

  // Dense output is a flattened vector with prediction

  // Softmax the output
  Softmax s(10);
  // Output vector size 10

  CrossEntropy ce(10);

  arma::cube c1Output = arma::zeros(24, 24, 6);
  arma::cube r1Output = arma::zeros(24, 24, 6);
  arma::cube mp1Output = arma::zeros(12, 12, 6);
  arma::cube c2Output = arma::zeros(8, 8, 16);
  arma::cube r2Output = arma::zeros(8, 8, 16);
  arma::cube mp2Output = arma::zeros(4, 4, 16);
  arma::vec dOutput = arma::zeros(10);
  arma::vec sOut = arma::zeros(10);

  double loss{0.0};
  double cumulative_loss{0.0};
}
