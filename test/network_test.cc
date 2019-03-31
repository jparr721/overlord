#define BOOST_TEST_MODULE FullNetworkTests
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include <iostream>
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

constexpr auto DEBUG_PREFIX = "[FULL NETWORK TEST]\t";

BOOST_AUTO_TEST_CASE(ConvMnist) {
  // Load in our csv data
  Mnist mn("../../data");

  auto train_data = mn.train_data;
  auto train_labels = mn.train_labels;

  auto validation_data = mn.validation_data;
  auto validation_labels = mn.validation_labels;

  auto test_data = mn.test_data;

  const size_t TRAIN_DATA_SIZE = train_data.size();
  const size_t VALIDATION_DATA_SIZE = validation_data.size();
  const size_t TEST_DATA_SIZE = test_data.size();
  const float LEARNING_RATE = 0.05;
  const size_t EPOCHS = 10;
  const size_t BATCH_SIZE = 64;
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
  Conv c2(12, 12, 5, 5, 5, 1, 1, 16);
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
  arma::vec sOutput = arma::zeros(10);


  // Make loss and cumulative loss, cumulative loss totals loss over all examples in a minibatch
  double loss{0.0};
  double cumulative_loss{0.0};

  for (size_t epoch = 0; epoch < EPOCHS; ++epoch) {
    std::cout << DEBUG_PREFIX << "Epoch: " << epoch << std::endl;

    for (size_t batch = 0; batch < NUM_BATCHES; ++batch) {
      // Make a random batch of the input feature cube
      arma::vec minibatch(BATCH_SIZE, arma::fill::randu);
      minibatch *= (TRAIN_DATA_SIZE - 1);

      for (size_t i = 0; i < BATCH_SIZE; ++i) {
        std::cout << "Begin forward pass" << std::endl;
        c1.forward(train_data[minibatch[i]], c1Output);
        r1.forward(c1Output, r1Output);
        std::cout << "here" << std::endl;
        mp1.forward(r1Output, mp1Output);
        std::cout << "here.5" << std::endl;
        c2.forward(mp1Output, c2Output);
        std::cout << "here2" << std::endl;
        r2.forward(c2Output, r2Output);
        std::cout << "here3" << std::endl;
        mp2.forward(r2Output, mp2Output);
        std::cout << "here4" << std::endl;
        d.forward(mp2Output, dOutput);
        dOutput /= 100;
        s.forward(dOutput, sOutput);

        // Use cross entropy to check our ouput ratio
        loss = ce.forward(sOutput, train_labels[minibatch[i]]);
        std::cout << "loss for epoch: " << loss << std::endl;
        cumulative_loss += loss;

        std::cout << "Begin backpropagation" << std::endl;
        // Backpropagate
        ce.backward();
        arma::vec predicted_gradient_weight_distribution = ce.gradient_predicted_distribution;
        s.backward(predicted_gradient_weight_distribution);
        arma::vec gradient_weight_softmax_input = s.gradient_input;

        d.backward(gradient_weight_softmax_input);
        arma::cube gradient_weight_dense_input = d.gradient_input;

        mp2.backward(gradient_weight_dense_input);
        arma::cube gradient_weight_max_pooling_2_input = mp2.gradient_input;

        r2.backward(gradient_weight_max_pooling_2_input);
        arma::cube gradient_weight_relu_2_input = r2.gradient_input;

        c2.backward(gradient_weight_relu_2_input);
        arma::cube gradient_weight_conv_2_input = c2.gradient_input;

        mp1.backward(gradient_weight_conv_2_input);
        arma::cube gradient_weight_max_pooling_input = mp1.gradient_input;

        r1.backward(gradient_weight_max_pooling_input);
        arma::cube gradient_weight_relu_input = r1.gradient_input;

        c1.backward(gradient_weight_relu_input);
        arma::cube gradient_weight_conv_input = c1.gradient_input;
      }

      // Update network parameters
      d.apply_gradients_at_each_neuron(BATCH_SIZE, LEARNING_RATE);
      c1.update_filter_weights(BATCH_SIZE, LEARNING_RATE);
      c2.update_filter_weights(BATCH_SIZE, LEARNING_RATE);
    }

   std::cout << DEBUG_PREFIX << "Training loss for epoch: " << epoch << ": " << cumulative_loss / (BATCH_SIZE * NUM_BATCHES) << std::endl;

    // Compute training accuracy after each epoch
    double correct = 0.0;
    for (size_t i = 0; i < TRAIN_DATA_SIZE; ++i) {
        c1.forward(train_data[i], c1Output);
        r1.forward(c1Output, r1Output);
        mp1.forward(r1Output, mp1Output);
        c2.forward(mp1Output, c2Output);
        r2.forward(c2Output, r2Output);
        mp2.forward(r2Output, mp2Output);
        d.forward(mp2Output, dOutput);
        dOutput /= 100;
        s.forward(dOutput, sOutput);

      if (train_labels[i].index_max() == sOutput.index_max()) {
        correct += 1.0;
      }
    }

    std::cout << DEBUG_PREFIX << "Training accuracy: " << correct/TRAIN_DATA_SIZE << std::endl;

    cumulative_loss = 0.0;
    correct = 0.0;

    for (size_t i = 0; i < VALIDATION_DATA_SIZE; ++i) {
        c1.forward(validation_data[i], c1Output);
        r1.forward(c1Output, r1Output);
        mp1.forward(r1Output, mp1Output);
        c2.forward(mp1Output, c2Output);
        r2.forward(c2Output, r2Output);
        mp2.forward(r2Output, mp2Output);
        d.forward(mp2Output, dOutput);
        dOutput /= 100;
        s.forward(dOutput, sOutput);

      if (validation_labels[i].index_max() == sOutput.index_max()) {
        correct += 1.0;
      }
    }

    std::cout << DEBUG_PREFIX << "Validation loss: " << cumulative_loss / (BATCH_SIZE * NUM_BATCHES) << std::endl;
    std::cout << DEBUG_PREFIX << "Validation accuracy: " << correct / VALIDATION_DATA_SIZE << std::endl;
  }
}
