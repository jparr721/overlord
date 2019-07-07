#define BOOST_TEST_MODULE InitializersTests
#define BOOST_TEST_DYN_LINK

#include <eigen3/Eigen/Dense>
#include <iostream>
#include <string>
#include <boost/test/unit_test.hpp>

#include <layers/base.h>
#include <initializers/initializers.h>

BOOST_AUTO_TEST_CASE(Constructor) {
  cerebrum::WeightsXf weights;
  weights.resize(1, 2);
  weights << 1, 1;

  std::string type = "glorot_uniform";
  cerebrum::Initializers<cerebrum::WeightsXf> i(
      type,
      weights);
}

BOOST_AUTO_TEST_CASE(GlorotUniformWorks) {
  cerebrum::WeightsXf weights;
  weights.resize(1, 2);
  weights << 1, 1;
  std::string type = "glorot_uniform";
  cerebrum::Initializers<cerebrum::WeightsXf> i(
      type,
      weights);
  std::cout << weights << std::endl;
}

BOOST_AUTO_TEST_CASE(GlorotNormalWorks) {
  cerebrum::WeightsXf weights;
  weights.resize(1, 2);
  weights << 1, 1;
  std::string type = "glorot_normal";
  cerebrum::Initializers<cerebrum::WeightsXf> i(
      type,
      weights);
  std::cout << weights << std::endl;
}

BOOST_AUTO_TEST_CASE(HeUniformWorks) {
  cerebrum::WeightsXf weights;
  weights.resize(1, 2);
  weights << 1, 1;
  std::string type = "he_uniform";
  cerebrum::Initializers<cerebrum::WeightsXf> i(
      type,
      weights);
  std::cout << weights << std::endl;
}

BOOST_AUTO_TEST_CASE(HeNormalWorks) {
  cerebrum::WeightsXf weights;
  weights.resize(1, 2);
  weights << 1, 1;
  std::string type = "he_normal";
  cerebrum::Initializers<cerebrum::WeightsXf> i(
      type,
      weights);
  std::cout << weights << std::endl;
}

BOOST_AUTO_TEST_CASE(RandomUniformWorks) {
  cerebrum::WeightsXf weights;
  weights.resize(1, 2);
  weights << 1, 1;
  std::string type = "random_uniform";
  cerebrum::Initializers<cerebrum::WeightsXf> i(
      type,
      weights);
  std::cout << weights << std::endl;
}

BOOST_AUTO_TEST_CASE(RandomNormalWorks) {
  cerebrum::WeightsXf weights;
  weights.resize(1, 2);
  weights << 1, 1;
  std::string type = "random_normal";
  cerebrum::Initializers<cerebrum::WeightsXf> i(
      type,
      weights);
  std::cout << weights << std::endl;
}
