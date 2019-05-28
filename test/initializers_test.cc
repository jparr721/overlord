#define BOOST_TEST_MODULE InitializersTests
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include <Eigen/Dense>
#include <initializers/initializers.h>
#include <iostream>

BOOST_AUTO_TEST_CASE(Constructor) {
  Eigen::Vector4f weights = Eigen::Vector4f::Zero();
  cerebrum::Initializers i(
      "glorot_uniform",
      weights);
}

BOOST_AUTO_TEST_CASE(GlorotUniformWorks) {
  Eigen::Vector4f weights = Eigen::Vector4f::Zero();
  cerebrum::Initializers i(
      "glorot_uniform",
      weights);
  std::cout << weights << std::endl;
}

BOOST_AUTO_TEST_CASE(GlorotNormalWorks) {
  Eigen::Vector4f weights = Eigen::Vector4f::Zero();
  cerebrum::Initializers i(
      "glorot_normal",
      weights);
  std::cout << weights << std::endl;
}

BOOST_AUTO_TEST_CASE(HeUniformWorks) {
  Eigen::Vector4f weights = Eigen::Vector4f::Zero();
  cerebrum::Initializers i(
      "he_uniform",
      weights);
  std::cout << weights << std::endl;
}

BOOST_AUTO_TEST_CASE(HeNormalWorks) {
  Eigen::Vector4f weights = Eigen::Vector4f::Zero();
  cerebrum::Initializers i(
      "he_normal",
      weights);
  std::cout << weights << std::endl;
}

BOOST_AUTO_TEST_CASE(RandomUniformWorks) {
  Eigen::Vector4f weights = Eigen::Vector4f::Zero();
  cerebrum::Initializers i(
      "random_uniform",
      weights);
  std::cout << weights << std::endl;
}

BOOST_AUTO_TEST_CASE(RandomNormalWorks) {
  Eigen::Vector4f weights = Eigen::Vector4f::Zero();
  cerebrum::Initializers i(
      "random_normal",
      weights);
  std::cout << weights << std::endl;
}
