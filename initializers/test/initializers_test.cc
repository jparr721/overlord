#define BOOST_TEST_MODULE InitializersTests
#define BOOST_TEST_DYN_LINK

#include <eigen3/Eigen/Dense>
#include <iostream>
#include <string>
#include <boost/test/unit_test.hpp>

#include <initializers/initializers.h>

BOOST_AUTO_TEST_CASE(Constructor) {
  Eigen::VectorXf weights = Eigen::VectorXf::Ones(4);
  std::string type = "glorot_uniform";
  cerebrum::Initializers i(
      type,
      weights);
}

BOOST_AUTO_TEST_CASE(GlorotUniformWorks) {
  Eigen::VectorXf weights = Eigen::VectorXf::Ones(4);
  std::string type = "glorot_uniform";
  cerebrum::Initializers i(
      type,
      weights);
  std::cout << weights << std::endl;
}

BOOST_AUTO_TEST_CASE(GlorotNormalWorks) {
  Eigen::VectorXf weights = Eigen::VectorXf::Ones(4);
  std::string type = "glorot_normal";
  cerebrum::Initializers i(
      type,
      weights);
  std::cout << weights << std::endl;
}

BOOST_AUTO_TEST_CASE(HeUniformWorks) {
  Eigen::VectorXf weights = Eigen::VectorXf::Ones(4);
  std::string type = "he_uniform";
  cerebrum::Initializers i(
      type,
      weights);
  std::cout << weights << std::endl;
}

BOOST_AUTO_TEST_CASE(HeNormalWorks) {
  Eigen::VectorXf weights = Eigen::VectorXf::Ones(4);
  std::string type = "he_normal";
  cerebrum::Initializers i(
      type,
      weights);
  std::cout << weights << std::endl;
}

BOOST_AUTO_TEST_CASE(RandomUniformWorks) {
  Eigen::VectorXf weights = Eigen::VectorXf::Ones(4);
  std::string type = "random_uniform";
  cerebrum::Initializers i(
      type,
      weights);
  std::cout << weights << std::endl;
}

BOOST_AUTO_TEST_CASE(RandomNormalWorks) {
  Eigen::VectorXf weights = Eigen::VectorXf::Ones(4);
  std::string type = "random_normal";
  cerebrum::Initializers i(
      type,
      weights);
  std::cout << weights << std::endl;
}
