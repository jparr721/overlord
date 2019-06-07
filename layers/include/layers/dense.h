#ifndef DENSE_H_
#define DENSE_H_

#include <eigen3/Eigen/Dense>
#include <functional>
#include <layers/base.h>
#include <string>

namespace cerebrum {
  class Dense : public Base {
    public:
      Dense(
          const int inputs,
          const int outputs,
          const bool use_bias=true,
          std::string& activation="relu",
          std::string& kernel_initializer="glorot_uniform",
          std::string& bias_initializer="zeros",
          std::string& kernel_regularizer=nullptr,
          std::string& bias_regularizer=nullptr,
          std::string& activity_regularizer=nullptr,
          std::string& kernel_constraint=nullptr,
          std::string& bias_constraint=nullptr) :
        kernel_initializer_(kernel_initializer),
        bias_initializer_(bias_initializer),
        kernel_regularizer_(kernel_regularizer),
        bias_regularizer_(bias_regularizer),
        activity_regularizer_(activity_regularizer),
        kernel_constraint_(kernel_constraint),
        bias_constraint_(bias_constraint),
        Base(inputs, outputs, use_bias) {};

      virtual ~Dense();
      virtual void build();
      virtual void forward(Eigen::VectorXf& input);

    private:
      std::string kernel_initializer_;
      std::string bias_initializer_;
      std::string kernel_regularizer_;
      std::string bias_regularizer_;
      std::string activity_regularizer_;
      std::string kernel_constraint_;
      std::string bias_constraint_;
  };

} // namespace cerebrum

#endif // DENSE_H_
