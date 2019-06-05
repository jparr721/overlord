#ifndef BASE_H_
#define BASE_H_

#include <eigen3/Eigen/Dense>

namespace cerebrum {
  class Base {
    public:
      Base(const int inputs, const int outputs, const bool bias=true)
        : inputs_(inputs), outputs_(outputs), bias_(bias) {};

      virtual ~Base();

      virtual void build() = 0;
      virtual void forward(Eigen::VectorXf& input) = 0;
      virtual void backward() = 0;
    protected:
      // The weights of our layer
      Eigen::VectorXf weights;

      // The bias units of our layer
      Eigen::VectorXf biases;

      int inputs_;
      int outputs_;
      bool bias_;

      bool built;
  };
} // namespace cerebrum

#endif // BASE_H_
