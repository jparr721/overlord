#ifndef BASE_H_
#define BASE_H_

#include <swiss/containers.h>

namespace cerebrum {
  class Base {
    public:
      Base(const int inputs, const int outputs, const bool bias=true)
        : inputs_(inputs), outputs_(outputs), bias_(bias) {
        weights.resize(inputs, outputs);
        biases.resize(outputs);
      }

      virtual ~Base();

      virtual void build() = 0;
      virtual Eigen::VectorXf forward(Eigen::VectorXf& input) = 0;
      virtual void backward() = 0;
    protected:
      int inputs_;
      int outputs_;
      bool bias_;

      // The weights of our layer
      swiss::WeightsXf weights;

      // The bias units of our layer
      swiss::BiasXf biases;

      bool built;
  };
} // namespace cerebrum

#endif // BASE_H_
