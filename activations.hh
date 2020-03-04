#pragma once

#include <eigen3/Eigen>

namespace overlord {
  class Activation {
    public:
      Activation();
      virtual ~Activation();
      virtual void Forward(Eigen::MatrixXf& inputs) = 0;
      virtual void Backward(Eigen::MatrixXf& inputs) = 0;

    protected:
      const Eigen::MatrixXf weights_;
  }

  class Sigmoid : private Activation {
    public:
      void Forward(Eigen::MatrixXf& inputs) {
        for (size_t i = 0; i < inputs.size(); ++i) {
          inputs(i) = Forward_(inputs(i));
        }
      }

      void Backward(Eigen::MatrixXf& inputs) {
        for (size_t i = 0; i < inputs.size(); ++i) {
          inputs(i) = Backward_(inputs(i));
        }
      }

    private:
      double Forward_(double x) {
        return 1 / (1 + std::exp(-x));
      }

      double Backward_(double x) {
        return Forward_(x) * (1 - Forward_(x));
      }
  };
} // namespace overlord
