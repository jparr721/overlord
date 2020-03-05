#pragma once

#include <eigen3/Eigen/Dense>
#include <unordered_map>

namespace overlord {
  template<class WeightsInitializer, class BiasInitializer>
  class Layer {
    public:
      Eigen::MatrixXf weights;
      Eigen::MatrixXf biases;
      bool is_init = false;

      Layer(WeightsInitializer w_init, BiasInitializer b_init) : w_init_(w_init), b_init_(b_init) {};

      virtual void Init() = 0;
      virtual int Forward() = 0;
      virtual void Backward () = 0;

    protected:

        const WeightsInitializer w_init_;
        const BiasInitializer b_init_;
  };

  template<class WeightsInitializer, class BiasInitializer>
  class Dense : private Layer<WeightsInitializer, BiasInitializer> {
    public:
      void Init() {
        w_init_->Init(weights);
        b_init_->Init(biases);

        is_init = true;
      }

      int Forward(Eigen::MatrixXf inputs) {
        this->inputs = inputs;
        return inputs.dot(weights) + biases;
      }

      int Backward(double gradient) {
        return weights.t().dot(gradient);
      }
  };
} // namespace overlord
