#pragma once

#include <eigen3/Eigen/Dense>
#include <iostream>
#include <unordered_map>

#include "debug.hh"

namespace overlord {
  template<class WeightsInitializer, class BiasInitializer>
  class Layer {
    public:
      Eigen::MatrixXf weights;
      Eigen::MatrixXf biases;
      bool is_init = false;

      Layer(WeightsInitializer w_init, BiasInitializer b_init, double lr) : w_init_(w_init), b_init_(b_init) {};

      virtual void Init(bool with_debug) = 0;
      virtual int Forward() = 0;
      virtual void Backward () = 0;

    protected:
        bool is_debug = false;
        const WeightsInitializer w_init_;
        const BiasInitializer b_init_;
  };

  template<class WeightsInitializer, class BiasInitializer>
  class Dense : private Layer<WeightsInitializer, BiasInitializer> {
    public:
      void Init(bool with_debug = false) {
        this->w_init_->Init(this->weights);
        this->b_init_->Init(this->biases);

        this->is_init = true;
        this->is_debug = with_debug;
      }

      int Forward(Eigen::MatrixXf inputs) {
        this->inputs = inputs;
        return inputs.dot(this->weights) + this->biases;
      }

      void Backward(double gradient, double output_delta) {
        if (this->is_debug) {
          debug::DbgPrint("Weights before backprop step", this->weights);
        }

        this->weights -= this->lr * this->weights.t().dot(output_delta) * gradient;
        this->biases -= this->lr * this->bises.sum();

        if (this->is_debug) {
          debug::DbgPrint("Weights after backprop step", this->weights);
        }
      }
  };
} // namespace overlord
