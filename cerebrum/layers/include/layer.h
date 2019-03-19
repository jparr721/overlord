#ifndef LAYER_H_
#define LAYER_H_

#include <Eigen/Dense>
#include <functional>
#include <stdexcept>
#include <layers/layer_types.h>
#include <linalg/tensor.h>

namespace layer {
  template<class LT>
  class Layer {
    public:
      Layer();

      const LayerType type = LT;
      linalg::Tensor<float> input_gradient;
      linalg::Tensor<float> input;
      linalg::Tensor<float> output;
  };

} // namespace layer

#endif // LAYER_H_
