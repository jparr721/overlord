#ifndef LAYER_H_
#define LAYER_H_

namespace layer {
  class Layer {
    public:
      Layer();
      const LayerType type = LT;
      Tensor<float> input_gradient;
      Tensor<float> input;
      Tensor<float> output;
  };

} // namespace layer

#endif // LAYER_H_
