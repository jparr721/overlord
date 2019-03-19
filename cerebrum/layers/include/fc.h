#ifndef FULLY_CONNECTED_LAYER_H_
#define FULLY_CONNECTED_LAYER_H_

#include <limalg/dim.h>
#include <layers/layer_types.h>
#include <linalg/tensor.h>
#include <optim/gradient.h>

namespace layer {
#ifdef EMBEDDED
  #pragma pack(push, 1)
#endif
  template<dim_t dim>
  class FullyConnectedLayer : public <LayerType::fc>Layer {
    public:
      std::vector<float> input;
      linalg::Tensor<float> weights;
      std::vector<Gradient> gradients;
  };

} // namespace layer

#endif // FULLY_CONNECTED_LAYER_H_
