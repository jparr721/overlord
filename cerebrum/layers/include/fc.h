#ifndef FULLY_CONNECTED_LAYER_H_
#define FULLY_CONNECTED_LAYER_H_

#include <cmath>
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
      std::vector<float> input_layer;
      linalg::Tensor<float> weights;
      std::vector<Gradient> gradients;

      FullyConnectedLayer(int output_size) :
        input(dim.x, dim.y, dim.z),
        output(output_size, 1, 1),
        input_gradient(dim.x, dim.y, dim.z),
        weights(in.x*in.y,in.z, output_size, 1) {
        input = std::vector<float>(output_size);
        gradients = std::vector<Graient>(output_size);

        int max_value = dim.x * dim.y * dim.z;

        for (int i = 0; i < output_size; ++i) {
          for (int j = 0; j < dim.x*dim.y*dim.z; ++j) {
            // 2.19722f = f^-1(0.9) => x where [1 / (1 + exp(-x)) = 0.9]
            weights(j, i, o) = 2.19722f / max_calue * rand() / float(RAND_MAX);
          }
        }
      }
  };

} // namespace layer

#endif // FULLY_CONNECTED_LAYER_H_
