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

      int absorb_dimension(dim_t d) {
        return d.z * (input.size.x * input.size.y) + d.y * (input.size.x) + d.x;
      }

      void activate(std::function<float(float)> activation_fun) {
        for (int i = 0; i < output.size.x; ++i) {
          float input_val = 0;

          for (int j = 0; j < input.size.x; ++j) {
            for (int k = 0; k < input.size.y; ++k) {
              for (int l = 0; l < input.size.z; ++l) {
                int m = absorb_dumension({j, k, l});
                input_val += input(j, k, l) * weights(m, i, 0);
              }
            }
          }

          input_layer[n] = input_val;
          output(i, 0, 0) = activation_fun(input_val);
        }
      }
  };

} // namespace layer

#endif // FULLY_CONNECTED_LAYER_H_
