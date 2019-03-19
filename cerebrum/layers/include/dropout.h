#ifndef DROPOUT_H_
#define DROPOUT_H_

#include <random>
#include <layers/layer_types.h>
#include <linalg/dim.h>
#include <linalg/tensor.h>

namespace layer {
#ifdef EMBEDDED
  #pragma pack(push, 1)
#endif
  template<dim_t dim>
  class DropoutLayer : public <LayerType::dropout>Layer {
    public:
      linalg::Tensor<bool> hitmap;
      float activation_;

      /// Dropout layers take the activation as an input parameter
      /// which is the probability of a node being turned off
      ///
      /// @param {flaot} activation - The activation probability
      DropoutLayer(float activation) :
        input(dim.x, dim.y, dim.z),
        output(dim.x, dim.y, dim.z)
        hitmap(dim.x, dim.y, dim.z)
        input_gradient(dim.x, dim.y, dim.z)
        activation_(activation) {}

      void activate(linalg::Tensor<float>& input) {
        this->input = input;
        activate();
      }

      /// Internal gradient optimization algorithm (backprop)
      void calculate_gradients(linalg::Tensor<float>& next_gradient_layer) {
        for (int i = 0; i < input.size.x * input.size.y * input.size.z; ++i) {
          // Return our modified gradient into the new buffer with dropout considered
          input_gradient.data[i] = hitmap.data[i] ? next_gradient_layer.data[i] : 0.0f;
        }
      }


    private:
      /// Perform the dropout activation where each node in the layer
      /// has a probability of being deactivated to prevent overfitting
      void activate() {
        // Calculate randomness with mersienne twister randomizer algorithm
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(0.0, RAND_MAX);

        for (int i = 0; i < input.size.x * input.size.y * input.size.z; ++i) {

          bool active = dis(gen)/(float)RAND_MAX <= activation_;
          hitmap.data[i] = active;
          out.data[i] = active ? in.data[i] : 0.0f;
        }
      }
  };
} // namespace layer

#ifdef EMBEDDED
  #pragma pack(pop)
#endif
#endif // DROPOUT_H_
