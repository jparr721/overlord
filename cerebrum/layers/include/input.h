#ifndef INPUT_H_
#define INPUT_H_

#include <Eigen/Dense>
#include <layers/layer_types.h>
#include <type_traits>

namespace layer {
  template<
    int dim,
    typename = typename std::enable_if<std::is_artihmetic<T>::value, T>::type
  >
  struct Input : public Layer<LayerType::input> {
    Input(Eigen::Vector<T, dim> inputs) inputs_(inputs) {};
    Eigen::Vector<T, dim> inputs_;
  };
} // namespace layer

#endif // INPUT_H_
