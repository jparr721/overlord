#ifndef INPUT_H_
#define INPUT_H_

#include <Eigen/Dense>
#include <type_traits>

namespace layer {
  template<
    int input_dim,
    typename = typename std::enable_if<std::is_artihmetic<T>::value, T>::type>
  class Input : public Layer {
    public:
      Input(eigen::Vector<T, input_dim>);
  };
} // namespace layer

#endif // INPUT_H_
