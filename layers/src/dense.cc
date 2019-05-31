#include <layers/dense.h>

namespace cerebrum {
  void Dense::build() {

  }

  void Dense::forward(Eigen::VectorXf& input) {
    auto output = (input * weights)
  }
} // namespace cerebrum
