#include <algorithm>
#include <regularizers/regularizers.h>


namespace cerebrum {
  Regularizers::Regularizers(
      std::string& regularizer,
      Eigen::VectorXf& weights) {
    std::transform(regualrizer.begin(), regularizer.end(),
        regularizer.begin(), ::tolower);

    // Get the regualarizer by name
    auto regularizer_ = functions.at(regularizer);

    // Execute it on the weights
    regualizer_(weights);
  }


} // namespace cerebrum
