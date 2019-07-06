#include <algorithm>
#include <regularizers/regularizers.h>
#include <swiss/strings.h>


namespace cerebrum {
  Regularizers::Regularizers(
      std::string& regularizer,
      Eigen::VectorXf& weights) {
    swiss::to_lower(regularizer);

    // Get the regualarizer by name
    auto regularizer_ = functions.at(regularizer);

    // Execute it on the weights
    regularizer_(weights);
  }


} // namespace cerebrum
