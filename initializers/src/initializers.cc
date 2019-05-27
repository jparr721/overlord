#include <algorithm>
#include <initializers/initializers.h>

namespace cerebrum {
  Initializers::Initializers(
      std::string& initializer,
      Eigen::VectorXf& weights) {
    std::transform(initializer.begin(), initializer.end(),
        initializer.begin(), ::tolower);

    // Get the initializer by name
    auto initializer = functions.at(initializer);

    // Execute on the weights
    initializer(weights);
  }
}
