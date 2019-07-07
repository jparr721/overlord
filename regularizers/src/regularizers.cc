#include <algorithm>
#include <cassert>
#include <cmath>
#include <random>

#include <regularizers/regularizers.h>
#include <swiss/strings.h>


namespace cerebrum {
  template<typename WeightType>
  Regularizers<WeightType>::Regularizers(
      std::string& regularizer,
      WeightType& weights,
      float lambda) {
    swiss::to_lower(regularizer);

    // Get the regualarizer by name
    auto regularizer_ = functions.at(regularizer, lambda);

    // Execute it on the weights
    regularizer_(weights);
  }

  /// Implements L1 or Lasso Regression
  template<typename WeightType>
  void Regularizers<WeightType>::L1(WeightType& weights, float lambda) {
    for (size_t i = 0; i < weights.size(); ++i) {
      weights[i] = lambda * std::fabs(weights[i]);
    }
  }

  /// Implements L2 regularization
  template<typename WeightType>
  void Regularizers<WeightType>::L2(WeightType& weights, float lambda) {
    for (size_t i = 0; i < weights.size(); ++i) {
      weights[i] = lambda * std::pow(weights[i], 2);
    }
  }

  /// Implements dropout
  template<typename WeightType>
  void Regularizers<WeightType>::Dropout(WeightType& weights, float lambda) {
    assert(lambda <= 1.0);
    std::default_random_engine gen;
    std::normal_distribution<float> norm(0.0, 1.0);

    for (size_t i = 0; i < weights.size(); ++i) {
      // In this case, lambda_ is our dropout chance
      if (norm(gen) < lambda) {
        weights[i] = 0.0f;
      }
    }
  }
} // namespace cerebrum
