#include <algorithm>
#include <cmath>
#include <random>

#include <initializers/initializers.h>
#include <swiss/strings.h>

namespace cerebrum {
  template<typename WeightType>
  Initializers<WeightType>::Initializers(
      std::string& initializer,
      WeightType& weights) {
    swiss::to_lower(initializer);

    // Get the initializer by name
    auto initializer_ = functions.at(initializer);

    // Execute on the weights
    initializer_(weights);
  }

  // TODO(jparr721) - Docs here
  template<typename WeightType>
  void Initializers<WeightType>::RandomNormal(WeightType& weights) {
    std::default_random_engine gen;
    std::normal_distribution<float> norm(0.0, 1.0);

    for (int i = 0u; i < weights.size(); ++i) {
      weights[i] = norm(gen);
    }
  }

  // TODO(jparr721) - Docs here
  template<typename WeightType>
  void Initializers<WeightType>::RandomUniform(WeightType& weights) {
    std::default_random_engine gen;
    std::uniform_real_distribution<float> norm(0.0, 1.0);

    for (int i = 0u; i < weights.size(); ++i) {
      weights[i] = norm(gen);
    }
  }

  // TODO(jparr721) - Docs here
  template<typename WeightType>
  void Initializers<WeightType>::Zeros(WeightType& weights) {
    for (int i = 0u; i < weights.size(); ++i) {
      weights[i] = 0.0;
    }
  }

  // TODO(jparr721) - Docs here
  template<typename WeightType>
  void Initializers<WeightType>::HeUniform(WeightType& weights) {
    std::default_random_engine gen;
    std::uniform_real_distribution<float> norm(0.0, 1.0);

    for (int i = 0u; i < weights.size(); ++i) {
      weights[i] = norm(gen) * std::sqrt(2/weights.size());
    }
  }

  // TODO(jparr721) - Docs here
  template<typename WeightType>
  void Initializers<WeightType>::HeNormal(WeightType& weights) {
    std::default_random_engine gen;
    std::normal_distribution<float> norm(0.0, 1.0);

    for (int i = 0u; i < weights.size(); ++i) {
      weights[i] = norm(gen) * std::sqrt(2/weights.size());
    }
  }

  // TODO(jparr721) - Docs here
  template<typename WeightType>
  void Initializers<WeightType>::GlorotUniform(WeightType& weights) {
    std::default_random_engine gen;
    std::uniform_real_distribution<float> norm(0.0, 1.0);

    for (int i = 0u; i < weights.size(); ++i) {
      weights[i] = norm(gen) * std::sqrt(1/weights.size());
    }
  }

  // TODO(jparr721) - Docs here
  template<typename WeightType>
  void Initializers<WeightType>::GlorotNormal(WeightType& weights) {
    std::default_random_engine gen;
    std::normal_distribution<float> norm(0.0, 1.0);

    for (int i = 0u; i < weights.size(); ++i) {
      weights[i] = norm(gen) * std::sqrt(1/weights.size());
    }
  }
}
