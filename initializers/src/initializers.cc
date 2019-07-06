#include <algorithm>
#include <cmath>
#include <random>

#include <initializers/initializers.h>
#include <swiss/strings.h>

namespace cerebrum {
  Initializers::Initializers(
      std::string& initializer,
      Eigen::VectorXf& weights) {
    swiss::to_lower(initializer);

    // Get the initializer by name
    auto initializer_ = functions.at(initializer);

    // Execute on the weights
    initializer_(weights);
  }

  // TODO(jparr721) - Docs here
  void Initializers::RandomNormal(Eigen::VectorXf& weights) {
    std::default_random_engine gen;
    std::normal_distribution<float> norm(0.0, 1.0);

    for (int i = 0u; i < weights.size(); ++i) {
      weights[i] = norm(gen);
    }
  }

  // TODO(jparr721) - Docs here
  void Initializers::RandomUniform(Eigen::VectorXf& weights) {
    std::default_random_engine gen;
    std::uniform_real_distribution<float> norm(0.0, 1.0);

    for (int i = 0u; i < weights.size(); ++i) {
      weights[i] = norm(gen);
    }
  }

  // TODO(jparr721) - Docs here
  void Initializers::Zeros(Eigen::VectorXf& weights) {
    for (int i = 0u; i < weights.size(); ++i) {
      weights[i] = 0.0;
    }
  }

  // TODO(jparr721) - Docs here
  void Initializers::HeUniform(Eigen::VectorXf& weights) {
    std::default_random_engine gen;
    std::uniform_real_distribution<float> norm(0.0, 1.0);

    for (int i = 0u; i < weights.size(); ++i) {
      weights[i] = norm(gen) * std::sqrt(2/weights.size());
    }
  }

  // TODO(jparr721) - Docs here
  void Initializers::HeNormal(Eigen::VectorXf& weights) {
    std::default_random_engine gen;
    std::normal_distribution<float> norm(0.0, 1.0);

    for (int i = 0u; i < weights.size(); ++i) {
      weights[i] = norm(gen) * std::sqrt(2/weights.size());
    }
  }

  // TODO(jparr721) - Docs here
  void Initializers::GlorotUniform(Eigen::VectorXf& weights) {
    std::default_random_engine gen;
    std::uniform_real_distribution<float> norm(0.0, 1.0);

    for (int i = 0u; i < weights.size(); ++i) {
      weights[i] = norm(gen) * std::sqrt(1/weights.size());
    }
  }

  // TODO(jparr721) - Docs here
  void Initializers::GlorotNormal(Eigen::VectorXf& weights) {
    std::default_random_engine gen;
    std::normal_distribution<float> norm(0.0, 1.0);

    for (int i = 0u; i < weights.size(); ++i) {
      weights[i] = norm(gen) * std::sqrt(1/weights.size());
    }
  }
}
