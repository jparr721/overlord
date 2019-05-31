#include <algorithm>
#include <cmath>
#include <initializers/initializers.h>
#include <random>

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

  // TODO(jparr721) - Docs here
  void Initializers::RandomNormal(Eigen::VectorXf& weights) {
    std::default_random_engine gen;
    std::normal_distribution<float> norm(0.0, 1.0);

    for (auto& weight : weights) {
      weight = norm(gen);
    }
  }

  // TODO(jparr721) - Docs here
  void Initializers::RandomUniform(Eigen::VectorXf& weights) {
    std::default_random_engine gen;
    std::uniform_real_distribution<float> norm(0.0, 1.0);

    for (auto& weight : weights) {
      weight = norm(gen);
    }
  }

  // TODO(jparr721) - Docs here
  void Initializers::Zeros(Eigen::VectorXf& weights) {
    for (auto& weight : weights) {
      weight = 0.0;
    }
  }

  // TODO(jparr721) - Docs here
  void Initializers::HeUniform(Eigen::VectorXf& weights) {
    std::default_random_engine gen;
    std::uniform_real_distribution<float> norm(0.0, 1.0);

    for (auto& weight : weights) {
      weight = norm(gen) * std::sqrt(2/weights.size());
    }
  }

  // TODO(jparr721) - Docs here
  void Initializers::HeNormal(Eigen::Vectorxf& weights) {
    std::default_random_engine gen;
    std::normal_distribution<float> norm(0.0, 1.0);

    for (auto& weight : weights) {
      weight = norm(gen) * std::sqrt(2/weights.size());
    }
  }

  // TODO(jparr721) - Docs here
  void Initializers::GlorotUniform(Eigen::Vectorxf& weights) {
    std::default_random_engine gen;
    std::uniform_real_distribution<float> norm(0.0, 1.0);

    for (auto& weight : weights) {
      weight = norm(gen) * std::sqrt(1/weights.size());
    }
  }

  // TODO(jparr721) - Docs here
  void Initializers::GlorotNormal(Eigen::Vectorxf& weights) {
    std::default_random_engine gen;
    std::normal_distribution<float> norm(0.0, 1.0);

    for (auto& weight : weights) {
      weight = norm(gen) * std::sqrt(1/weights.size());
    }
  }
}
