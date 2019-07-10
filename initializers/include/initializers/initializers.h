#ifndef INITIALIZERS_H_
#define INITIALIZERS_H_

#include <algorithm>
#include <cmath>
#include <eigen3/Eigen/Dense>
#include <functional>
#include <random>
#include <string>
#include <unordered_map>

#include <swiss/strings.h>

namespace cerebrum {
  template<typename WeightType>
  class Initializers {
    public:
      Initializers(
          std::string& initializer,
          WeightType& weights) {
        swiss::to_lower(initializer);

        // Get the initializer by name
        auto initializer_ = functions.at(initializer);
        // Execute on the weights
        initializer_(weights);
      }

      static void Zeros(WeightType& weights) {

      }

      static void GlorotUniform(WeightType& weights) {

      }

      static void GlorotNormal(WeightType& weights) {

      }

      static void HeUniform(WeightType& weights) {

      }

      static void HeNormal(WeightType& weights) {

      }

      // TODO(jparr721) - Docs here
      static void RandomNormal(WeightType& weights) {
        std::default_random_engine gen;
        std::normal_distribution<float> norm(0.0, 1.0);

        for (int i = 0u; i < weights.size(); ++i) {
          weights(i) = norm(gen);
        }
      }

      // TODO(jparr721) - Docs here
      static void RandomUniform(WeightType& weights) {
        std::default_random_engine gen;
        std::uniform_real_distribution<float> norm(0.0, 1.0);

        for (int i = 0u; i < weights.size(); ++i) {
          weights(i) = norm(gen);
        }
      }

    private:
      const std::unordered_map<
        std::string,
        std::function<void(WeightType& weights)>> functions {
          { "glorot_uniform", GlorotUniform },
          { "glorot_normal", GlorotNormal },
          { "he_uniform", HeUniform },
          { "he_normal", HeNormal },
          { "random_uniform", RandomUniform },
          { "random_normal", RandomNormal },
          { "zeros", Zeros },
        };
  };
} // namespace cerebrum

#endif // INITIALIZERS_H_
