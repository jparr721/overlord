#ifndef INITIALIZERS_H_
#define INITIALIZERS_H_

#include <eigen3/Eigen/Dense>
#include <functional>
#include <string>
#include <unordered_map>

namespace cerebrum {
  template<typename WeightType>
  class Initializers {
    public:
      Initializers(
          std::string& initializer,
          WeightType& weights);

      static void Zeros(WeightType& weights);
      static void GlorotUniform(WeightType& weights);
      static void GlorotNormal(WeightType& weights);
      static void HeUniform(WeightType& weights);
      static void HeNormal(WeightType& weights);
      static void RandomNormal(WeightType& weights);
      static void RandomUniform(WeightType& weights);
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
