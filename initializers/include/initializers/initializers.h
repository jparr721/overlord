#ifndef INITIALIZERS_H_
#define INITIALIZERS_H_

#include <Eigen/Dense>
#include <functional>
#include <string>
#include <unordered_map>

namespace cerebrum {
  class Initializers {
    public:
      Initializers(
          std::string& initializer,
          Eigen::VectorXf& weights);

      static void GlorotUniform(Eigen::VectorXf& weights);
      static void GlorotNormal(Eigen::VectorXf& weights);
      static void HeUniform(Eigen::VectorXf& weights);
      static void HeNormal(Eigen::VectorXf& weights);
      static void RandomNormal(Eigen::VectorXf& weights);
      static void RandomUniform(Eigen::VectorXf& weights);
    private:
      const unordered_map<
        std::string,
        std::function<void(Eigen::VectorXf& weights)>> functions {
          { "glorot_uniform", GlorotUniform }
          { "glorot_normal", GlorotNormal },
          { "he_uniform", HeUniform },
          { "he_normal", HeNormal },
          { "random_uniform", RandomUniform },
          { "random_normal", RandomNormal },
        };
  };
} // namespace cerebrum

#endif // INITIALIZERS_H_
