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

      // TODO(jparr721) - Docs here
      static void GlorotUniform(Eigen::VectorXf& weights);

      // TODO(jparr721) - Docs here
      static void GlorotNormal(Eigen::VectorXf& weights);

      // TODO(jparr721) - Docs here
      static void HeUniform(Eigen::VectorXf& weights);

      // TODO(jparr721) - Docs here
      static void HeNormal(Eigen::VectorXf& weights);
    private:
      const unordered_map<
        std::string,
        std::function<void(Eigen::VectorXf& weights)>> functions {
          { "glorot_uniform", GlorotUniform }
          { "glorot_normal", GlorotNormal },
          { "he_uniform", HeUniform },
          { "he_normal", HeNormal },
        };
  };
} // namespace cerebrum

#endif // INITIALIZERS_H_
