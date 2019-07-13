#ifndef REGULARIZERS_H_
#define REGULARIZERS_H_

#include <algorithm>
#include <cassert>
#include <cmath>
#include <random>
#include <eigen3/Eigen/Dense>
#include <functional>
#include <string>
#include <unordered_map>
#include <swiss/strings.h>

////////////////////////////////////////////////////
// Regularizers help us avoid overfitting by penalizing
// the coefficients (or weights in the case of neural
// networks) by lowering their "say" on the outcome.
//
// Regularization can take place in many different ways
// and all versions server their purpose. The functions
// defined here will have documentation in the .cc file
// explaining how/why it works, and when you might want
// to use it.
//
// Note: Regulaization's are tricky. In the event of
// over-regularization, we can run into issues where
// the model will under fit because too many weights
// are dialed back too much.
////////////////////////////////////////////////////

namespace cerebrum {
  template<typename WeightType>
  class Regularizers {
    public:
      Regularizers(std::string& regulaizer, WeightType& weights, float lambda) {
        swiss::to_lower(regularizer);

        // Get the regualarizer by name
        auto regularizer_ = functions.at(regularizer, lambda);

        // Execute it on the weights
        regularizer_(weights);
      }

      static void L1(WeightType& weights, float lambda) {
        for (size_t i = 0; i < weights.size(); ++i) {
          weights[i] = lambda * std::fabs(weights[i]);
        }
      }

      static void L2(WeightType& weights, float lambda) {
        for (size_t i = 0; i < weights.size(); ++i) {
          weights[i] = lambda * std::pow(weights[i], 2);
        }
      }

      static void Dropout(WeightType& weights, float lambda) {
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

    private:
      const std::unordered_map<
        std::string,
        std::function<void(WeightType& weights)>> functions {
          { "l1", L1 },
          { "l2", L2 },
          { "dropout", Dropout },
        };
  };
} // namespace cerebrum

#endif // REGULARIZERS_H_
