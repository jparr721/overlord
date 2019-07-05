#ifndef REGULARIZERS_H_
#define REGULARIZERS_H_

#include <eigen3/Eigen/Dense>
#include <functional>
#include <string>
#include <unordered_map>

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
  class Regularizers {
    public:
      Regularizers(std::string& regulaizer, Eigen::VectorXf& weights);

      static void L1(Eigen::VectorXf& weights);
      static void L2(Eigen::VectorXf& weights);
      static void Dropout(Eigen::VectorXf& weights);

    private:
      const std::unordered_map<
        std::string,
        std::function<void(Eigen::VectorXf& weights)>> functions {
          { "l1", L1 },
          { "l2", L2 },
          { "dropout", Dropout },
        };
  };
} // namespace cerebrum

#endif // REGULARIZERS_H_
