#ifndef REGULARIZERS_H_
#define REGULARIZERS_H_

#include <eigen3/Eigen/Dense>
#include <string>

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
    Regularizers(std::string& regulaizer, Eigen::VectorXf& weights);

    void L1(Eigen::VectorXf& weights);
    void L2(Eigen::VectorXf& weights);
    void Dropout(Eigen::VectorXf& weights);
  };
} // namespace cerebrum

#endif // REGULARIZERS_H_
