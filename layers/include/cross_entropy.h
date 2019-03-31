#ifndef BRINARY_CROSS_ENTROPY_H_
#define BINARY_CROSS_ENTROPY_H_

#include <armadillo>
#include <cassert>

namespace layer {
  class CrossEntropy {
    public:
      arma::vec gradient_predicted_distribution;

      CrossEntropy(size_t num_inputs) : num_inputs(num_inputs) {}

      double forward(arma::vec& predicted_distribution, arma::vec& actual_distribution) {
        assert(predicted_distribution.n_elem == num_inputs);
        assert(actual_distribution.n_elem == num_inputs);

        // Cache the predicted and actual labels for the backward pass
        this->predicted_distribution = predicted_distribution;
        this->actual_distribution = actual_distribution;

        // Compute loss and cache it into the class
        loss = -arma::dot(actual_distribution,
            arma::log(predicted_distribution));

        return loss;
      }

      void backward() {
        gradient_predicted_distribution = -(actual_distribution % (1/predicted_distribution));
      }
    private:
      size_t num_inputs;
      arma::vec predicted_distribution;
      arma::vec actual_distribution;

      double loss;
  };
} // namespace layer

#endif // BINARY_CROSS_ENTROPY_H_
