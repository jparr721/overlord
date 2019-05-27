#ifndef VARIANCE_SCALING_H_
#define VARIANCE_SCALING_H_

#include <Eigen/Dense>
#include <string>

namespace cerebrum {
  class VarianceScaling {
    public:
      VarianceScaling(
          const double scale=1.0,
          const std::string& mode="fan_in",
          const std::string& distribution="normal",
          const int seed=nullptr) :
        scale_(scale), mode_(mode), distribution_(distribution), seed_(seed) {};
    private:
      double scale_;
      std::string mode_;
      std::string distribution_;
      int seed_;
  };
} // namespace cerebrum
