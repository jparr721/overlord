#pragma once

#include <array>
#include <eigen3/Eigen/Dense>
#include <random>
#include <vector>

namespace overlord {
  class Initializer {
    public:
      Initializer(Eigen::MatrixXf weights) : weights_(weights) {}
      virtual ~Initializer();
      virtual void Init() = 0;
    protected:
      const Eigen::MatrixXf weights_;
  };

  class Zeros : private Initializer {
    void Init() {
      for (std::size_t i = 0u; i < weights_.size(); ++i) {
        weights_(i) = 0.0;
      }
    }
  };

  class Ones : private Initializer {
    void Init() {
      for (std::size_t i = 0u; i < weights_.size(); ++i) {
        weights_(i) = 1.0;
      }
    }
  };

  class N : private Initializer {
    void Init(double n) {
      for (std::size_t i = 0u; i < weights_.size(); ++i) {
        weights_(i) = 1.0;
      }
    }
  };

  class RandomUniform : private Initializer {
    void Init(double min=0.0, double max=1.0) {
      std::default_random_engine gen;
      std::uniform_real_distribution<double> norm(min, max);

      for (std::size_t i = 0; i < weights_.size(); ++i) {
        weights_(i) = norm(gen);
      }
    }
  };

  class GlorotUniform : private Initializer {
    void Init(double min=0.0, double max=1.0) {
      std::default_random_engine gen;
      std::uniform_real_distribution<double> norm(min, max);

      for (std::size_t i = 0; i < weights_.size(); ++i) {
        weights_(i) = norm(gen) * std::sqrt(1/weights._size());
      }
    }
  };

  class GlorotNormal : private Initializer {
    void Init(double min=0.0, double max=1.0) {
      std::default_random_engine gen;
      std::normal_distribution<double> norm(min, max);

      for (std::size_t i = 0; i < weights_.size(); ++i) {
        weights_(i) = norm(gen) * std::sqrt(1/weights._size());
      }
    }
  };

  class HeUniform : private Initializer {
    void Init(double min=0.0, double max=1.0) {
      std::default_random_engine gen;
      std::uniform_real_distribution<double> norm(min, max);

      for (std::size_t i = 0; i < weights_.size(); ++i) {
        weights_(i) = norm(gen) * std::sqrt(2/weights._size());
      }
    }
  };

  class HeNormal : private Initializer {
    void Init(double min=0.0, double max=1.0) {
      std::default_random_engine gen;
      std::normal_distribution<double> norm(min, max);

      for (std::size_t i = 0; i < weights_.size(); ++i) {
        weights_(i) = norm(gen) * std::sqrt(2/weights._size());
      }
    }
  };
} // namespace overlord
