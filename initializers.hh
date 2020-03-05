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
      virtual void Init(Eigen::MatrixXf weights) = 0;
    protected:
      Eigen::MatrixXf weights_;
  };

  class Zeros : private Initializer {
    void Init(Eigen::MatrixXf weights) {
      for (std::size_t i = 0u; i < weights_.size(); ++i) {
        weights_(i) = 0.0;
      }
    }
  };

  class Ones : private Initializer {
    void Init(Eigen::MatrixXf weights) {
      for (std::size_t i = 0u; i < weights_.size(); ++i) {
        weights_(i) = 1.0;
      }
    }
  };

  class N : private Initializer {
    void Init(Eigen::MatrixXf weights, double n) {
      for (std::size_t i = 0u; i < weights_.size(); ++i) {
        weights_(i) = 1.0;
      }
    }
  };

  class RandomUniform : private Initializer {
    void Init(Eigen::MatrixXf weights, double min=0.0, double max=1.0) {
      std::default_random_engine gen;
      std::uniform_real_distribution<double> norm(min, max);

      for (std::size_t i = 0; i < weights_.size(); ++i) {
        weights_(i) = norm(gen);
      }
    }
  };

  class GlorotUniform : private Initializer {
    void Init(Eigen::MatrixXf weights, double min=0.0, double max=1.0) {
      std::default_random_engine gen;
      std::uniform_real_distribution<double> norm(min, max);

      for (std::size_t i = 0; i < weights_.size(); ++i) {
        weights_(i) = norm(gen) * std::sqrt(1/weights_.size());
      }
    }
  };

  class GlorotNormal : private Initializer {
    void Init(Eigen::MatrixXf weights, double min=0.0, double max=1.0) {
      std::default_random_engine gen;
      std::normal_distribution<double> norm(min, max);

      for (std::size_t i = 0; i < weights_.size(); ++i) {
        weights_(i) = norm(gen) * std::sqrt(1/weights_.size());
      }
    }
  };

  class HeUniform : private Initializer {
    void Init(Eigen::MatrixXf weights, double min=0.0, double max=1.0) {
      std::default_random_engine gen;
      std::uniform_real_distribution<double> norm(min, max);

      for (std::size_t i = 0; i < weights_.size(); ++i) {
        weights_(i) = norm(gen) * std::sqrt(2/weights_.size());
      }
    }
  };

  class HeNormal : private Initializer {
    void Init(Eigen::MatrixXf weights, double min=0.0, double max=1.0) {
      std::default_random_engine gen;
      std::normal_distribution<double> norm(min, max);

      for (std::size_t i = 0; i < weights_.size(); ++i) {
        weights_(i) = norm(gen) * std::sqrt(2/weights_.size());
      }
    }
  };
} // namespace overlord
