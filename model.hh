#pragma once

#include <vector>

namespace overlord {
  template<typename LossFunc, class... Args>
  class SequentialModel {
    public:
      SequentialModel(const Args&... args) : layers_(args...) {
        // Initialize all layers
        for (const auto& layer : layers_) {
          layer.Init(true);
        }
      }

      void Train(LossFunc loss_function) {

      }

    private:
      const std::vector<Args...> layers_;
  };
} // namespace overlord
