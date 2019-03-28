#ifndef RUNTIME_H_
#define RUNTIME_H_

#include <armadillo>
#include <vector>
#include "../../layers/include/base.h"

namespace engine {
  /// The runtime class establishes a running environment for a
  /// neural network
  class Runtime {
    public:
      Runtime();
      Runtime(const layer::Base& layer) {
        layers.push_back(layer);
      };

      void add(const layer::Base& layer) {
        layers.push_back(layer);
      }

      void fit(const arma::vec& inputs, const std::string optimizer);
    private:
      std::vector<layer::Base> layers;
  };
} // namespace engine


#endif // RUNTIME_H_
