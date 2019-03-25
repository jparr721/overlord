#ifndef RUNTIME_H_
#define RUNTIME_H_

namespace engine {
  /// The runtime class establishes a running environment for a
  /// neural network
  template <class NetworkLayerType>
  class Runtime {
    public:
      Runtime(NetworkLayerType network) : local_net_(network) {};
    private:
      NetworkLayerType local_net_;
  };
} // namespace engine


#endif // RUNTIME_H_
