#ifndef CRITERION_H_
#define CRITERION_H_

#include <string>
#include <swiss/containers.h>

namespace cerebrum {
  class Criterion {
    public:
      Criterion(
          std::string& name,
          swiss::WeightsXf targets,
          swiss::WeightsXf y);
  };
} // namespace cerebrum

#endif // CRITERION_H_
