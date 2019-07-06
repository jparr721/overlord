#include <algorithm>
#include <swiss/strings.h>

namespace cerebrum { namespace swiss {
  void to_lower(std::string& str) {
    std::transform(str.begin(), str.end(), str.begin(), ::tolower);
  }
} // namespace swiss
} // namespace cerebrum
