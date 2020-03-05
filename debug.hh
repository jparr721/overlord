#pragma once

#include <iostream>

namespace overlord::debug {
  template <class... Args>
  void DbgPrint(Args... args) {
    (std::cout << ... << args) << "\n";
  }
} // namespace overlord::debug
