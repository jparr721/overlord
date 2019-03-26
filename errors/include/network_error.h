#ifndef NETWORK_ERROR_H_
#define NETWORK_ERROR_H_

#include <exception>

namespace error {
  class NetworkError : public std::exception {
    const char* file;
    int line;
    const char* func;
    const char* info;

    public:
      NetworkError(
          const char* msg,
          const char* file_,
          int line_,
          const char* func_,
          const char* info_="") :
        std::exception(msg),
        file(file_),
        line(line_),
        func(func_),
        info(info_) {};

      const char* get_file() const{ return file; }
      int get_line() const { return line; }
      const char* get_func() const { return func; }
      const char* get_info() const { return info; }
      const char* NetworkError::what() const throw() { return std::exception::what(); }

  }
} // namespace error

#endif // NETWORK_ERROR_H_
