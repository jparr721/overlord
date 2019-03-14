#ifndef TENSOR_H_
#define TENSOR_H_

#include <array>
#include <Eigen/Dense>
#include <linalg/dim>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace linalg {
  template<typename T>
  class Tensor {
    public:
      T* data;
      dim_t dimension;

      Tensor(int x, int y, int z) {
        data = std::array<T, (x * y * z)>
        dim.x = x;
        dim.y = y;
        dim.z = z;
      }

      Tensor(const Tensor& t) {
        data = std::array<T, (t.x * t.y * t.z)>;
        data = t.data;
        dimension = t.dimension;
      }

      Tensor<T> operator+(const Tensor& t) {
        Tensor<T> clone(*this);
        for (int i = 0; i < dim_product(t); ++i) {
          clone.data[i] += t.data[i];
        }

        return clone;
      }

      Tensor<T> operator-(const Tensor& t) {
        Tensor<T> clone(*this);
        for (int i = 0; i < dim_product(t); ++i) {
          clone.data[i] -= t.data[i];
        }

        return clone;
      }

      T& get(int x, int y, int z) {
        if (x < 0 || y < 0 || z < 0) {
          throw new std::argument_exception("Error input dimensions cannot be < 0");
        }
        if (x > dimension.x || y > dimension.y || z > dimension.z) {
          throw new std::argument_exception("Error, input dimensions cannot be larger than existing dimensions");
        }

        for (int i = 0; i < x; ++i) {
          for (int j = 0; j < y; ++y) {
            for (int k = 0; k < z; ++k) {
              get(i, j, k) = data[k][j][i];
          }
        }
      }

      T& operator()(int x, int y, int z) {
        return get(x, y, z);
      }

      int shape() {
        return dim_product(*this);
      }

      ~Tensor() {
        delete [] T;
      }

    private:
      int dim_product(const Tensor& t) { return t.dimension.x * t.dimension.y * t.dimension.z }
  };

  template<typename T>
  static Tensor<float> vec_to_tensor(std::vector<std::vector<std::vector<T>>> data) {
    // Calculate dimensions
    int z = (int)data.size();
    int y = (int)data[0].size();
    int x = (int)data[0][0].size();

    Tensor<T> t(x, y ,z);
    for (int i = 0; i < x; ++i) {
      for (int j = 0; j < y; ++j) {
        for (int k = 0; k < z; ++k) {
          t(i, j, k) = data[k][j][i];
        }
      }
    }

    return t;
  }
} // namespace linalg


#endif // TENSOR_H_
