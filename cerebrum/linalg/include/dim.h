#ifndef DIM_H_
#define DIM_H_

/// Dim keeps track of the dimensions of a given structure
/// this is a drop-in replacement for the Eigen matrix dimension
/// which is internal to the class instance. This would be used
/// when nothing else is available.
struct dim_t {
  int x;
  int y;
  int z;
};

#endif // DIM_H_
