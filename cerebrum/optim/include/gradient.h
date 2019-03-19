#ifndef GRADIENT_H_
#define GRADIENT_H_

struct Gradient {
  float grad;
  float old_grad;

  Gradient() {
    grad = 0;
    old_grad = 0;
  }
}

#endif // GRADIENT_H_
