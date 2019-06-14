use std::f64

pub fn ReLu(z: f64) -> f64 {
    let zero: f64 = 0.0
    zero.max(z)
}

pub fn Sigmoid(z: f64) {
    (1.0 / 1.0 + -z.exp())
}

pub fn Tanh(z: f64) {
    (1.0 - z.tanh().powf(2.0))
}
