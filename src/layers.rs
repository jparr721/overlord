use std::f64
use ndarray::{Array1, Array2, Array3};

use crate::activations::*;

// TODO(jparr721) - Add initializer options
#[derive(Debug)]
pub struct Dense {
    weights: Array2,
    biases: Array1,
    input_shape: (u16, u16),
    output_shape: (u16, u16),
    bias: bool,
}

#[derive(Debug)]
pub struct Convolutional {
    weights: Array2,
    biases: Array1,
    input_shape: (u16, u16),
    output_shape: u16,
    bias: bool,
}

trait Engine<VectorType> {
    fn new(in_shape: (u16, u16), out_shape: u16, bias: bool) -> Self;

    fn init(&self, ActivationKind);
    fn forward(&self, input: VectorType) -> f64;
    fn backward(&self, gradient: f64) -> f64;
}

impl Engine<Array2> for Dense {
    fn new(in_shape: (u16, u16), out_shape: u16, bias: bool) -> Self {
        Dense {
            weights: Array2::<f64>::zeros(in_shape),
            biases: Array1::<f64>::zeros(in_shape.0),
            input_shape: in_shape,
            output_shape: out_shape,
            bias: bias
        }
    }

    fn init(ActivationKind) {
        let activation_fn = make_activation(ActivationKind);

        self.weights.map(|x| activation_fn(x));
    }
}
