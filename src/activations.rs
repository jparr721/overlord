use std::f64;

enum ActivationKind {
    ReLu,
    Sigmoid,
    Tanh,
}

fn ReLu(z: f64) -> f64 {
    (0.0_f64).max(z)
}

fn Sigmoid(z: f64) -> f64 {
    (1.0 / 1.0 + -z.exp())
}

fn Tanh(z: f64) -> f64 {
    (1.0 - z.tanh().powf(2.0))
}

pub fn make_activation(activation: Option<ActivationKind>) -> Option<fn(f64) -> f64> {
    match ActivationKind {
        ActivationKind::ReLu => Some(ReLu),
        ActivationKind::Sigmoid => Some(Sigmoid),
        ActivationKind::Tanh => Some(Tang),
        _ => None,
    }
}
