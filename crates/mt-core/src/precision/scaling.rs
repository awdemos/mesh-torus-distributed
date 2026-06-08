use float8::{F8E4M3, F8E5M2};

use super::config::Fp8Format;

#[derive(Debug, Clone, PartialEq)]
pub struct DelayedScaling {
    pub amax_history: Vec<f32>,
    pub history_length: usize,
    cursor: usize,
}

impl DelayedScaling {
    pub fn new(history_length: usize) -> Self {
        Self {
            amax_history: vec![0.0; history_length],
            history_length,
            cursor: 0,
        }
    }

    pub fn update(&mut self, amax: f32) {
        self.amax_history[self.cursor] = amax;
        self.cursor = (self.cursor + 1) % self.history_length;
    }

    pub fn current_amax(&self) -> f32 {
        self.amax_history
            .iter()
            .copied()
            .fold(0.0f32, f32::max)
    }

    pub fn compute_scale(&self, format: Fp8Format) -> f32 {
        let amax = self.current_amax();
        if amax == 0.0 {
            return 1.0;
        }
        let fp8_max = fp8_format_max(format);
        fp8_max / amax
    }
}

pub(crate) fn fp8_format_max(format: Fp8Format) -> f32 {
    match format {
        Fp8Format::E4M3 => F8E4M3::MAX.to_f32(),
        Fp8Format::E5M2 => F8E5M2::MAX.to_f32(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    #[test]
    fn new_initializes_zero_history() {
        let ds = DelayedScaling::new(4);
        assert_eq!(ds.amax_history, vec![0.0, 0.0, 0.0, 0.0]);
        assert_eq!(ds.current_amax(), 0.0);
    }

    #[test]
    fn update_writes_to_ring_buffer() {
        let mut ds = DelayedScaling::new(3);
        ds.update(1.0);
        ds.update(2.0);
        ds.update(3.0);
        assert_eq!(ds.amax_history, vec![1.0, 2.0, 3.0]);
        assert_eq!(ds.current_amax(), 3.0);
    }

    #[test]
    fn ring_buffer_wraps_around() {
        let mut ds = DelayedScaling::new(3);
        ds.update(1.0);
        ds.update(2.0);
        ds.update(3.0);
        ds.update(4.0);
        assert_eq!(ds.amax_history, vec![4.0, 2.0, 3.0]);
        assert_eq!(ds.current_amax(), 4.0);
    }

    #[test]
    fn ring_buffer_full_overwrite() {
        let mut ds = DelayedScaling::new(2);
        ds.update(10.0);
        ds.update(5.0);
        ds.update(1.0);
        ds.update(2.0);
        assert_eq!(ds.amax_history, vec![1.0, 2.0]);
        assert_eq!(ds.current_amax(), 2.0);
    }

    #[rstest]
    #[case::e4m3(Fp8Format::E4M3, 416.0)]
    #[case::e5m2(Fp8Format::E5M2, 49152.0)]
    fn compute_scale_zero_amax_returns_one(#[case] format: Fp8Format, #[case] _max: f32) {
        let ds = DelayedScaling::new(4);
        assert_eq!(ds.compute_scale(format), 1.0);
    }

    #[rstest]
    #[case::e4m3(Fp8Format::E4M3, 2.0, 416.0 / 2.0)]
    #[case::e5m2(Fp8Format::E5M2, 2.0, 49152.0 / 2.0)]
    fn compute_scale_with_amax(
        #[case] format: Fp8Format,
        #[case] amax: f32,
        #[case] expected: f32,
    ) {
        let mut ds = DelayedScaling::new(4);
        ds.update(amax);
        let scale = ds.compute_scale(format);
        assert!((scale - expected).abs() < 1e-3);
    }

    #[test]
    fn scale_uses_max_amax_from_history() {
        let mut ds = DelayedScaling::new(4);
        ds.update(1.0);
        ds.update(5.0);
        ds.update(3.0);
        let scale = ds.compute_scale(Fp8Format::E4M3);
        let expected = 416.0 / 5.0;
        assert!((scale - expected).abs() < 1e-3);
    }

    #[test]
    fn delayed_scaling_convergence() {
        let mut ds = DelayedScaling::new(4);
        let amax_values = [1.0, 2.0, 4.0, 8.0, 4.0, 2.0, 1.0, 0.5];
        let mut scales = Vec::new();
        for amax in amax_values {
            ds.update(amax);
            scales.push(ds.compute_scale(Fp8Format::E4M3));
        }
        assert!(scales[3] < scales[0], "scale should increase as amax grows");
        assert!(scales[7] > scales[3], "scale should increase as amax shrinks");
    }

    #[test]
    fn clone_equality() {
        let mut ds = DelayedScaling::new(4);
        ds.update(1.0);
        ds.update(2.0);
        assert_eq!(ds, ds.clone());
    }
}
