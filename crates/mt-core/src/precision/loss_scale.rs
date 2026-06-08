#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LossScaleMode {
    Dynamic,
    Static,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LossScaler {
    pub scale: f32,
    pub mode: LossScaleMode,
    pub growth_interval: usize,
    pub growth_counter: usize,
}

impl LossScaler {
    pub fn new_dynamic(initial_scale: f32, growth_interval: usize) -> Self {
        Self {
            scale: initial_scale,
            mode: LossScaleMode::Dynamic,
            growth_interval,
            growth_counter: 0,
        }
    }

    pub fn new_static(scale: f32) -> Self {
        Self {
            scale,
            mode: LossScaleMode::Static,
            growth_interval: 0,
            growth_counter: 0,
        }
    }

    pub fn update(&mut self, overflow: bool) {
        if self.mode == LossScaleMode::Static {
            return;
        }
        if overflow {
            self.scale /= 2.0;
            if self.scale < 1.0 {
                self.scale = 1.0;
            }
            self.growth_counter = 0;
        } else {
            self.growth_counter += 1;
            if self.growth_counter >= self.growth_interval {
                self.scale *= 2.0;
                self.growth_counter = 0;
            }
        }
    }

    pub fn scale(&self) -> f32 {
        self.scale
    }

    pub fn reset(&mut self, initial_scale: f32) {
        self.scale = initial_scale;
        self.growth_counter = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    #[test]
    fn dynamic_scaler_initial_values() {
        let scaler = LossScaler::new_dynamic(65536.0, 2000);
        assert_eq!(scaler.scale(), 65536.0);
        assert_eq!(scaler.mode, LossScaleMode::Dynamic);
        assert_eq!(scaler.growth_interval, 2000);
        assert_eq!(scaler.growth_counter, 0);
    }

    #[test]
    fn static_scaler_initial_values() {
        let scaler = LossScaler::new_static(1024.0);
        assert_eq!(scaler.scale(), 1024.0);
        assert_eq!(scaler.mode, LossScaleMode::Static);
    }

    #[test]
    fn overflow_halves_scale() {
        let mut scaler = LossScaler::new_dynamic(1024.0, 100);
        scaler.update(true);
        assert!((scaler.scale() - 512.0).abs() < 1e-6);
    }

    #[test]
    fn overflow_resets_growth_counter() {
        let mut scaler = LossScaler::new_dynamic(1024.0, 100);
        for _ in 0..50 {
            scaler.update(false);
        }
        assert_eq!(scaler.growth_counter, 50);
        scaler.update(true);
        assert_eq!(scaler.growth_counter, 0);
    }

    #[test]
    fn growth_doubles_after_interval() {
        let mut scaler = LossScaler::new_dynamic(1024.0, 4);
        scaler.update(false);
        assert!((scaler.scale() - 1024.0).abs() < 1e-6);
        scaler.update(false);
        assert!((scaler.scale() - 1024.0).abs() < 1e-6);
        scaler.update(false);
        assert!((scaler.scale() - 1024.0).abs() < 1e-6);
        scaler.update(false);
        assert!((scaler.scale() - 2048.0).abs() < 1e-6);
        assert_eq!(scaler.growth_counter, 0);
    }

    #[test]
    fn overflow_floor_is_one() {
        let mut scaler = LossScaler::new_dynamic(1.0, 100);
        scaler.update(true);
        assert!((scaler.scale() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn static_mode_ignores_overflow() {
        let mut scaler = LossScaler::new_static(1024.0);
        scaler.update(true);
        assert!((scaler.scale() - 1024.0).abs() < 1e-6);
    }

    #[test]
    fn static_mode_ignores_growth() {
        let mut scaler = LossScaler::new_static(1024.0);
        for _ in 0..100 {
            scaler.update(false);
        }
        assert!((scaler.scale() - 1024.0).abs() < 1e-6);
    }

    #[test]
    fn reset_restores_initial() {
        let mut scaler = LossScaler::new_dynamic(1024.0, 100);
        scaler.update(true);
        scaler.update(true);
        scaler.reset(65536.0);
        assert!((scaler.scale() - 65536.0).abs() < 1e-6);
        assert_eq!(scaler.growth_counter, 0);
    }

    #[rstest]
    #[case::no_overflow_sequence(false, false, false, false, 2048.0)]
    #[case::overflow_then_recovery(true, false, false, false, 1024.0)]
    fn loss_scaling_scenarios(
        #[case] s1: bool,
        #[case] s2: bool,
        #[case] s3: bool,
        #[case] s4: bool,
        #[case] expected: f32,
    ) {
        let mut scaler = LossScaler::new_dynamic(1024.0, 3);
        scaler.update(s1);
        scaler.update(s2);
        scaler.update(s3);
        scaler.update(s4);
        assert!((scaler.scale() - expected).abs() < 1e-6);
    }

    #[test]
    fn overflow_behavior_sequence() {
        let mut scaler = LossScaler::new_dynamic(65536.0, 2);
        scaler.update(true);
        assert!((scaler.scale() - 32768.0).abs() < 1e-6);
        scaler.update(false);
        assert!((scaler.scale() - 32768.0).abs() < 1e-6);
        scaler.update(false);
        assert!((scaler.scale() - 65536.0).abs() < 1e-6);
    }

    #[test]
    fn clone_equality() {
        let scaler = LossScaler::new_dynamic(1024.0, 100);
        assert_eq!(scaler, scaler.clone());
    }
}
