#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Fp8Format {
    E4M3,
    E5M2,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum MasterFormat {
    Fp32,
    Bf16,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum ScalingStrategy {
    Delayed { history_length: usize },
    Dynamic,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct MixedPrecisionConfig {
    pub forward_format: Fp8Format,
    pub backward_format: Fp8Format,
    pub master_format: MasterFormat,
    pub scaling_strategy: ScalingStrategy,
}

impl Default for MixedPrecisionConfig {
    fn default() -> Self {
        Self {
            forward_format: Fp8Format::E4M3,
            backward_format: Fp8Format::E5M2,
            master_format: MasterFormat::Fp32,
            scaling_strategy: ScalingStrategy::Delayed { history_length: 1024 },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    #[test]
    fn default_config_has_e4m3_forward() {
        let cfg = MixedPrecisionConfig::default();
        assert_eq!(cfg.forward_format, Fp8Format::E4M3);
    }

    #[test]
    fn default_config_has_e5m2_backward() {
        let cfg = MixedPrecisionConfig::default();
        assert_eq!(cfg.backward_format, Fp8Format::E5M2);
    }

    #[test]
    fn default_config_has_fp32_master() {
        let cfg = MixedPrecisionConfig::default();
        assert_eq!(cfg.master_format, MasterFormat::Fp32);
    }

    #[test]
    fn default_config_has_delayed_scaling() {
        let cfg = MixedPrecisionConfig::default();
        assert_eq!(
            cfg.scaling_strategy,
            ScalingStrategy::Delayed { history_length: 1024 }
        );
    }

    #[rstest]
    #[case::e4m3(Fp8Format::E4M3)]
    #[case::e5m2(Fp8Format::E5M2)]
    fn fp8_format_clone_eq(#[case] format: Fp8Format) {
        assert_eq!(format, format.clone());
    }

    #[rstest]
    #[case::fp32(MasterFormat::Fp32)]
    #[case::bf16(MasterFormat::Bf16)]
    fn master_format_clone_eq(#[case] format: MasterFormat) {
        assert_eq!(format, format.clone());
    }

    #[test]
    fn custom_config() {
        let cfg = MixedPrecisionConfig {
            forward_format: Fp8Format::E5M2,
            backward_format: Fp8Format::E4M3,
            master_format: MasterFormat::Bf16,
            scaling_strategy: ScalingStrategy::Dynamic,
        };
        assert_eq!(cfg.forward_format, Fp8Format::E5M2);
        assert_eq!(cfg.backward_format, Fp8Format::E4M3);
        assert_eq!(cfg.master_format, MasterFormat::Bf16);
        assert_eq!(cfg.scaling_strategy, ScalingStrategy::Dynamic);
    }
}
