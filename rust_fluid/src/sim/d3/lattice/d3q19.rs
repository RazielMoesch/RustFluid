//! Standard D3Q19 directions, weights, opposites, and reflections.

use super::Lattice3D;

/// Nineteen-velocity cubic lattice used by `Lbm3D`.
pub struct D3Q19;

impl D3Q19 {
    pub fn new() -> Self {
        Self
    }
}

impl Lattice3D for D3Q19 {
    fn name(&self) -> &'static str {
        "D3Q19"
    }

    fn q(&self) -> u32 {
        19
    }

    fn ex(&self) -> &'static str {
        r#"
const EX = array<i32, 19>(
    0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0
);
"#
    }

    fn ey(&self) -> &'static str {
        r#"
const EY = array<i32, 19>(
    0, 0, 0, 1, -1, 0, 0, 1, 1, -1, -1, 0, 0, 0, 0, 1, -1, 1, -1
);
"#
    }

    fn ez(&self) -> &'static str {
        r#"
const EZ = array<i32, 19>(
    0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, 1, -1, -1, 1, 1, -1, -1
);
"#
    }

    fn weights(&self) -> &'static str {
        r#"
const WEIGHTS = array<f32, 19>(
    1.0 / 3.0,
    1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0,
    1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0,
    1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0,
    1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0
);
"#
    }

    fn opp(&self) -> &'static str {
        r#"
const OPP = array<u32, 19>(
    0u, 2u, 1u, 4u, 3u, 6u, 5u, 10u, 9u, 8u, 7u, 14u, 13u, 12u, 11u, 18u, 17u, 16u, 15u
);
"#
    }

    fn reflect_x(&self) -> Option<&'static str> {
        Some(
            r#"
const REFLECT_X = array<u32, 19>(
    0u, 2u, 1u, 3u, 4u, 5u, 6u, 8u, 7u, 10u, 9u, 12u, 11u, 14u, 13u, 15u, 16u, 17u, 18u
);
"#,
        )
    }

    fn reflect_y(&self) -> Option<&'static str> {
        Some(
            r#"
const REFLECT_Y = array<u32, 19>(
    0u, 1u, 2u, 4u, 3u, 5u, 6u, 9u, 10u, 7u, 8u, 11u, 12u, 13u, 14u, 16u, 15u, 18u, 17u
);
"#,
        )
    }

    fn reflect_z(&self) -> Option<&'static str> {
        Some(
            r#"
const REFLECT_Z = array<u32, 19>(
    0u, 1u, 2u, 3u, 4u, 6u, 5u, 7u, 8u, 9u, 10u, 13u, 14u, 11u, 12u, 17u, 18u, 15u, 16u
);
"#,
        )
    }

    fn ex_array(&self) -> &[i32] {
        &[0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0]
    }

    fn ey_array(&self) -> &[i32] {
        &[0, 0, 0, 1, -1, 0, 0, 1, 1, -1, -1, 0, 0, 0, 0, 1, -1, 1, -1]
    }

    fn ez_array(&self) -> &[i32] {
        &[0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, 1, -1, -1, 1, 1, -1, -1]
    }

    fn opp_array(&self) -> &[u32] {
        &[
            0, 2, 1, 4, 3, 6, 5, 10, 9, 8, 7, 14, 13, 12, 11, 18, 17, 16, 15,
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn d3q19_opposites_weights_and_moments_are_symmetric() {
        let lattice = D3Q19::new();
        let ex = lattice.ex_array();
        let ey = lattice.ey_array();
        let ez = lattice.ez_array();
        let opp = lattice.opp_array();
        let weights = [
            1.0 / 3.0,
            1.0 / 18.0,
            1.0 / 18.0,
            1.0 / 18.0,
            1.0 / 18.0,
            1.0 / 18.0,
            1.0 / 18.0,
            1.0 / 36.0,
            1.0 / 36.0,
            1.0 / 36.0,
            1.0 / 36.0,
            1.0 / 36.0,
            1.0 / 36.0,
            1.0 / 36.0,
            1.0 / 36.0,
            1.0 / 36.0,
            1.0 / 36.0,
            1.0 / 36.0,
            1.0 / 36.0,
        ];

        for i in 0..19 {
            let j = opp[i] as usize;
            assert_eq!((ex[j], ey[j], ez[j]), (-ex[i], -ey[i], -ez[i]));
            assert_eq!(opp[j] as usize, i);
            assert_eq!(weights[j], weights[i]);
        }

        let mut zeroth = 0.0f64;
        let mut first = [0.0f64; 3];
        let mut second = [[0.0f64; 3]; 3];
        for i in 0..19 {
            let c = [ex[i] as f64, ey[i] as f64, ez[i] as f64];
            zeroth += weights[i];
            for a in 0..3 {
                first[a] += weights[i] * c[a];
                for b in 0..3 {
                    second[a][b] += weights[i] * c[a] * c[b];
                }
            }
        }
        assert!((zeroth - 1.0).abs() < 1e-15);
        assert_eq!(first, [0.0; 3]);
        for a in 0..3 {
            for b in 0..3 {
                let expected = if a == b { 1.0 / 3.0 } else { 0.0 };
                assert!((second[a][b] - expected).abs() < 1e-15);
            }
        }
    }
}
