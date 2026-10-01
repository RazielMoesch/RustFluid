use super::Lattice2D;

pub struct D2Q9;

impl D2Q9 {
    pub fn new() -> Self {
        Self
    }
}

impl Lattice2D for D2Q9 {
    fn name(&self) -> &'static str {
        "D2Q9"
    }

    fn q(&self) -> u32 {
        9
    }

    fn ex(&self) -> &'static str {
        r#"
const EX = array<i32, 9>(
    0, 1, 0, -1, 0, 1, -1, -1, 1
);
"#
    }

    fn ey(&self) -> &'static str {
        r#"
const EY = array<i32, 9>(
    0, 0, 1, 0, -1, 1, 1, -1, -1
);
"#
    }

    fn weights(&self) -> &'static str {
        r#"
const WEIGHTS = array<f32, 9>(
    4.0 / 9.0,
    1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0,
    1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0
);
"#
    }

    fn opp(&self) -> &'static str {
        r#"
const OPP = array<u32, 9>(
    0u, 3u, 4u, 1u, 2u, 7u, 8u, 5u, 6u
);
"#
    }

    fn reflect_x(&self) -> Option<&'static str> {
        Some(r#"
const REFLECT_X = array<u32, 9>(
    0u, 3u, 2u, 1u, 4u, 6u, 5u, 8u, 7u
);
"#)
    }

    fn reflect_y(&self) -> Option<&'static str> {
        Some(r#"
const REFLECT_Y = array<u32, 9>(
    0u, 1u, 4u, 3u, 2u, 8u, 7u, 6u, 5u
);
"#)
    }

    fn ex_array(&self) -> &[i32] {
        &[0, 1, 0, -1, 0, 1, -1, -1, 1]
    }

    fn ey_array(&self) -> &[i32] {
        &[0, 0, 1, 0, -1, 1, 1, -1, -1]
    }

    fn opp_array(&self) -> &[u32] {
        &[0, 3, 4, 1, 2, 7, 8, 5, 6]
    }
}
