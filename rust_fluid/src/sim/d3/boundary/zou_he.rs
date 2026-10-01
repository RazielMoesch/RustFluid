use super::Boundary3D;
use crate::sim::d3::boundary::fluid::Fluid;

pub struct ZouHeLeftVelocity;

impl Boundary3D for ZouHeLeftVelocity {
    fn type_id(&self) -> u32 {
        5
    }

    fn name(&self) -> &'static str {
        "Zou-He Left Velocity"
    }

    fn pull_even(&self) -> Option<&'static str> {
        Fluid.pull_even()
    }

    fn pull_odd(&self) -> Option<&'static str> {
        Fluid.pull_odd()
    }

    fn post_streaming(&self) -> Option<&'static str> {
        Some(r#"
    let my_type = my_flag >> FLAG_TYPE_SHIFT;
    
    if (my_type == 5u) {
        let cfg_id = my_flag & FLAG_ID_MASK;
        let cfg = boundary_configs[cfg_id];
        let ux = cfg.vel.x;
        let uy = cfg.vel.y;
        let uz = cfg.vel.z;
        
        let rho_in = (f_local[0] + f_local[3] + f_local[4] + f_local[5] + f_local[6] + 
                      f_local[15] + f_local[16] + f_local[17] + f_local[18] + 
                      2.0 * (f_local[2] + f_local[8] + f_local[10] + f_local[12] + f_local[14])) / (1.0 - ux);
        
        f_local[1] = f_local[2] + (1.0 / 3.0) * rho_in * ux;
        
        let diff_y = 0.5 * (f_local[4] - f_local[3]);
        let diff_z = 0.5 * (f_local[6] - f_local[5]);
        
        f_local[7]  = f_local[10] + diff_y + (1.0 / 6.0) * rho_in * ux + 0.5 * rho_in * uy;
        f_local[9]  = f_local[8]  - diff_y + (1.0 / 6.0) * rho_in * ux - 0.5 * rho_in * uy;
        f_local[11] = f_local[14] + diff_z + (1.0 / 6.0) * rho_in * ux + 0.5 * rho_in * uz;
        f_local[13] = f_local[12] - diff_z + (1.0 / 6.0) * rho_in * ux - 0.5 * rho_in * uz;
        
        rho = rho_in;
        u = cfg.vel * rho_in;
    }
"#)
    }
}
