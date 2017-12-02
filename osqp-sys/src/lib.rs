#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]

mod bindings;
pub use bindings::*;

#[cfg(feature = "osqp_dlong")]
pub type osqp_int = ::std::os::raw::c_longlong;
#[cfg(not(feature = "osqp_dlong"))]
pub type osqp_int = ::std::os::raw::c_int;
pub type osqp_float = f64;

type c_int = osqp_int;
type c_float = osqp_float;

pub enum OSQPTimer { }

#[cfg(test)]
mod tests {
    use std::mem;
    use super::*;

    extern "C" {
        fn free(ptr: *mut ());
        fn csc_matrix(
            m: c_int,
            n: c_int,
            nzmax: c_int,
            x: *mut c_float,
            i: *mut c_int,
            p: *mut c_int,
        ) -> *mut csc;
    }

    // examples/osqp_demo.c converted into rust
    #[test]
    fn osqp_demo_rust() {
        unsafe {
            osqp_demo_rust_unsafe();
        }
    }

    unsafe fn osqp_demo_rust_unsafe() {
        // Load problem data
        let mut P_x: [c_float; 4] = [4.0, 1.0, 1.0, 2.0];
        let P_nnz: c_int = 4;
        let mut P_i: [c_int; 4] = [0, 1, 0, 1];
        let mut P_p: [c_int; 3] = [0, 2, 4];
        let mut q: [c_float; 2] = [1.0, 1.0];
        let mut A_x: [c_float; 4] = [1.0, 1.0, 1.0, 1.0];
        let A_nnz: c_int = 4;
        let mut A_i: [c_int; 4] = [0, 1, 0, 2];
        let mut A_p: [c_int; 3] = [0, 2, 4];
        let mut l: [c_float; 3] = [1.0, 0.0, 0.0];
        let mut u: [c_float; 3] = [1.0, 0.7, 0.7];
        let n: c_int = 2;
        let m: c_int = 3;

        // Populate data
        let P = csc_matrix(
            n,
            n,
            P_nnz,
            P_x.as_mut_ptr(),
            P_i.as_mut_ptr(),
            P_p.as_mut_ptr(),
        );
        let A = csc_matrix(
            m,
            n,
            A_nnz,
            A_x.as_mut_ptr(),
            A_i.as_mut_ptr(),
            A_p.as_mut_ptr(),
        );

        let mut data: OSQPData = mem::zeroed();
        data.n = n;
        data.m = m;
        data.P = P;
        data.q = q.as_mut_ptr();
        data.A = A;
        data.l = l.as_mut_ptr();
        data.u = u.as_mut_ptr();
        let data = &data as *const OSQPData;

        // Define solver settings
        let mut settings: OSQPSettings = mem::zeroed();
        set_default_settings(&mut settings);
        settings.alpha = 1.0;
        settings.adaptive_rho = 0;
        let settings = &mut settings as *mut OSQPSettings;

        // Setup workspace
        let work: *mut OSQPWorkspace = osqp_setup(data, settings);

        // Zero data and settings on the stack to ensure osqp does not reference them
        *(data as *mut OSQPData) = mem::zeroed();
        *settings = mem::zeroed();
        *P = mem::zeroed();
        *A = mem::zeroed();

        // Solve problem
        osqp_solve(work);

        // Check the results
        let eps = 1e-9;
        let x = (*(*work).solution).x;
        let x0 = *x;
        let x1 = *(x.offset(1));
        println!("[{}, {}]", x0, x1);
        assert!((0.2987710845986426 - x0).abs() < eps);
        assert!((0.701227995544065 - x1).abs() < eps);

        // Clean workspace
        osqp_cleanup(work);
        free(A as *mut ());
        free(P as *mut ());
    }
}
