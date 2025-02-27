#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]

mod bindings;
pub use bindings::*;

#[cfg(osqp_dlong)]
pub type osqp_int = ::std::os::raw::c_longlong;
#[cfg(not(osqp_dlong))]
pub type osqp_int = ::std::os::raw::c_int;
pub type osqp_float = f64;

type OSQPInt = osqp_int;
type OSQPFloat = osqp_float;

pub enum OSQPTimer {}

#[cfg(test)]
mod tests {
    use std::ptr;
    use super::*;

    // examples/osqp_simple_demo.c converted into rust
    #[test]
    fn osqp_simple_demo_rust() {
        unsafe {
            osqp_simple_demo_rust_unsafe();
        }
    }

    unsafe fn osqp_simple_demo_rust_unsafe() {
        // Load problem data
        let mut P_x: [OSQPFloat; 3] = [4.0, 1.0, 2.0];
        let P_nnz: OSQPInt = 3;
        let mut P_i: [OSQPInt; 3] = [0, 0, 1];
        let mut P_p: [OSQPInt; 3] = [0, 1, 3];
        let q: [OSQPFloat; 2] = [1.0, 1.0];
        let mut A_x: [OSQPFloat; 4] = [1.0, 1.0, 1.0, 1.0];
        let A_nnz: OSQPInt = 4;
        let mut A_i: [OSQPInt; 4] = [0, 1, 0, 2];
        let mut A_p: [OSQPInt; 3] = [0, 2, 4];
        let l: [OSQPFloat; 3] = [1.0, 0.0, 0.0];
        let u: [OSQPFloat; 3] = [1.0, 0.7, 0.7];
        let n: OSQPInt = 2;
        let m: OSQPInt = 3;

        // Populate data
        let P = OSQPCscMatrix_new(
            n,
            n,
            P_nnz,
            P_x.as_mut_ptr(),
            P_i.as_mut_ptr(),
            P_p.as_mut_ptr(),
        );
        let A = OSQPCscMatrix_new(
            m,
            n,
            A_nnz,
            A_x.as_mut_ptr(),
            A_i.as_mut_ptr(),
            A_p.as_mut_ptr(),
        );

        // Define solver settings
        let settings: *mut OSQPSettings = OSQPSettings_new();
        osqp_set_default_settings(settings);
        (*settings).polishing = 1;

        // set up a solver null pointer that we can pass to osqp_setup
        let mut solver: *mut OSQPSolver = ptr::null_mut();
        let status = osqp_setup(&mut solver, P, q.as_ptr(), A, l.as_ptr(), u.as_ptr(), m, n, settings);
        if status != 0 {
            panic!("osqp_setup failed");
        }

        // Solve problem
        osqp_solve(solver);

        // Check the results
        let eps = 1e-9;
        let x = (*(*solver).solution).x;
        let x0 = *x;
        let x1 = *(x.offset(1));
        println!("[{}, {}]", x0, x1);
        assert!((0.3 - x0).abs() < eps);
        assert!((0.7 - x1).abs() < eps);

        // Cleanup
        osqp_cleanup(solver);
        OSQPSettings_free(settings);
        OSQPCscMatrix_free(P);
        OSQPCscMatrix_free(A);
    }
}
