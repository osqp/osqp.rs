//! # Example
//!
//! Consider the following QP
//!
//! <div style="text-align:center">
//! <img src="example_qp.svg" alt="Example QP" />
//! </div>
//!
//! ```rust
//! use osqp::{Settings, Workspace};
//!
//! // Define problem data
//! let P = &[[4.0, 1.0],
//!           [1.0, 2.0]];
//! let q = &[1.0, 1.0];
//! let A = &[[1.0, 1.0],
//!           [1.0, 0.0],
//!           [0.0, 1.0]];
//! let l = &[1.0, 0.0, 0.0];
//! let u = &[1.0, 0.7, 0.7];
//!
//! // Change the default alpha and disable verbose output
//! let settings = Settings::default()
//!     .alpha(1.0)
//!     .verbose(false);
//! # let settings = settings.adaptive_rho(false);
//!
//! // Create an OSQP workspace
//! let mut workspace = Workspace::new(P, q, A, l, u, &settings);
//!
//! // Solve problem
//! let solution = workspace.solve();
//!
//! // Print the solution
//! println!("{:?}", solution.x());
//! #
//! # // Check the solution
//! # let expected = &[0.2987710845986426, 0.701227995544065];
//! # assert_eq!(expected.len(), solution.x().len());
//! # assert!(expected.iter().zip(solution.x()).all(|(&a, &b)| (a - b).abs() < 1e-9));
//! ```

extern crate osqp_sys;
#[macro_use]
extern crate static_assertions;

use osqp_sys as ffi;
use std::ptr::null_mut;

mod csc;
pub use csc::CscMatrix;

mod settings;
pub use settings::{LinsysSolver, Settings};

mod solution;
pub use solution::{Solution, Status};

#[allow(non_camel_case_types)]
type float = f64;

// Ensure osqp_int is the same size as usize/isize.
assert_eq_size!(osqp_int_size; usize, ffi::osqp_int);

macro_rules! check {
    ($ret:expr) => {
        assert_eq!($ret, 0);
    };
}

#[allow(non_snake_case)]
pub struct Workspace {
    inner: *mut ffi::OSQPWorkspace,
    /// Number of variables
    n: usize,
    /// Number of constraints
    m: usize,
    /// Number of data values in P
    P_nnz: usize,
    /// Number of data values in A
    A_nnz: usize,
}

impl Workspace {
    #[allow(non_snake_case)]
    pub fn new<'a, 'b, T: Into<CscMatrix<'a>>, U: Into<CscMatrix<'b>>>(
        P: T,
        q: &[float],
        A: U,
        l: &[float],
        u: &[float],
        settings: &Settings,
    ) -> Workspace {
        // Function split to avoid monomorphising the main body of Workspace::new.
        Workspace::new_inner(P.into(), q, A.into(), l, u, settings)
    }

    #[allow(non_snake_case)]
    fn new_inner(
        P: CscMatrix,
        q: &[float],
        A: CscMatrix,
        l: &[float],
        u: &[float],
        settings: &Settings,
    ) -> Workspace {
        unsafe {
            // Ensure the provided data is valid. While OSQP internally performs some validity
            // checks it can be made to read outside the provided buffers so all the invariants
            // are checked here.

            // Dimensions must be consistent with the number of variables
            let n = P.nrows;
            assert_eq!(n, P.ncols, "P must be a square matrix");
            assert_eq!(n, q.len(), "q must be the same number of rows as P");
            assert_eq!(n, A.ncols, "A must have the same number of columns as P");

            // Dimensions must be consistent with the number of constraints
            let m = A.nrows;
            assert_eq!(m, l.len(), "l must have the same number of rows as A");
            assert_eq!(m, u.len(), "u must have the same number of rows as A");

            // csc_to_ffi ensures sparse matrices have valid structure and that indices do not
            // exceed isize::MAX
            let mut P_ffi = P.to_ffi();
            let mut A_ffi = A.to_ffi();

            let data = ffi::OSQPData {
                n: n as ffi::osqp_int,
                m: m as ffi::osqp_int,
                P: &mut P_ffi,
                A: &mut A_ffi,
                q: q.as_ptr() as *mut float,
                l: l.as_ptr() as *mut float,
                u: u.as_ptr() as *mut float,
            };

            let settings = &settings.inner as *const ffi::OSQPSettings as *mut ffi::OSQPSettings;

            let inner = ffi::osqp_setup(&data, settings);
            assert!(inner as usize != 0, "osqp setup failure");

            Workspace {
                inner,
                n,
                m,
                P_nnz: P.data.len(),
                A_nnz: A.data.len(),
            }
        }
    }

    pub fn update_lin_cost(&mut self, q: &[float]) {
        unsafe {
            assert_eq!(self.n, q.len());
            check!(ffi::osqp_update_lin_cost(
                self.inner,
                q.as_ptr() as *mut float
            ));
        }
    }

    pub fn update_bounds(&mut self, l: &[float], u: &[float]) {
        unsafe {
            assert_eq!(self.m, l.len());
            assert_eq!(self.m, u.len());
            check!(ffi::osqp_update_bounds(
                self.inner,
                l.as_ptr() as *mut float,
                u.as_ptr() as *mut float,
            ));
        }
    }

    pub fn update_lower_bound(&mut self, l: &[float]) {
        unsafe {
            assert_eq!(self.m, l.len());
            check!(ffi::osqp_update_lower_bound(
                self.inner,
                l.as_ptr() as *mut float
            ));
        }
    }

    pub fn update_upper_bound(&mut self, u: &[float]) {
        unsafe {
            assert_eq!(self.m, u.len());
            check!(ffi::osqp_update_upper_bound(
                self.inner,
                u.as_ptr() as *mut float
            ));
        }
    }

    pub fn warm_start(&mut self, x: &[float], y: &[float]) {
        unsafe {
            assert_eq!(self.n, x.len());
            assert_eq!(self.m, y.len());
            check!(ffi::osqp_warm_start(
                self.inner,
                x.as_ptr() as *mut float,
                y.as_ptr() as *mut float,
            ));
        }
    }

    pub fn warm_start_x(&mut self, x: &[float]) {
        unsafe {
            assert_eq!(self.n, x.len());
            check!(ffi::osqp_warm_start_x(self.inner, x.as_ptr() as *mut float));
        }
    }

    pub fn warm_start_y(&mut self, y: &[float]) {
        unsafe {
            assert_eq!(self.m, y.len());
            check!(ffi::osqp_warm_start_y(self.inner, y.as_ptr() as *mut float));
        }
    }

    #[allow(non_snake_case)]
    pub fn update_P(&mut self, P_data: &[float]) {
        unsafe {
            assert_eq!(self.P_nnz, P_data.len());
            check!(ffi::osqp_update_P(
                self.inner,
                P_data.as_ptr() as *mut float,
                null_mut(),
                P_data.len() as ffi::osqp_int
            ));
        }
    }

    #[allow(non_snake_case)]
    pub fn update_A(&mut self, A_data: &[float]) {
        unsafe {
            assert_eq!(self.A_nnz, A_data.len());
            check!(ffi::osqp_update_A(
                self.inner,
                A_data.as_ptr() as *mut float,
                null_mut(),
                A_data.len() as ffi::osqp_int
            ));
        }
    }

    #[allow(non_snake_case)]
    pub fn update_P_A(&mut self, P_data: &[float], A_data: &[float]) {
        unsafe {
            assert_eq!(self.P_nnz, P_data.len());
            assert_eq!(self.A_nnz, A_data.len());
            check!(ffi::osqp_update_P_A(
                self.inner,
                P_data.as_ptr() as *mut float,
                null_mut(),
                P_data.len() as ffi::osqp_int,
                A_data.as_ptr() as *mut float,
                null_mut(),
                A_data.len() as ffi::osqp_int,
            ));
        }
    }

    pub fn solve<'a>(&'a mut self) -> Solution<'a> {
        unsafe {
            check!(ffi::osqp_solve(self.inner));
            Solution { ws: self }
        }
    }
}

impl Drop for Workspace {
    fn drop(&mut self) {
        unsafe {
            ffi::osqp_cleanup(self.inner);
        }
    }
}
