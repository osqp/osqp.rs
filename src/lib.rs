//! <script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.2/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>
//!
//! <p>
//! The OSQP (Operator Splitting Quadratic Program) solver is a numerical optimization package for
//! solving convex quadratic programs in the form
//!
//! \[\begin{split}\begin{array}{ll}
//!   \mbox{minimize} &amp; \frac{1}{2} x^T P x + q^T x \\
//!   \mbox{subject to} &amp; l \leq A x \leq u
//! \end{array}\end{split}\]
//!
//! where \(x\) is the optimization variable and \(P \in \mathbf{S}^{n}_{+}\) a positive
//! semidefinite matrix.
//! </p>
//!
//! Further information about the solver is available at
//! [osqp.readthedocs.io](https://osqp.readthedocs.io/).
//!
//! # Example
//!
//! Consider the following QP
//!
//! <div class="math">
//! \[\begin{split}\begin{array}{ll}
//!   \mbox{minimize} &amp; \frac{1}{2} x^T \begin{bmatrix}4 &amp; 1\\ 1 &amp; 2 \end{bmatrix} x + \begin{bmatrix}1 \\ 1\end{bmatrix}^T x \\
//!   \mbox{subject to} &amp; \begin{bmatrix}1 \\ 0 \\ 0\end{bmatrix} \leq \begin{bmatrix} 1 &amp; 1\\ 1 &amp; 0\\ 0 &amp; 1\end{bmatrix} x \leq  \begin{bmatrix}1 \\ 0.7 \\ 0.7\end{bmatrix}
//! \end{array}\end{split}\]
//! </div>
//!
//! ```rust
//! use osqp::{Settings, Problem};
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
//! // Create an OSQP problem
//! let mut prob = Problem::new(P, q, A, l, u, &settings);
//!
//! // Solve problem
//! let result = prob.solve();
//!
//! // Print the solution
//! println!("{:?}", result.x().expect("failed to solve problem"));
//! #
//! # // Check the solution
//! # let expected = &[0.2987710845986426, 0.701227995544065];
//! # let x = result.solution().unwrap().x();
//! # assert_eq!(expected.len(), x.len());
//! # assert!(expected.iter().zip(x).all(|(&a, &b)| (a - b).abs() < 1e-9));
//! ```

extern crate osqp_sys;

use osqp_sys as ffi;
use std::ptr::null_mut;

mod csc;
pub use csc::CscMatrix;

mod settings;
pub use settings::{LinsysSolver, Settings};

mod status;
pub use status::{DualInfeasibilityCertificate, PolishStatus, PrimalInfeasibilityCertificate,
                 Solution, Status};

#[allow(non_camel_case_types)]
type float = f64;

// Ensure osqp_int is the same size as usize/isize.
#[allow(dead_code)]
fn assert_osqp_int_size() {
    let _osqp_int_must_be_usize = ::std::mem::transmute::<ffi::osqp_int, usize>;
}

macro_rules! check {
    ($ret:expr) => {
        assert_eq!($ret, 0);
    };
}

/// An instance of the OSQP solver.
#[allow(non_snake_case)]
pub struct Problem {
    inner: *mut ffi::OSQPWorkspace,
    /// Number of variables
    n: usize,
    /// Number of constraints
    m: usize,
    /// P upper triangle CSC data
    P_upper_tri_data: Vec<float>,
}

impl Problem {
    /// Initialises the solver and validates the problem.
    ///
    /// Panics if the problem is invalid.
    #[allow(non_snake_case)]
    pub fn new<'a, 'b, T: Into<CscMatrix<'a>>, U: Into<CscMatrix<'b>>>(
        P: T,
        q: &[float],
        A: U,
        l: &[float],
        u: &[float],
        settings: &Settings,
    ) -> Problem {
        // Function split to avoid monomorphising the main body of Problem::new.
        Problem::new_inner(P.into(), q, A.into(), l, u, settings)
    }

    #[allow(non_snake_case)]
    fn new_inner(
        P: CscMatrix,
        q: &[float],
        A: CscMatrix,
        l: &[float],
        u: &[float],
        settings: &Settings,
    ) -> Problem {
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

            Problem {
                inner,
                n,
                m,
                P_upper_tri_data: Vec::with_capacity((P.data.len() + n + 1) / 2),
            }
        }
    }

    /// Sets the linear part of the cost function to `q`.
    pub fn update_lin_cost(&mut self, q: &[float]) {
        unsafe {
            assert_eq!(self.n, q.len());
            check!(ffi::osqp_update_lin_cost(
                self.inner,
                q.as_ptr() as *mut float
            ));
        }
    }

    /// Sets the lower and upper bounds of the constraints to `l` and `u`.
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

    /// Sets the lower bound of the constraints to `l`.
    pub fn update_lower_bound(&mut self, l: &[float]) {
        unsafe {
            assert_eq!(self.m, l.len());
            check!(ffi::osqp_update_lower_bound(
                self.inner,
                l.as_ptr() as *mut float
            ));
        }
    }

    /// Sets the upper bound of the constraints to `u`.
    pub fn update_upper_bound(&mut self, u: &[float]) {
        unsafe {
            assert_eq!(self.m, u.len());
            check!(ffi::osqp_update_upper_bound(
                self.inner,
                u.as_ptr() as *mut float
            ));
        }
    }

    /// Warm starts the primal variables at `x` and the dual variables at `y`.
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

    /// Warm starts the primal variables at `x`.
    pub fn warm_start_x(&mut self, x: &[float]) {
        unsafe {
            assert_eq!(self.n, x.len());
            check!(ffi::osqp_warm_start_x(self.inner, x.as_ptr() as *mut float));
        }
    }

    /// Warms start the dual variables at `y`.
    pub fn warm_start_y(&mut self, y: &[float]) {
        unsafe {
            assert_eq!(self.m, y.len());
            check!(ffi::osqp_warm_start_y(self.inner, y.as_ptr() as *mut float));
        }
    }

    /// Updates the elements of matrix `P` without changing its sparsity structure.
    #[allow(non_snake_case)]
    pub fn update_P<'a, T: Into<CscMatrix<'a>>>(&mut self, P: T) {
        self.update_P_inner(P.into());
    }

    #[allow(non_snake_case)]
    fn update_P_inner(&mut self, P: CscMatrix) {
        unsafe {
            P.assert_valid();
            let P_ffi = CscMatrix::from_ffi((*(*self.inner).data).P);
            P.assert_same_upper_tri_sparsity_structure(&P_ffi);

            self.fill_P_upper_tri_data(P);
            check!(ffi::osqp_update_P(
                self.inner,
                self.P_upper_tri_data.as_ptr() as *mut float,
                null_mut(),
                self.P_upper_tri_data.len() as ffi::osqp_int
            ));
        }
    }

    /// Updates the elements of matrix `A` without changing its sparsity structure.
    #[allow(non_snake_case)]
    pub fn update_A<'a, T: Into<CscMatrix<'a>>>(&mut self, A: T) {
        self.update_A_inner(A.into());
    }

    #[allow(non_snake_case)]
    fn update_A_inner(&mut self, A: CscMatrix) {
        unsafe {
            A.assert_valid();
            let A_ffi = CscMatrix::from_ffi((*(*self.inner).data).A);
            A.assert_same_sparsity_structure(&A_ffi);

            check!(ffi::osqp_update_A(
                self.inner,
                A.data.as_ptr() as *mut float,
                null_mut(),
                A.data.len() as ffi::osqp_int
            ));
        }
    }

    /// Updates the elements of matrices `P` and `A` without changing either's sparsity structure.
    #[allow(non_snake_case)]
    pub fn update_P_A<'a, 'b, T: Into<CscMatrix<'a>>, U: Into<CscMatrix<'b>>>(
        &mut self,
        P: T,
        A: U,
    ) {
        self.update_P_A_inner(P.into(), A.into());
    }

    #[allow(non_snake_case)]
    fn update_P_A_inner(&mut self, P: CscMatrix, A: CscMatrix) {
        unsafe {
            P.assert_valid();
            let P_ffi = CscMatrix::from_ffi((*(*self.inner).data).P);
            P.assert_same_upper_tri_sparsity_structure(&P_ffi);

            A.assert_valid();
            let A_ffi = CscMatrix::from_ffi((*(*self.inner).data).A);
            A.assert_same_sparsity_structure(&A_ffi);

            self.fill_P_upper_tri_data(P);
            check!(ffi::osqp_update_P_A(
                self.inner,
                self.P_upper_tri_data.as_ptr() as *mut float,
                null_mut(),
                self.P_upper_tri_data.len() as ffi::osqp_int,
                A.data.as_ptr() as *mut float,
                null_mut(),
                A.data.len() as ffi::osqp_int,
            ));
        }
    }

    /// Copies the upper triangular elements of P to self.P_upper_tri_data.
    #[allow(non_snake_case)]
    fn fill_P_upper_tri_data(&mut self, P: CscMatrix) {
        self.P_upper_tri_data.truncate(0);

        let mut col_start_idx = 0;
        for (col_num, &col_end_idx) in P.indptr.iter().skip(1).enumerate() {
            for (row_idx, &row_num) in P.indices[col_start_idx..col_end_idx].iter().enumerate() {
                // Copy only the diagonal and all elements above it
                if row_num > col_num {
                    break;
                }
                self.P_upper_tri_data.push(P.data[col_start_idx + row_idx]);
            }
            col_start_idx = col_end_idx;
        }
    }

    /// Attempts to solve the quadratic program.
    pub fn solve<'a>(&'a mut self) -> Status<'a> {
        unsafe {
            check!(ffi::osqp_solve(self.inner));
            Status::from_problem(self)
        }
    }
}

impl Drop for Problem {
    fn drop(&mut self) {
        unsafe {
            ffi::osqp_cleanup(self.inner);
        }
    }
}

unsafe impl Send for Problem {}
unsafe impl Sync for Problem {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[allow(non_snake_case)]
    fn update_matrices() {
        // Define problem data
        let P_wrong = &[[2.0, 1.0], [1.0, 4.0]];
        let A_wrong = &[[2.0, 3.0], [1.0, 0.0], [0.0, 9.0]];

        let P = &[[4.0, 1.0], [1.0, 2.0]];
        let q = &[1.0, 1.0];
        let A = &[[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]];
        let l = &[1.0, 0.0, 0.0];
        let u = &[1.0, 0.7, 0.7];

        // Change the default alpha and disable verbose output
        let settings = Settings::default().alpha(1.0).verbose(false);
        let settings = settings.adaptive_rho(false);

        // Check updating P and A together
        let mut prob = Problem::new(P_wrong, q, A_wrong, l, u, &settings);
        prob.update_P_A(P, A);
        let result = prob.solve();
        let x = result.solution().unwrap().x();
        let expected = &[0.2987710845986426, 0.701227995544065];
        assert_eq!(expected.len(), x.len());
        assert!(expected.iter().zip(x).all(|(&a, &b)| (a - b).abs() < 1e-9));

        // Check updating P and A separately
        let mut prob = Problem::new(P_wrong, q, A_wrong, l, u, &settings);
        prob.update_P(P);
        prob.update_A(A);
        let result = prob.solve();
        let x = result.solution().unwrap().x();
        let expected = &[0.2987710845986426, 0.701227995544065];
        assert_eq!(expected.len(), x.len());
        assert!(expected.iter().zip(x).all(|(&a, &b)| (a - b).abs() < 1e-9));
    }
}
