//! <script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.2/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>
//!
//! The OSQP (Operator Splitting Quadratic Program) solver is a numerical optimization package for
//! solving convex quadratic programs in the form
//!
//! <div class="math">
//! \[\begin{split}\begin{array}{ll}
//!   \mbox{minimize} &amp; \frac{1}{2} x^T P x + q^T x \\
//!   \mbox{subject to} &amp; l \leq A x \leq u
//! \end{array}\end{split}\]
//! </div>
//!
//! <p>
//! where \(x\) is the optimization variable and \(P \in \mathbf{S}^{n}_{+}\) a positive
//! semidefinite matrix.
//! </p>
//!
//! Further information about the solver is available at
//! [osqp.org](https://osqp.org/).
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
//! use osqp::{CscMatrix, Problem, Settings};
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
//! // Extract the upper triangular elements of `P`
//! let P = CscMatrix::from(P).into_upper_tri();
//!
//! // Disable verbose output
//! let settings = Settings::default()
//!     .verbose(false);
//! # let settings = settings.adaptive_rho(false);
//!
//! // Create an OSQP problem
//! let mut prob = Problem::new(P, q, A, l, u, &settings).expect("failed to setup problem");
//!
//! // Solve problem
//! let result = prob.solve();
//!
//! // Print the solution
//! println!("{:?}", result.x().expect("failed to solve problem"));
//! #
//! # // Check the solution
//! # let expected = &[0.30137570387082474, 0.6983956863817343];
//! # let x = result.solution().unwrap().x();
//! # assert_eq!(expected.len(), x.len());
//! # assert!(expected.iter().zip(x).all(|(&a, &b)| (a - b).abs() < 1e-9));
//! ```

extern crate osqp_sys;

use osqp_sys as ffi;
use std::error::Error;
use std::fmt;
use std::ptr;

mod csc;
pub use csc::CscMatrix;

mod settings;
pub use settings::{LinsysSolver, Settings};

mod status;

pub use status::{
    DualInfeasibilityCertificate, Failure, PolishStatus, PrimalInfeasibilityCertificate, Solution,
    Status,
};

#[allow(non_camel_case_types)]
type float = f64;

// Ensure osqp_int is the same size as usize/isize.
#[allow(dead_code)]
fn assert_osqp_int_size() {
    let _osqp_int_must_be_usize = ::std::mem::transmute::<ffi::osqp_int, usize>;
}

macro_rules! check {
    ($fun:ident, $ret:expr) => {
        assert!(
            $ret == 0,
            "osqp_{} failed with exit code {}",
            stringify!($fun),
            $ret
        );
    };
}

/// An instance of the OSQP solver.
pub struct Problem {
    solver: *mut ffi::OSQPSolver,
    /// Number of variables
    n: usize,
    /// Number of constraints
    m: usize,
}

impl Problem {
    /// Initialises the solver and validates the problem.
    ///
    /// Returns an error if the problem is non-convex or the solver cannot be initialised.
    ///
    /// Panics if any of the matrix or vector dimensions are incompatible, if `P` or `A` are not
    /// valid CSC matrices, or if `P` is not structurally upper triangular.
    #[allow(non_snake_case)]
    pub fn new<'a, 'b, T: Into<CscMatrix<'a>>, U: Into<CscMatrix<'b>>>(
        P: T,
        q: &[float],
        A: U,
        l: &[float],
        u: &[float],
        settings: &Settings,
    ) -> Result<Problem, SetupError> {
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
    ) -> Result<Problem, SetupError> {
        let invalid_data = |msg| Err(SetupError::DataInvalid(msg));

        unsafe {
            // Ensure the provided data is valid. While OSQP internally performs some validity
            // checks it can be made to read outside the provided buffers so all the invariants
            // are checked here.

            // Dimensions must be consistent with the number of variables
            let n = P.nrows;
            if P.ncols != n {
                return invalid_data("P must be a square matrix");
            }
            if q.len() != n {
                return invalid_data("q must be the same number of rows as P");
            }
            if A.ncols != n {
                return invalid_data("A must have the same number of columns as P");
            }

            // Dimensions must be consistent with the number of constraints
            let m = A.nrows;
            if l.len() != m {
                return invalid_data("l must have the same number of rows as A");
            }
            if u.len() != m {
                return invalid_data("u must have the same number of rows as A");
            }
            if l.iter().zip(u.iter()).any(|(&l, &u)| !(l <= u)) {
                return invalid_data("all elements of l must be less than or equal to the corresponding element of u");
            }

            // `A` and `P` must be valid CSC matrices and `P` must be structurally upper triangular
            if !P.is_valid() {
                return invalid_data("P must be a valid CSC matrix");
            }
            if !A.is_valid() {
                return invalid_data("A must be a valid CSC matrix");
            }
            if !P.is_structurally_upper_tri() {
                return invalid_data("P must be structurally upper triangular");
            }

            // Calling `to_ffi` is safe as we have ensured that `P` and `A` are valid CSC matrices.
            let P_ffi = P.to_ffi();
            let A_ffi = A.to_ffi();

            let settings = &settings.inner as *const ffi::OSQPSettings as *mut ffi::OSQPSettings;
            let mut solver: *mut ffi::OSQPSolver = ptr::null_mut();

            let status = ffi::osqp_setup(&mut solver, P_ffi, q.as_ptr(), A_ffi, l.as_ptr(), u.as_ptr(), m as ffi::osqp_int, n as ffi::osqp_int, settings);
            let err = match status as ffi::osqp_error_type {
                0 => return Ok(Problem { solver, n, m }),
                ffi::OSQP_DATA_VALIDATION_ERROR => SetupError::DataInvalid(""),
                ffi::OSQP_SETTINGS_VALIDATION_ERROR => SetupError::SettingsInvalid,
                ffi::OSQP_ALGEBRA_LOAD_ERROR => SetupError::LinsysSolverLoadFailed,
                ffi::OSQP_LINSYS_SOLVER_INIT_ERROR => SetupError::LinsysSolverInitFailed,
                ffi::OSQP_NONCVX_ERROR => SetupError::NonConvex,
                ffi::OSQP_MEM_ALLOC_ERROR => SetupError::MemoryAllocationFailed,
                _ => unreachable!(),
            };

            // If the call to `osqp_setup` fails the `OSQPSolver` may be partially allocated
            if !solver.is_null() {
                ffi::osqp_cleanup(solver);
            }
            Err(err)
        }
    }

    /// Sets the linear part of the cost function to `q`.
    ///
    /// Panics if the length of `q` is not the same as the number of problem variables.
    pub fn update_lin_cost(&mut self, q: &[float]) {
        unsafe {
            assert_eq!(self.n, q.len());
            check!(
                update_lin_cost,
                ffi::osqp_update_data_vec(self.solver, q.as_ptr(), ptr::null(), ptr::null())
            );
        }
    }

    /// Sets the lower and upper bounds of the constraints to `l` and `u`.
    ///
    /// Panics if the length of `l` or `u` is not the same as the number of problem constraints.
    pub fn update_bounds(&mut self, l: &[float], u: &[float]) {
        unsafe {
            assert_eq!(self.m, l.len());
            assert_eq!(self.m, u.len());
            check!(
                update_bounds,
                ffi::osqp_update_data_vec(self.solver, ptr::null(), l.as_ptr(), u.as_ptr())
            );
        }
    }

    /// Sets the lower bound of the constraints to `l`.
    ///
    /// Panics if the length of `l` is not the same as the number of problem constraints.
    pub fn update_lower_bound(&mut self, l: &[float]) {
        unsafe {
            assert_eq!(self.m, l.len());
            check!(
                update_lower_bound,
                ffi::osqp_update_data_vec(self.solver, ptr::null(), l.as_ptr(), ptr::null())
            );
        }
    }

    /// Sets the upper bound of the constraints to `u`.
    ///
    /// Panics if the length of `u` is not the same as the number of problem constraints.
    pub fn update_upper_bound(&mut self, u: &[float]) {
        unsafe {
            assert_eq!(self.m, u.len());
            check!(
                update_upper_bound,
                ffi::osqp_update_data_vec(self.solver, ptr::null(), ptr::null(), u.as_ptr())
            );
        }
    }

    /// Warm starts the primal variables at `x` and the dual variables at `y`.
    ///
    /// Panics if the length of `x` is not the same as the number of problem variables or the
    /// length of `y` is not the same as the number of problem constraints.
    pub fn warm_start(&mut self, x: &[float], y: &[float]) {
        unsafe {
            assert_eq!(self.n, x.len());
            assert_eq!(self.m, y.len());
            check!(
                warm_start,
                ffi::osqp_warm_start(self.solver, x.as_ptr(), y.as_ptr())
            );
        }
    }

    /// Warm starts the primal variables at `x`.
    ///
    /// Panics if the length of `x` is not the same as the number of problem variables.
    pub fn warm_start_x(&mut self, x: &[float]) {
        unsafe {
            assert_eq!(self.n, x.len());
            check!(
                warm_start_x,
                ffi::osqp_warm_start(self.solver, x.as_ptr(), ptr::null())
            );
        }
    }

    /// Warms start the dual variables at `y`.
    ///
    /// Panics if the length of `y` is not the same as the number of problem constraints.
    pub fn warm_start_y(&mut self, y: &[float]) {
        unsafe {
            assert_eq!(self.m, y.len());
            check!(
                warm_start_y,
                ffi::osqp_warm_start(self.solver, ptr::null(), y.as_ptr())
            );
        }
    }

    /// Updates the elements of matrix `P` without changing its sparsity structure.
    ///
    /// Panics if the sparsity structure of `P` differs from the sparsity structure of the `P`
    /// matrix provided to `Problem::new`.
    #[allow(non_snake_case)]
    pub fn update_P<'a, T: Into<CscMatrix<'a>>>(&mut self, P: T) {
        self.update_P_inner(P.into());
    }

    #[allow(non_snake_case)]
    fn update_P_inner(&mut self, P: CscMatrix) {
        unsafe {
            // TODO: can no longer get current P/q/A/l/u from the solver?
            // let P_ffi = CscMatrix::from_ffi((*(*self.solver).work).P);
            // P.assert_same_sparsity_structure(&P_ffi);

            check!(
                update_P,
                ffi::osqp_update_data_mat(
                    self.solver,
                    P.data.as_ptr(),
                    ptr::null(),
                    P.data.len() as ffi::osqp_int,
                    ptr::null(),
                    ptr::null(),
                    0
                )
            );
        }
    }

    /// Updates the elements of matrix `A` without changing its sparsity structure.
    ///
    /// Panics if the sparsity structure of `A` differs from the sparsity structure of the `A`
    /// matrix provided to `Problem::new`.
    #[allow(non_snake_case)]
    pub fn update_A<'a, T: Into<CscMatrix<'a>>>(&mut self, A: T) {
        self.update_A_inner(A.into());
    }

    #[allow(non_snake_case)]
    fn update_A_inner(&mut self, A: CscMatrix) {
        unsafe {
            // TODO: can no longer get current P/q/A/l/u from the solver?
            // let A_ffi = CscMatrix::from_ffi((*(*self.workspace).data).A);
            // A.assert_same_sparsity_structure(&A_ffi);

            check!(
                update_A,
                ffi::osqp_update_data_mat(
                    self.solver,
                    ptr::null(),
                    ptr::null(),
                    0,
                    A.data.as_ptr(),
                    ptr::null(),
                    A.data.len() as ffi::osqp_int,
                )
            );
        }
    }

    /// Attempts to solve the quadratic program.
    pub fn solve<'a>(&'a mut self) -> Status<'a> {
        unsafe {
            check!(solve, ffi::osqp_solve(self.solver));
            Status::from_problem(self)
        }
    }
}

impl Drop for Problem {
    fn drop(&mut self) {
        unsafe {
            ffi::osqp_cleanup(self.solver);
        }
    }
}

unsafe impl Send for Problem {}
unsafe impl Sync for Problem {}

/// An error that can occur when setting up the solver.
#[derive(Debug)]
pub enum SetupError {
    DataInvalid(&'static str),
    SettingsInvalid,
    LinsysSolverLoadFailed,
    LinsysSolverInitFailed,
    NonConvex,
    MemoryAllocationFailed,
    // Prevent exhaustive enum matching
    #[doc(hidden)]
    __Nonexhaustive,
}

impl fmt::Display for SetupError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SetupError::DataInvalid(msg) => {
                "problem data invalid".fmt(f)?;
                if !msg.is_empty() {
                    ": ".fmt(f)?;
                    msg.fmt(f)?;
                }
                Ok(())
            }
            SetupError::SettingsInvalid => "problem settings invalid".fmt(f),
            SetupError::LinsysSolverLoadFailed => "linear system solver failed to load".fmt(f),
            SetupError::LinsysSolverInitFailed => {
                "linear system solver failed to initialise".fmt(f)
            }
            SetupError::NonConvex => "problem non-convex".fmt(f),
            SetupError::MemoryAllocationFailed => "memory allocation failed".fmt(f),
            SetupError::__Nonexhaustive => unreachable!(),
        }
    }
}

impl Error for SetupError {}

#[cfg(test)]
mod tests {
    use std::iter;

    use super::*;

    #[test]
    #[allow(non_snake_case)]
    fn update_settings() {
        // Directly update some settings using methods on the Settings object

        let settings = Settings::default().alpha(0.7).verbose(false);
        let settings = settings.adaptive_rho(true);

        assert_eq!(settings.inner.alpha, 0.7);
        assert_eq!(settings.inner.verbose, 0);
        assert_eq!(settings.inner.adaptive_rho, 1);
    }

    #[test]
    #[allow(non_snake_case)]
    fn update_prob_settings() {
        // Indirectly update some settings using methods on the Problem object

        // Define problem data
        let P = CscMatrix::from(&[[4.0, 1.0], [1.0, 2.0]]).into_upper_tri();
        let q = &[1.0, 1.0];
        let A = &[[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]];
        let l = &[1.0, 0.0, 0.0];
        let u = &[1.0, 0.7, 0.7];

        let settings = Settings::default().verbose(false);
        let mut prob = Problem::new(&P, q, A, l, u, &settings).unwrap();

        prob.update_rho(0.7);
        unsafe {
            let solver_ref = prob.solver.as_ref().unwrap();
            let settings_ref = solver_ref.settings.as_ref().unwrap();

            assert!((*settings_ref).rho == 0.7, "Expected rho to be 0.7 but found {}", (*settings_ref).rho);

        }
    }

    #[test]
    #[allow(non_snake_case)]
    fn update_matrices() {
        // Define problem data
        let P_wrong = CscMatrix::from(&[[2.0, 1.0], [1.0, 4.0]]).into_upper_tri();
        let A_wrong = &[[2.0, 3.0], [1.0, 0.0], [0.0, 9.0]];

        let P = CscMatrix::from(&[[4.0, 1.0], [1.0, 2.0]]).into_upper_tri();
        let q = &[1.0, 1.0];
        let A = &[[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]];
        let l = &[1.0, 0.0, 0.0];
        let u = &[1.0, 0.7, 0.7];

        // Change the default alpha and disable verbose output
        let settings = Settings::default().alpha(1.0).verbose(false);
        let settings = settings.adaptive_rho(false);

        // Check updating P and A separately
        let mut prob = Problem::new(&P_wrong, q, A_wrong, l, u, &settings).unwrap();
        prob.update_P(&P);
        prob.update_A(A);
        let result = prob.solve();
        let x = result.solution().unwrap().x();
        let expected = &[0.2987710845986426, 0.701227995544065];
        assert_eq!(expected.len(), x.len());
        assert!(expected.iter().zip(x).all(|(&a, &b)| (a - b).abs() < 1e-9));
    }

    #[test]
    #[allow(non_snake_case)]
    fn empty_A() {
        let P = CscMatrix::from(&[[4.0, 1.0], [1.0, 2.0]]).into_upper_tri();
        let q = &[1.0, 1.0];

        let A = CscMatrix::from_column_iter_dense(0, 2, iter::empty());
        let l = &[];
        let u = &[];
        let mut prob = Problem::new(&P, q, &A, l, u, &Settings::default()).unwrap();
        prob.update_A(&A);

        let A = CscMatrix::from(&[[0.0, 0.0], [0.0, 0.0]]);
        assert_eq!(A.data.len(), 0);
        let l = &[0.0, 0.0];
        let u = &[1.0, 1.0];
        let mut prob = Problem::new(&P, q, &A, l, u, &Settings::default()).unwrap();
        prob.update_A(&A);
    }
}

