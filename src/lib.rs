extern crate osqp_sys;
#[macro_use]
extern crate static_assertions;

use osqp_sys as ffi;
use std::borrow::Cow;
use std::ptr::null_mut;
use std::slice;

pub use ffi::OSQPSettings as Settings;

#[allow(non_camel_case_types)]
type float = f64;

// Ensure osqp's c_int is the same size as usize/isize.
assert_eq_size!(osqp_c_int; usize, ffi::c_int);

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
    pub fn new(
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
            let mut P_ffi = csc_to_ffi(&P);
            let mut A_ffi = csc_to_ffi(&A);

            let data = ffi::OSQPData {
                n: n as ffi::c_int,
                m: m as ffi::c_int,
                P: &mut P_ffi,
                A: &mut A_ffi,
                q: q.as_ptr() as *mut float,
                l: l.as_ptr() as *mut float,
                u: u.as_ptr() as *mut float,
            };

            let inner = ffi::osqp_setup(&data, settings as *const Settings as *mut Settings);
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
                P_data.len() as ffi::c_int
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
                A_data.len() as ffi::c_int
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
                P_data.len() as ffi::c_int,
                A_data.as_ptr() as *mut float,
                null_mut(),
                A_data.len() as ffi::c_int,
            ));
        }
    }

    pub fn solve<'a>(&'a mut self) -> Solution<'a> {
        unsafe {
            check!(ffi::osqp_solve(self.inner));

            let status = match (*(*self.inner).info).status_val {
                1 => Status::Solved,
                2 => Status::SolvedInaccurate,
                -2 => Status::MaxIterationsReached,
                -3 => Status::PrimalInfeasible,
                3 => Status::PrimalInfeasibleInaccurate,
                -4 => Status::DualInfeasible,
                4 => Status::DualInfeasibleInaccurate,
                _ => unreachable!(),
            };

            Solution {
                status,
                x: slice::from_raw_parts((*(*self.inner).solution).x, self.n),
                y: slice::from_raw_parts((*(*self.inner).solution).y, self.m),
            }
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

pub struct Solution<'a> {
    pub status: Status,
    pub x: &'a [float],
    pub y: &'a [float],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Status {
    Solved,
    SolvedInaccurate,
    MaxIterationsReached,
    PrimalInfeasible,
    PrimalInfeasibleInaccurate,
    DualInfeasible,
    DualInfeasibleInaccurate,
}

pub struct CscMatrix<'a> {
    pub nrows: usize,
    pub ncols: usize,
    pub indptr: Cow<'a, [usize]>,
    pub indices: Cow<'a, [usize]>,
    pub data: Cow<'a, [float]>,
}

unsafe fn csc_to_ffi(csc: &CscMatrix) -> ffi::csc {
    assert_valid_csc(csc);

    // Casting is safe as at this point no indices exceed isize::MAX and c_int is a signed integer
    // of the same size as usize/isize
    ffi::csc {
        nzmax: csc.data.len() as ffi::c_int,
        m: csc.nrows as ffi::c_int,
        n: csc.ncols as ffi::c_int,
        p: csc.indptr.as_ptr() as *mut usize as *mut ffi::c_int,
        i: csc.indices.as_ptr() as *mut usize as *mut ffi::c_int,
        x: csc.data.as_ptr() as *mut float,
        nz: -1,
    }
}

fn assert_valid_csc(matrix: &CscMatrix) {
    use std::isize::MAX;
    let max_idx = MAX as usize;
    assert!(matrix.nrows <= max_idx);
    assert!(matrix.ncols <= max_idx);
    assert!(matrix.indptr.len() <= max_idx);
    assert!(matrix.indices.len() <= max_idx);
    assert!(matrix.data.len() <= max_idx);

    // Check row pointers
    assert_eq!(matrix.indptr[matrix.ncols], matrix.data.len());
    assert_eq!(matrix.indptr.len(), matrix.ncols + 1);
    matrix.indptr.iter().fold(0, |acc, i| {
        assert!(
            *i >= acc,
            "csc row pointers must be monotonically nondecreasing"
        );
        *i
    });

    // Check index values
    assert_eq!(
        matrix.data.len(),
        matrix.indices.len(),
        "csc row indices must be the same length as data"
    );
    assert!(matrix.indices.iter().all(|r| *r < matrix.nrows));
    for i in 0..matrix.ncols {
        let row_indices = &matrix.indices[matrix.indptr[i] as usize..matrix.indptr[i + 1] as usize];
        let first_index = *row_indices.get(0).unwrap_or(&0);
        row_indices.iter().skip(1).fold(first_index, |acc, i| {
            assert!(*i > acc, "csc row indices must be monotonically increasing");
            *i
        });
    }
}
