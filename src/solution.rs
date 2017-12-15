use osqp_sys as ffi;
use std::slice;
use std::time::Duration;

use {float, Problem};

/// A solution returned by the solver.
pub struct Solution<'a> {
    pub(crate) ws: &'a Problem,
}

/// The reason for the solver returning a solution.
#[derive(Clone, Debug, PartialEq)]
pub enum Status {
    Solved,
    SolvedInaccurate,
    MaxIterationsReached,
    PrimalInfeasible,
    PrimalInfeasibleInaccurate,
    DualInfeasible,
    DualInfeasibleInaccurate,
    // Prevent exhaustive enum matching
    #[doc(hidden)] __Nonexhaustive,
}

impl<'a> Solution<'a> {
    /// Returns the primal variables at the solution.
    pub fn x(&self) -> &'a [float] {
        unsafe { slice::from_raw_parts((*(*self.ws.inner).solution).x, self.ws.n) }
    }

    /// Returns the dual variables at the solution.
    ///
    /// These are the Lagrange multipliers of the constraints `l <= Ax <= u`.
    pub fn y(&self) -> &'a [float] {
        unsafe { slice::from_raw_parts((*(*self.ws.inner).solution).y, self.ws.m) }
    }

    /// Returns the number of iterations taken by the solver.
    pub fn iter(&self) -> u32 {
        unsafe {
            // cast safe as more than 2 billion iterations would be unreasonable
            (*(*self.ws.inner).info).iter as u32
        }
    }

    /// Returns the status of the solution.
    pub fn status(&self) -> Status {
        use std::os::raw::c_int;
        unsafe {
            match (*(*self.ws.inner).info).status_val as c_int {
                ffi::OSQP_SOLVED => Status::Solved,
                ffi::OSQP_SOLVED_INACCURATE => Status::SolvedInaccurate,
                ffi::OSQP_MAX_ITER_REACHED => Status::MaxIterationsReached,
                ffi::OSQP_PRIMAL_INFEASIBLE => Status::PrimalInfeasible,
                ffi::OSQP_PRIMAL_INFEASIBLE_INACCURATE => Status::PrimalInfeasibleInaccurate,
                ffi::OSQP_DUAL_INFEASIBLE => Status::DualInfeasible,
                ffi::OSQP_DUAL_INFEASIBLE_INACCURATE => Status::DualInfeasibleInaccurate,
                _ => unreachable!(),
            }
        }
    }

    /// Returns the primal objective value.
    pub fn obj_val(&self) -> float {
        unsafe { (*(*self.ws.inner).info).obj_val }
    }

    /// Returns the norm of primal residual.
    pub fn pri_res(&self) -> float {
        unsafe { (*(*self.ws.inner).info).pri_res }
    }

    /// Returns the norm of dual residual.
    pub fn dua_res(&self) -> float {
        unsafe { (*(*self.ws.inner).info).dua_res }
    }

    /// Returns the time taken for the setup phase.
    pub fn setup_time(&self) -> Duration {
        unsafe { secs_to_duration((*(*self.ws.inner).info).setup_time) }
    }

    /// Returns the time taken for the solve phase.
    pub fn solve_time(&self) -> Duration {
        unsafe { secs_to_duration((*(*self.ws.inner).info).solve_time) }
    }

    /// Returns the time taken for the polish phase.
    pub fn polish_time(&self) -> Duration {
        unsafe { secs_to_duration((*(*self.ws.inner).info).polish_time) }
    }

    /// Returns the total time taken by the solver.
    pub fn run_time(&self) -> Duration {
        unsafe { secs_to_duration((*(*self.ws.inner).info).run_time) }
    }

    /// Returns the number of rho updates.
    pub fn rho_updates(&self) -> u32 {
        unsafe {
            // cast safe as more than 2 billion updates would be unreasonable
            (*(*self.ws.inner).info).rho_updates as u32
        }
    }

    /// Returns the current best estimate of rho.
    pub fn rho_estimate(&self) -> float {
        unsafe { (*(*self.ws.inner).info).rho_estimate }
    }
}

fn secs_to_duration(secs: float) -> Duration {
    let whole_secs = secs.floor() as u64;
    let nanos = (secs.fract() * 1e9) as u32;
    Duration::new(whole_secs, nanos)
}
