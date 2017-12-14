use osqp_sys as ffi;
use std::mem;
use std::ptr;

use {float, Workspace};

/// The linear system solver for OSQP to use.
#[derive(Clone, Debug, PartialEq)]
pub enum LinsysSolver {
    SuiteSparse,
    MklPardiso,
    // Prevent exhaustive enum matching
    #[doc(hidden)] __Nonexhaustive,
}

macro_rules! u32_to_osqp_int {
    ($name:ident, $value:expr) => ({
        let value = $value;
        assert!(
            value as u64 <= ffi::osqp_int::max_value() as u64,
            "{} must be smaller than the largest isize value",
            stringify!($name)
        );
        value as ffi::osqp_int
    });
}

macro_rules! rust_type {
    (float) => (float);
    (u32) => (u32);
    (option_u32) => (Option<u32>);
    (bool) => (bool);
    (linsys_solver) => (LinsysSolver);
}

macro_rules! convert_rust_type {
    ($name:ident, float, $value:expr) => ($value);
    ($name:ident, u32, $value:expr) => (u32_to_osqp_int!($name, $value));
    ($name:ident, option_u32, $value:expr) => (u32_to_osqp_int!($name, $value.unwrap_or(0)));
    ($name:ident, bool, $value:expr) => ($value as ffi::osqp_int);
    ($name:ident, linsys_solver, $value:expr) => (
        match $value {
            LinsysSolver::SuiteSparse => ffi::SUITESPARSE_LDL_SOLVER,
            LinsysSolver::MklPardiso => ffi::MKL_PARDISO_SOLVER,
            LinsysSolver::__Nonexhaustive => unreachable!(),
        }
    );
}

macro_rules! settings {
    ($workspace:path, $(
        #[$doc:meta] $name:ident: $typ:ident $([$update_name:ident, $update_ffi:ident])*,
    )*) => (
        /// The settings used when initialising a solver.
        pub struct Settings {
            pub(crate) inner: ffi::OSQPSettings,
        }

        impl Settings {
            $(
                #[$doc]
                pub fn $name(mut self, value: rust_type!($typ)) -> Settings {
                    self.inner.$name = convert_rust_type!($name, $typ, value);
                    Settings {
                        inner: self.inner
                    }
                }
            )*
        }

        impl Clone for Settings {
            fn clone(&self) -> Settings {
                unsafe {
                    Settings {
                        inner: ptr::read(&self.inner)
                    }
                }
            }
        }

        impl Default for Settings {
            fn default() -> Settings {
                unsafe {
                    let mut settings: ffi::OSQPSettings = mem::zeroed();
                    ffi::set_default_settings(&mut settings);
                    Settings {
                        inner: settings
                    }
                }
            }
        }

        impl $workspace {
            $($(
                #[$doc]
                pub fn $update_name(&mut self, value: rust_type!($typ)) {
                    unsafe {
                        let ret = ffi::$update_ffi(
                            self.inner,
                            convert_rust_type!($name, $typ, value)
                        );
                        if ret != 0 {
                            panic!("updating {} failed", stringify!($name));
                        }
                    }
                }
            )*)*
        }
    );
}

settings! {
    Workspace,

    #[doc = "Sets the ADMM step rho."]
    rho: float [update_rho, osqp_update_rho],

    #[doc = "Sets the ADMM step sigma."]
    sigma: float,

    #[doc = "
    Sets the number of heuristic data scaling iterations.

    If `None` scaling is disabled.

    Panics on 32-bit platforms if the value is above `i32::max_value()`.
    "]
    scaling: option_u32,

    #[doc = "Enables choosing rho adaptively."]
    adaptive_rho: bool,

    #[doc = "
    Sets the number of iterations between rho adaptations.

    If `None` it is automatic.

    Panics on 32-bit platforms if the value is above `i32::max_value()`.
    "]
    adaptive_rho_interval: option_u32,

    #[doc = "
    Sets the tolerance for adapting rho.

    The new rho has to be `x` times larger or `1/x` times smaller than the current rho to trigger
    a new factorization.
    "]
    adaptive_rho_tolerance: float,

    #[doc = "Set the interval for adapting rho as a fraction of the setup time."]
    adaptive_rho_fraction: float,

    #[doc = "
    Sets the maximum number of ADMM iterations.

    Panics on 32-bit platforms if the value is above `i32::max_value()`.
    "]
    max_iter: u32 [update_max_iter, osqp_update_max_iter],

    #[doc = "Sets the absolute convergence tolerance."]
    eps_abs: float [update_eps_abs, osqp_update_eps_abs],

    #[doc = "Sets the relative convergence tolerance."]
    eps_rel: float [update_eps_rel, osqp_update_eps_rel],

    #[doc = "Sets the primal infeasibility tolerance."]
    eps_prim_inf: float [update_eps_prim_inf, osqp_update_eps_prim_inf],

    #[doc = "Sets the dual infeasibility tolerance."]
    eps_dual_inf: float [update_eps_dual_inf, osqp_update_eps_dual_inf],

    #[doc = "Sets the linear solver relaxation parameter."]
    alpha: float [update_alpha, osqp_update_alpha],

    #[doc = "Sets the linear system solver to use."]
    linsys_solver: linsys_solver,

    #[doc = "Sets the polishing regularization parameter."]
    delta: float [update_delta, osqp_update_delta],

    #[doc = "Enables polishing the ADMM solution."]
    polish: bool [update_polish, osqp_update_polish],

    #[doc = "
    Sets the number of iterative refinement steps to use when polishing.

    Panics on 32-bit platforms if the value is above `i32::max_value()`.
    "]
    polish_refine_iter: u32 [update_polish_refine_iter, osqp_update_polish_refine_iter],

    #[doc = "Enables writing progress to stdout."]
    verbose: bool [update_verbose, osqp_update_verbose],

    #[doc = "Enables scaled termination criteria."]
    scaled_termination: bool [update_scaled_termination, osqp_update_scaled_termination],

    #[doc = "
    Sets the number of ADMM iterations between termination checks.

    If `None` termination checking is disabled.

    Panics on 32-bit platforms if the value is above `i32::max_value()`.
    "]
    check_termination: option_u32 [update_check_termination, osqp_update_check_termination],

    #[doc = "Enables warm starting the primal and dual variables from the previous solution."]
    warm_start: bool [update_warm_start, osqp_update_warm_start],
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(target_pointer_width = "32")]
    #[test]
    #[should_panic]
    fn large_u32_settings_value_panics_on_32_bit() {
        // Larger than i32::max_value()
        Settings::default().polish_refine_iter(3_000_000_000);
    }
}
