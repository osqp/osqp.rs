/* automatically generated by rust-bindgen */

use {c_float, c_int, OSQPTimer};

pub const SUITESPARSE_LDL_SOLVER: linsys_solver_type = 0;
pub const MKL_PARDISO_SOLVER: linsys_solver_type = 1;
/// Linear System Solvers *
pub type linsys_solver_type = u32;
/// Matrix in compressed-column or triplet form
#[repr(C)]
pub struct csc {
    /// < maximum number of entries.
    pub nzmax: c_int,
    /// < number of rows
    pub m: c_int,
    /// < number of columns
    pub n: c_int,
    /// < column pointers (size n+1) (col indices (size nzmax)
    pub p: *mut c_int,
    /// < row indices, size nzmax starting from 0
    pub i: *mut c_int,
    /// < numerical values, size nzmax
    pub x: *mut c_float,
    /// < # of entries in triplet matrix, -1 for csc
    pub nz: c_int,
}
/// Linear system solver structure (sublevel objects initialize it differently)
pub type LinSysSolver = linsys_solver;
/// Problem scaling matrices stored as vectors
#[repr(C)]
pub struct OSQPScaling {
    /// < cost function scaling
    pub c: c_float,
    /// < primal variable scaling
    pub D: *mut c_float,
    /// < dual variable scaling
    pub E: *mut c_float,
    /// < cost function rescaling
    pub cinv: c_float,
    /// < primal variable rescaling
    pub Dinv: *mut c_float,
    /// < dual variable rescaling
    pub Einv: *mut c_float,
}
/// Solution structure
#[repr(C)]
pub struct OSQPSolution {
    /// < Primal solution
    pub x: *mut c_float,
    /// < Lagrange multiplier associated to \f$l <= Ax <= u\f$
    pub y: *mut c_float,
}
/// Solver return nformation
#[repr(C)]
pub struct OSQPInfo {
    /// < number of iterations taken
    pub iter: c_int,
    /// < status string, e.g. 'solved'
    pub status: [::std::os::raw::c_char; 32usize],
    /// < status as c_int, defined in constants.h
    pub status_val: c_int,
    /// < polish status: successful (1), unperformed (0), (-1)
    pub status_polish: c_int,
    /// < primal objective
    pub obj_val: c_float,
    /// < norm of primal residual
    pub pri_res: c_float,
    /// < norm of dual residual
    pub dua_res: c_float,
    /// < time taken for setup phase (seconds)
    pub setup_time: c_float,
    /// < time taken for solve phase (seconds)
    pub solve_time: c_float,
    /// < time taken for polish phase (seconds)
    pub polish_time: c_float,
    /// < total time  (seconds)
    pub run_time: c_float,
    /// < number of rho updates
    pub rho_updates: c_int,
    /// < best rho estimate so far from residuals
    pub rho_estimate: c_float,
}
/// Polish structure
#[repr(C)]
pub struct OSQPPolish {
    /// < Active rows of A.
    /// ///<    Ared = vstack[Alow, Aupp]
    pub Ared: *mut csc,
    /// < number of lower-active rows
    pub n_low: c_int,
    /// < number of upper-active rows
    pub n_upp: c_int,
    /// < Maps indices in A to indices in Alow
    pub A_to_Alow: *mut c_int,
    /// < Maps indices in A to indices in Aupp
    pub A_to_Aupp: *mut c_int,
    /// < Maps indices in Alow to indices in A
    pub Alow_to_A: *mut c_int,
    /// < Maps indices in Aupp to indices in A
    pub Aupp_to_A: *mut c_int,
    /// < optimal x-solution obtained by polish
    pub x: *mut c_float,
    /// < optimal z-solution obtained by polish
    pub z: *mut c_float,
    /// < optimal y-solution obtained by polish
    pub y: *mut c_float,
    /// < objective value at polished solution
    pub obj_val: c_float,
    /// < primal residual at polished solution
    pub pri_res: c_float,
    /// < dual residual at polished solution
    pub dua_res: c_float,
}
/// Data structure
#[repr(C)]
pub struct OSQPData {
    /// < number of variables n
    pub n: c_int,
    /// < number of constraints m
    pub m: c_int,
    /// < quadratic part of the cost P in csc format (size n x n). It
    pub P: *mut csc,
    /// < linear constraints matrix A in csc format (size m x n)
    pub A: *mut csc,
    /// < dense array for linear part of cost function (size n)
    pub q: *mut c_float,
    /// < dense array for lower bound (size m)
    pub l: *mut c_float,
    /// < dense array for upper bound (size m)
    pub u: *mut c_float,
}
/// Settings struct
#[repr(C)]
pub struct OSQPSettings {
    /// < ADMM step rho
    pub rho: c_float,
    /// < ADMM step sigma
    pub sigma: c_float,
    /// < heuristic data scaling iterations. If 0,
    pub scaling: c_int,
    /// < boolean, is rho step size adaptive?
    pub adaptive_rho: c_int,
    /// < Number of iterations between rho
    pub adaptive_rho_interval: c_int,
    /// < Tolerance X for adapting rho. The new rho
    pub adaptive_rho_tolerance: c_float,
    /// < Interval for adapting rho (fraction of
    pub adaptive_rho_fraction: c_float,
    /// < maximum iterations
    pub max_iter: c_int,
    /// < absolute convergence tolerance
    pub eps_abs: c_float,
    /// < relative convergence tolerance
    pub eps_rel: c_float,
    /// < primal infeasibility tolerance
    pub eps_prim_inf: c_float,
    /// < dual infeasibility tolerance
    pub eps_dual_inf: c_float,
    /// < relaxation parameter
    pub alpha: c_float,
    /// < linear system solver to use
    pub linsys_solver: linsys_solver_type,
    /// < regularization parameter for
    pub delta: c_float,
    /// < boolean, polish ADMM solution
    pub polish: c_int,
    /// < iterative refinement steps in
    pub polish_refine_iter: c_int,
    /// < boolean, write out progres
    pub verbose: c_int,
    /// < boolean, use scaled termination
    pub scaled_termination: c_int,
    /// < integer, check termination
    pub check_termination: c_int,
    /// < boolean, warm start
    pub warm_start: c_int,
    /// < maximum seconds allowed to solve
    pub time_limit: c_float,
}
/// OSQP Workspace
#[repr(C)]
pub struct OSQPWorkspace {
    /// Problem data to work on (possibly scaled)
    pub data: *mut OSQPData,
    /// Linear System solver structure
    pub linsys_solver: *mut LinSysSolver,
    /// Polish structure
    pub pol: *mut OSQPPolish,
    /// < vector of rho values
    pub rho_vec: *mut c_float,
    /// < vector of inv rho values
    pub rho_inv_vec: *mut c_float,
    /// < Type of constraints: loose (-1), equality (1),
    pub constr_type: *mut c_int,
    /// < Iterate x
    pub x: *mut c_float,
    /// < Iterate y
    pub y: *mut c_float,
    /// < Iterate z
    pub z: *mut c_float,
    /// < Iterate xz_tilde
    pub xz_tilde: *mut c_float,
    /// < Previous x
    pub x_prev: *mut c_float,
    /// < Previous z
    pub z_prev: *mut c_float,
    /// < Scaled A * x
    pub Ax: *mut c_float,
    /// < Scaled P * x
    pub Px: *mut c_float,
    /// < Scaled A * x
    pub Aty: *mut c_float,
    /// < Difference of consecutive dual iterates
    pub delta_y: *mut c_float,
    /// < A' * delta_y
    pub Atdelta_y: *mut c_float,
    /// < Difference of consecutive primal iterates
    pub delta_x: *mut c_float,
    /// < P * delta_x
    pub Pdelta_x: *mut c_float,
    /// < A * delta_x
    pub Adelta_x: *mut c_float,
    /// < temporary primal variable scaling vectors
    pub D_temp: *mut c_float,
    /// < temporary primal variable scaling vectors storing
    pub D_temp_A: *mut c_float,
    /// < temporary constraints scaling vectors storing norms of
    pub E_temp: *mut c_float,
    /// < Problem settings
    pub settings: *mut OSQPSettings,
    /// < Scaling vectors
    pub scaling: *mut OSQPScaling,
    /// < Problem solution
    pub solution: *mut OSQPSolution,
    /// < Solver information
    pub info: *mut OSQPInfo,
    /// < Timer object
    pub timer: *mut OSQPTimer,
    /// flag indicating whether the solve function has been run before
    pub first_run: c_int,
    /// < Has last summary been printed? (true/false)
    pub summary_printed: c_int,
}
/// Define linsys_solver prototype structure
///
/// NB: The details are defined when the linear solver is initialized depending
/// on the choice
#[repr(C)]
pub struct linsys_solver {
    /// < Linear system solver type (see type.h)
    pub type_: linsys_solver_type,
    pub solve: ::std::option::Option<
        unsafe extern "C" fn(
            self_: *mut LinSysSolver,
            b: *mut c_float,
            settings: *const OSQPSettings,
        ) -> c_int,
    >,
    /// < Free linear system solver
    pub free: ::std::option::Option<unsafe extern "C" fn(self_: *mut LinSysSolver)>,
    pub update_matrices: ::std::option::Option<
        unsafe extern "C" fn(
            self_: *mut LinSysSolver,
            P: *const csc,
            A: *const csc,
            settings: *const OSQPSettings,
        ) -> c_int,
    >,
    pub update_rho_vec: ::std::option::Option<
        unsafe extern "C" fn(s: *mut LinSysSolver, rho_vec: *const c_float, m: c_int) -> c_int,
    >,
    /// < Number of threads active
    pub nthreads: c_int,
}
extern "C" {
    /// Set default settings from constants.h file
    /// assumes settings already allocated in memory
    /// @param settings settings structure
    pub fn osqp_set_default_settings(settings: *mut OSQPSettings);
}
extern "C" {
    /// Initialize OSQP solver allocating memory.
    ///
    /// All the inputs must be already allocated in memory before calling.
    ///
    /// It performs:
    /// - data and settings validation
    /// - problem data scaling
    /// - automatic parameters tuning (if enabled)
    /// - setup linear system solver:
    /// - direct solver: KKT matrix factorization is performed here
    /// - indirect solver: KKT matrix preconditioning is performed here
    ///
    /// NB: This is the only function that allocates dynamic memory and is not used
    /// during code generation
    ///
    /// @param  data         Problem data
    /// @param  settings     Solver settings
    /// @return              Solver environment
    pub fn osqp_setup(data: *const OSQPData, settings: *mut OSQPSettings) -> *mut OSQPWorkspace;
}
extern "C" {
    /// Solve quadratic program
    ///
    /// The final solver information is stored in the \a work->info  structure
    ///
    /// The solution is stored in the  \a work->solution  structure
    ///
    /// If the problem is primal infeasible, the certificate is stored
    /// in \a work->delta_y
    ///
    /// If the problem is dual infeasible, the certificate is stored in \a
    /// work->delta_x
    ///
    /// @param  work Workspace allocated
    /// @return      Exitflag for errors
    pub fn osqp_solve(work: *mut OSQPWorkspace) -> c_int;
}
extern "C" {
    /// Cleanup workspace by deallocating memory
    ///
    /// This function is not used in code generation
    /// @param  work Workspace
    /// @return      Exitflag for errors
    pub fn osqp_cleanup(work: *mut OSQPWorkspace) -> c_int;
}
extern "C" {
    /// Update linear cost in the problem
    /// @param  work  Workspace
    /// @param  q_new New linear cost
    /// @return       Exitflag for errors and warnings
    pub fn osqp_update_lin_cost(work: *mut OSQPWorkspace, q_new: *const c_float) -> c_int;
}
extern "C" {
    /// Update lower and upper bounds in the problem constraints
    /// @param  work   Workspace
    /// @param  l_new New lower bound
    /// @param  u_new New upper bound
    /// @return        Exitflag: 1 if new lower bound is not <= than new upper bound
    pub fn osqp_update_bounds(
        work: *mut OSQPWorkspace,
        l_new: *const c_float,
        u_new: *const c_float,
    ) -> c_int;
}
extern "C" {
    /// Update lower bound in the problem constraints
    /// @param  work   Workspace
    /// @param  l_new New lower bound
    /// @return        Exitflag: 1 if new lower bound is not <= than upper bound
    pub fn osqp_update_lower_bound(work: *mut OSQPWorkspace, l_new: *const c_float) -> c_int;
}
extern "C" {
    /// Update upper bound in the problem constraints
    /// @param  work   Workspace
    /// @param  u_new New upper bound
    /// @return        Exitflag: 1 if new upper bound is not >= than lower bound
    pub fn osqp_update_upper_bound(work: *mut OSQPWorkspace, u_new: *const c_float) -> c_int;
}
extern "C" {
    /// Warm start primal and dual variables
    /// @param  work Workspace structure
    /// @param  x    Primal variable
    /// @param  y    Dual variable
    /// @return      Exitflag
    pub fn osqp_warm_start(work: *mut OSQPWorkspace, x: *const c_float, y: *const c_float)
        -> c_int;
}
extern "C" {
    /// Warm start primal variable
    /// @param  work Workspace structure
    /// @param  x    Primal variable
    /// @return      Exitflag
    pub fn osqp_warm_start_x(work: *mut OSQPWorkspace, x: *const c_float) -> c_int;
}
extern "C" {
    /// Warm start dual variable
    /// @param  work Workspace structure
    /// @param  y    Dual variable
    /// @return      Exitflag
    pub fn osqp_warm_start_y(work: *mut OSQPWorkspace, y: *const c_float) -> c_int;
}
extern "C" {
    /// Update elements of matrix P (upper-diagonal)
    /// without changing sparsity structure.
    ///
    ///
    /// If Px_new_idx is OSQP_NULL, Px_new is assumed to be as long as P->x
    /// and the whole P->x is replaced.
    ///
    /// @param  work       Workspace structure
    /// @param  Px_new     Vector of new elements in P->x (upper triangular)
    /// @param  Px_new_idx Index mapping new elements to positions in P->x
    /// @param  P_new_n    Number of new elements to be changed
    /// @return            output flag:  0: OK
    /// 1: P_new_n > nnzP
    /// <0: error in the update
    pub fn osqp_update_P(
        work: *mut OSQPWorkspace,
        Px_new: *const c_float,
        Px_new_idx: *const c_int,
        P_new_n: c_int,
    ) -> c_int;
}
extern "C" {
    /// Update elements of matrix A without changing sparsity structure.
    ///
    ///
    /// If Ax_new_idx is OSQP_NULL, Ax_new is assumed to be as long as A->x
    /// and the whole P->x is replaced.
    ///
    /// @param  work       Workspace structure
    /// @param  Ax_new     Vector of new elements in A->x
    /// @param  Ax_new_idx Index mapping new elements to positions in A->x
    /// @param  A_new_n    Number of new elements to be changed
    /// @return            output flag:  0: OK
    /// 1: A_new_n > nnzA
    /// <0: error in the update
    pub fn osqp_update_A(
        work: *mut OSQPWorkspace,
        Ax_new: *const c_float,
        Ax_new_idx: *const c_int,
        A_new_n: c_int,
    ) -> c_int;
}
extern "C" {
    /// Update elements of matrix P (upper-diagonal) and elements of matrix A
    /// without changing sparsity structure.
    ///
    ///
    /// If Px_new_idx is OSQP_NULL, Px_new is assumed to be as long as P->x
    /// and the whole P->x is replaced.
    ///
    /// If Ax_new_idx is OSQP_NULL, Ax_new is assumed to be as long as A->x
    /// and the whole P->x is replaced.
    ///
    /// @param  work       Workspace structure
    /// @param  Px_new     Vector of new elements in P->x (upper triangular)
    /// @param  Px_new_idx Index mapping new elements to positions in P->x
    /// @param  P_new_n    Number of new elements to be changed
    /// @param  Ax_new     Vector of new elements in A->x
    /// @param  Ax_new_idx Index mapping new elements to positions in A->x
    /// @param  A_new_n    Number of new elements to be changed
    /// @return            output flag:  0: OK
    /// 1: P_new_n > nnzP
    /// 2: A_new_n > nnzA
    /// <0: error in the update
    pub fn osqp_update_P_A(
        work: *mut OSQPWorkspace,
        Px_new: *const c_float,
        Px_new_idx: *const c_int,
        P_new_n: c_int,
        Ax_new: *const c_float,
        Ax_new_idx: *const c_int,
        A_new_n: c_int,
    ) -> c_int;
}
extern "C" {
    /// Update rho. Limit it between RHO_MIN and RHO_MAX.
    /// @param  work         Workspace
    /// @param  rho_new      New rho setting
    /// @return              Exitflag
    pub fn osqp_update_rho(work: *mut OSQPWorkspace, rho_new: c_float) -> c_int;
}
extern "C" {
    /// Update max_iter setting
    /// @param  work         Workspace
    /// @param  max_iter_new New max_iter setting
    /// @return              Exitflag
    pub fn osqp_update_max_iter(work: *mut OSQPWorkspace, max_iter_new: c_int) -> c_int;
}
extern "C" {
    /// Update absolute tolernace value
    /// @param  work        Workspace
    /// @param  eps_abs_new New absolute tolerance value
    /// @return             Exitflag
    pub fn osqp_update_eps_abs(work: *mut OSQPWorkspace, eps_abs_new: c_float) -> c_int;
}
extern "C" {
    /// Update relative tolernace value
    /// @param  work        Workspace
    /// @param  eps_rel_new New relative tolerance value
    /// @return             Exitflag
    pub fn osqp_update_eps_rel(work: *mut OSQPWorkspace, eps_rel_new: c_float) -> c_int;
}
extern "C" {
    /// Update primal infeasibility tolerance
    /// @param  work          Workspace
    /// @param  eps_prim_inf_new  New primal infeasibility tolerance
    /// @return               Exitflag
    pub fn osqp_update_eps_prim_inf(work: *mut OSQPWorkspace, eps_prim_inf_new: c_float) -> c_int;
}
extern "C" {
    /// Update dual infeasibility tolerance
    /// @param  work          Workspace
    /// @param  eps_dual_inf_new  New dual infeasibility tolerance
    /// @return               Exitflag
    pub fn osqp_update_eps_dual_inf(work: *mut OSQPWorkspace, eps_dual_inf_new: c_float) -> c_int;
}
extern "C" {
    /// Update relaxation parameter alpha
    /// @param  work  Workspace
    /// @param  alpha_new New relaxation parameter value
    /// @return       Exitflag
    pub fn osqp_update_alpha(work: *mut OSQPWorkspace, alpha_new: c_float) -> c_int;
}
extern "C" {
    /// Update warm_start setting
    /// @param  work           Workspace
    /// @param  warm_start_new New warm_start setting
    /// @return                Exitflag
    pub fn osqp_update_warm_start(work: *mut OSQPWorkspace, warm_start_new: c_int) -> c_int;
}
extern "C" {
    /// Update scaled_termination setting
    /// @param  work                 Workspace
    /// @param  scaled_termination_new  New scaled_termination setting
    /// @return                      Exitflag
    pub fn osqp_update_scaled_termination(
        work: *mut OSQPWorkspace,
        scaled_termination_new: c_int,
    ) -> c_int;
}
extern "C" {
    /// Update check_termination setting
    /// @param  work                   Workspace
    /// @param  check_termination_new  New check_termination setting
    /// @return                        Exitflag
    pub fn osqp_update_check_termination(
        work: *mut OSQPWorkspace,
        check_termination_new: c_int,
    ) -> c_int;
}
extern "C" {
    /// Update regularization parameter in polish
    /// @param  work      Workspace
    /// @param  delta_new New regularization parameter
    /// @return           Exitflag
    pub fn osqp_update_delta(work: *mut OSQPWorkspace, delta_new: c_float) -> c_int;
}
extern "C" {
    /// Update polish setting
    /// @param  work          Workspace
    /// @param  polish_new New polish setting
    /// @return               Exitflag
    pub fn osqp_update_polish(work: *mut OSQPWorkspace, polish_new: c_int) -> c_int;
}
extern "C" {
    /// Update number of iterative refinement steps in polish
    /// @param  work                Workspace
    /// @param  polish_refine_iter_new New iterative reginement steps
    /// @return                     Exitflag
    pub fn osqp_update_polish_refine_iter(
        work: *mut OSQPWorkspace,
        polish_refine_iter_new: c_int,
    ) -> c_int;
}
extern "C" {
    /// Update verbose setting
    /// @param  work        Workspace
    /// @param  verbose_new New verbose setting
    /// @return             Exitflag
    pub fn osqp_update_verbose(work: *mut OSQPWorkspace, verbose_new: c_int) -> c_int;
}
extern "C" {
    /// Update time_limit setting
    /// @param  work            Workspace
    /// @param  time_limit_new  New time_limit setting
    /// @return                 Exitflag
    pub fn osqp_update_time_limit(work: *mut OSQPWorkspace, time_limit_new: c_float) -> c_int;
}
pub const OSQP_DUAL_INFEASIBLE_INACCURATE: ffi_osqp_status = 4;
pub const OSQP_PRIMAL_INFEASIBLE_INACCURATE: ffi_osqp_status = 3;
pub const OSQP_SOLVED_INACCURATE: ffi_osqp_status = 2;
pub const OSQP_SOLVED: ffi_osqp_status = 1;
pub const OSQP_MAX_ITER_REACHED: ffi_osqp_status = -2;
pub const OSQP_PRIMAL_INFEASIBLE: ffi_osqp_status = -3;
pub const OSQP_DUAL_INFEASIBLE: ffi_osqp_status = -4;
pub const OSQP_SIGINT: ffi_osqp_status = -5;
pub const OSQP_TIME_LIMIT_REACHED: ffi_osqp_status = -6;
pub const OSQP_UNSOLVED: ffi_osqp_status = -10;
pub type ffi_osqp_status = i32;
