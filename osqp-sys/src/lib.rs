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

