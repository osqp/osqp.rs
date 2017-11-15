#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

use std::mem;

mod ffi;
pub use ffi::*;

mod ffi_internal;

impl Default for OSQPSettings {
    fn default() -> Self {
        unsafe {
            let mut settings = mem::zeroed();
            ffi_internal::set_default_settings(&mut settings);
            settings
        }
    }
}
