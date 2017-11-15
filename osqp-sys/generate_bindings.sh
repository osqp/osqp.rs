#!/bin/bash

bindgen osqp/include/osqp.h -o src/ffi.rs \
--whitelist-function osqp_setup \
--whitelist-function osqp_solve \
--whitelist-function osqp_cleanup \
--whitelist-function osqp_update_lin_cost \
--whitelist-function osqp_update_bounds \
--whitelist-function osqp_update_lower_bound \
--whitelist-function osqp_update_upper_bound \
--whitelist-function osqp_warm_start \
--whitelist-function osqp_warm_start_x \
--whitelist-function osqp_warm_start_y \
--whitelist-function osqp_update_P \
--whitelist-function osqp_update_A \
--whitelist-function osqp_update_P_A \
-- -DDLONG -DPRINTING -DPROFILING

bindgen osqp/include/osqp.h -o src/ffi_internal.rs \
--raw-line "use OSQPSettings;" \
--whitelist-function set_default_settings \
--no-recursive-whitelist \
-- -DDLONG -DPRINTING -DPROFILING


