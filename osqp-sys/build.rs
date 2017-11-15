extern crate cc;

use std::env;
use std::path::Path;
use std::process::Command;

fn main() {
    if !Path::new("osqp/.git").exists() {
        let _ = Command::new("git")
            .args(&["submodule", "update", "--init"])
            .status();
    }

    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap();

    let os_define = match &target_os[..] {
        "linux" => "IS_LINUX",
        "macos" => "IS_MAC",
        "windows" => "IS_WINDOWS",
        _ => panic!("unspported os {}", target_os),
    };

    cc::Build::new()
        .warnings(false)
        .opt_level(3)
        .define(os_define, None)
        .define("DLONG", None)
        .define("PRINTING", None)
        .define("PROFILING", None)
        // OSQP
        .include("osqp/include")
        .file("osqp/src/auxil.c")
        .file("osqp/src/cs.c")
        .file("osqp/src/ctrlc.c")
        .file("osqp/src/kkt.c")
        .file("osqp/src/lin_alg.c")
        .file("osqp/src/osqp.c")
        .file("osqp/src/polish.c")
        .file("osqp/src/proj.c")
        .file("osqp/src/scaling.c")
        .file("osqp/src/util.c")
        // SuiteSparse
        .include("osqp/lin_sys/direct/suitesparse")
        .file("osqp/lin_sys/direct/suitesparse/SuiteSparse_config.c")
        .file("osqp/lin_sys/direct/suitesparse/private.c")
        // SuiteSparse AMD
        .include("osqp/lin_sys/direct/suitesparse/amd/include")
        .file("osqp/lin_sys/direct/suitesparse/amd/src/amd_1.c")
        .file("osqp/lin_sys/direct/suitesparse/amd/src/amd_2.c")
        .file("osqp/lin_sys/direct/suitesparse/amd/src/amd_aat.c")
        .file("osqp/lin_sys/direct/suitesparse/amd/src/amd_control.c")
        .file("osqp/lin_sys/direct/suitesparse/amd/src/amd_defaults.c")
        .file("osqp/lin_sys/direct/suitesparse/amd/src/amd_info.c")
        .file("osqp/lin_sys/direct/suitesparse/amd/src/amd_order.c")
        .file("osqp/lin_sys/direct/suitesparse/amd/src/amd_post_tree.c")
        .file("osqp/lin_sys/direct/suitesparse/amd/src/amd_postorder.c")
        .file("osqp/lin_sys/direct/suitesparse/amd/src/amd_preprocess.c")
        .file("osqp/lin_sys/direct/suitesparse/amd/src/amd_valid.c")
        // SuiteSparse LDL
        .include("osqp/lin_sys/direct/suitesparse/ldl/include")
        .file("osqp/lin_sys/direct/suitesparse/ldl/src/ldl.c")
        .compile("osqp");
}
