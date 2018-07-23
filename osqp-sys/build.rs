extern crate cmake;
use cmake::Config;

use std::env;
use std::path::Path;
use std::process::Command;

fn main() {
    if !Path::new("osqp/README.md").exists() {
        let _ = Command::new("git")
            .args(&["submodule", "update", "--init"])
            .status();
    }

    // Try to make c_int the same size as the target pointer width (i.e. 32 or 64 bits)
    let dlong_enabled = match &*env::var("CARGO_CFG_TARGET_POINTER_WIDTH").unwrap() {
        "64" => {
            println!(r#"cargo:rustc-cfg=feature="osqp_dlong""#);
            "ON"
        }
        "32" => "OFF",
        other => panic!(
            "{} bit targets are not supported. If you want this feature please file a bug.",
            other
        ),
    };

    let out_dir = env::var("OUT_DIR").unwrap();

    Config::new("osqp")
        .define("CTRLC", "OFF")
        .define("DFLOAT", "OFF")
        .define("DLONG", dlong_enabled)
        .define("PRINTING", "ON")
        .define("PROFILING", "ON")
        .define("UNITTESTS", "OFF")
        // Ensure build outputs are always in OUT_DIR whichever generator CMake uses
        .define("CMAKE_ARCHIVE_OUTPUT_DIRECTORY", &out_dir)
        .define("CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG", &out_dir)
        .define("CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE", &out_dir)
        .define("CMAKE_ARCHIVE_OUTPUT_DIRECTORY_MINSIZEREL", &out_dir)
        .define("CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELWITHDEBINFO", &out_dir)
        .build_target("osqp")
        .build();

    println!("cargo:rustc-link-search=native={}", out_dir);
    println!("cargo:rustc-link-lib=static=osqp");
}
