extern crate cmake;
use cmake::Config;

use std::env;
use std::fs;
use std::path::Path;
use std::process::Command;

fn main() {
    if !Path::new("osqp/README.md").exists() {
        let _ = Command::new("git")
            .args(&["submodule", "update", "--init", "--recursive"])
            .status();
    }

    // Try to make c_int the same size as the target pointer width (i.e. 32 or 64 bits)
    println!("cargo:rustc-check-cfg=cfg(osqp_dlong)");
    let dlong_enabled = match &*env::var("CARGO_CFG_TARGET_POINTER_WIDTH").unwrap() {
        "64" => {
            println!("cargo:rustc-cfg=osqp_dlong");
            "ON"
        }
        "32" => "OFF",
        other => panic!(
            "{} bit targets are not supported. If you want this feature please file a bug.",
            other
        ),
    };

    // The CMake build script for OSQP generates files inside the source directory.
    // The docs.rs builder does not like this, so we copy the OSQP source tree into `OUT_DIR`.
    let out_dir = env::var("OUT_DIR").unwrap();
    let src_dir = Path::new(&out_dir).join("src");
    let build_dir = Path::new(&out_dir).join("build");
    let build_dir = build_dir.to_str().unwrap();

    fs::create_dir_all(&src_dir).expect("failed to create OSQP sources directory in `OUT_DIR`");
    fs::remove_dir_all(&src_dir).expect("failed to delete old OSQP sources directory in `OUT_DIR`");
    fs::create_dir_all(&src_dir).expect("failed to create OSQP sources directory in `OUT_DIR`");

    fs_extra::dir::copy(
        "osqp",
        &src_dir,
        &fs_extra::dir::CopyOptions {
            overwrite: true,
            skip_exist: false,
            content_only: true,
            ..fs_extra::dir::CopyOptions::new()
        },
    )
    .expect("failed to copy OSQP sources to `OUT_DIR`");

    fs::create_dir_all(build_dir).expect("failed to create OSQP build directory in `OUT_DIR`");

    Config::new(&src_dir)
        .define("OSQP_ENABLE_INTERRUPT", "OFF")
        .define("OSQP_USE_FLOAT", "OFF")
        .define("OSQP_USE_LONG", dlong_enabled)
        .define("OSQP_ENABLE_PRINTING", "ON")
        .define("OSQP_ENABLE_PROFILING", "ON")
        .define("OSQP_BUILD_UNITTESTS", "OFF")
        // Ensure build outputs are always in `build_dir` whichever generator CMake uses
        .define("CMAKE_ARCHIVE_OUTPUT_DIRECTORY", &build_dir)
        .define("CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG", &build_dir)
        .define("CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE", &build_dir)
        .define("CMAKE_ARCHIVE_OUTPUT_DIRECTORY_MINSIZEREL", &build_dir)
        .define("CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELWITHDEBINFO", &build_dir)
        .build_target("osqpstatic")
        .build();

    println!("cargo:rustc-link-search=native={}", build_dir);
    println!("cargo:rustc-link-lib=static=osqpstatic");
}
