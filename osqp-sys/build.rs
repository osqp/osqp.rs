extern crate cmake;
use cmake::Config;

use std::path::Path;
use std::process::Command;

fn main() {
    if !Path::new("osqp/.git").exists() {
        let _ = Command::new("git")
            .args(&["submodule", "update", "--init"])
            .status();
    }

    // TODO: Figure out the story around cmake and cross-compilation
    let dst = Config::new("osqp")
        .define("CTRLC", "OFF")
        .define("DFLOAT", "OFF")
        // TODO: c_int is either int or long long make the default the target pointer size
        .define("DLONG", "ON")
        .define("PRINTING", "ON")
        .define("PROFILING", "ON")
        .define("UNITTESTS", "OFF")
        .build_target("osqpstatic")
        .build();

    println!(
        "cargo:rustc-link-search=native={}",
        dst.join("build").join("out").display()
    );
    println!("cargo:rustc-link-lib=static=osqpstatic");
}
