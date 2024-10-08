use std::env;
use std::path::{Path, PathBuf};

use bindgen::EnumVariation;

fn include(path: &Path, header: &str) -> String {
    path.join(header).to_str().unwrap().to_string()
}

fn main() {
    let amf_sdk_include = if cfg!(target_os = "windows") {
        PathBuf::from(env::var("AMF_SDK_PATH").expect(
            "environment variable AMF_SDK_PATH must be set for nv-video-codec-sys to find the CUDA SDK",
        ))
    } else if cfg!(target_os = "linux") {
        PathBuf::from(env::var("AMF_SDK_PATH").unwrap_or("/usr/include/AMF".to_string()))
    } else {
        todo!("Unsupported platform")
    };
    if !amf_sdk_include.exists() || !amf_sdk_include.is_dir() {
        panic!(
            "Path to the AMF SDK is invalid or inaccessible : {}",
            amf_sdk_include.display()
        );
    }

    let bindgen = bindgen::Builder::default()
        .clang_args(["-I", amf_sdk_include.to_str().unwrap()])
        .header(include(&amf_sdk_include, "core/Factory.h"))
        .header(include(&amf_sdk_include, "core/Version.h"))
        .header(include(&amf_sdk_include, "components/VideoDecoderUVD.h"))
        .generate_comments(false)
        .default_enum_style(EnumVariation::Rust {
            non_exhaustive: false,
        })
        .derive_default(true)
        .use_core()
        .sort_semantically(true)
        .merge_extern_blocks(true)
        .allowlist_function("^AMF.*")
        .allowlist_type("^AMF.*")
        .parse_callbacks(Box::new(
            bindgen::CargoCallbacks::new().rerun_on_header_files(true),
        ));

    let bindings = bindgen.generate().expect("Unable to generate bindings");

    // Write the bindings to the $OUT_DIR/bindings.rs file.
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}
