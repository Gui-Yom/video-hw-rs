use std::env;
use std::path::PathBuf;

use bindgen::{CargoCallbacks, EnumVariation};

fn link_lib(lib: &str) {
    if cfg!(feature = "static") {
        println!("cargo:rustc-link-lib=static={lib}_static");
    } else {
        println!("cargo:rustc-link-lib={lib}");
    }
}

fn main() {
    println!("cargo:rerun-if-changed=wrapper.h");

    // TODO platform dependant detection
    let CUDA_HOME = r#"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3"#;
    let include_path = format!("{CUDA_HOME}/include");

    // println!("cargo:rustc-link-lib=cuda");
    println!("cargo:rustc-link-search=C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.3\\lib\\x64");
    let mut bindgen = bindgen::Builder::default()
        .clang_args(["-I", &include_path])
        .header(format!("{include_path}/nppdefs.h"))
        .header(format!("{include_path}/nppcore.h"))
        .must_use_type("NppStatus")
        .blocklist_type("(Npp8u)|(Npp8s)|(Npp16u)|(Npp16s)|(Npp32f)")
        .generate_comments(false)
        .default_enum_style(EnumVariation::Rust {
            non_exhaustive: false,
        })
        .derive_default(true)
        .use_core()
        .sort_semantically(true)
        .merge_extern_blocks(true)
        .parse_callbacks(Box::new(CargoCallbacks::new()));

    link_lib("nppc");
    #[cfg(all(feature = "static", target_os = "linux"))]
    link_lib("culibos");

    #[cfg(feature = "icc")]
    {
        link_lib("nppic");
        bindgen = bindgen.header(format!("{include_path}/nppi_color_conversion.h"));
    }
    #[cfg(feature = "if")]
    {
        link_lib("nppif");
        bindgen = bindgen.header(format!("{include_path}/nppi_image_filtering.h"));
    }
    #[cfg(feature = "ig")]
    {
        link_lib("nppig");
        bindgen = bindgen.header(format!("{include_path}/nppi_geometry_transforms.h"));
    }
    #[cfg(feature = "isu")]
    {
        link_lib("nppisu");
        bindgen = bindgen.header(format!("{include_path}/nppi_support_functions.h"));
    }
    let bindings = bindgen.generate().expect("Unable to generate bindings");

    // Write the bindings to the $OUT_DIR/bindings.rs file.
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}
