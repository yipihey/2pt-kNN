//! Build script: generates a Rust file that embeds all vendored Typst package
//! files (both .typ sources and typst.toml manifests) as string constants,
//! allowing in-memory package resolution without filesystem access (for WASM).

use std::env;
use std::fs;
use std::io::Write;
use std::path::Path;

fn main() {
    let out_dir = env::var("OUT_DIR").unwrap();
    let dest = Path::new(&out_dir).join("typst_packages.rs");
    let mut f = fs::File::create(&dest).unwrap();

    let pkg_root = Path::new("src/typst-packages");

    // Generate source files (.typ)
    writeln!(f, "/// Auto-generated: vendored Typst package .typ sources.").unwrap();
    writeln!(
        f,
        "pub fn package_sources() -> Vec<(&'static str, &'static str, &'static str, &'static str, &'static str)> {{"
    ).unwrap();
    writeln!(f, "    vec![").unwrap();
    if pkg_root.exists() {
        walk_packages(&mut f, pkg_root, "typ", "include_str!");
    }
    writeln!(f, "    ]").unwrap();
    writeln!(f, "}}").unwrap();

    // Generate binary files (typst.toml)
    writeln!(f).unwrap();
    writeln!(f, "/// Auto-generated: vendored Typst package binary files (typst.toml).").unwrap();
    writeln!(
        f,
        "pub fn package_binaries() -> Vec<(&'static str, &'static str, &'static str, &'static str, &'static [u8])> {{"
    ).unwrap();
    writeln!(f, "    vec![").unwrap();
    if pkg_root.exists() {
        walk_packages(&mut f, pkg_root, "toml", "include_bytes!");
    }
    writeln!(f, "    ]").unwrap();
    writeln!(f, "}}").unwrap();

    println!("cargo:rerun-if-changed=src/typst-packages");
}

fn walk_packages(f: &mut fs::File, pkg_root: &Path, ext: &str, macro_name: &str) {
    for namespace_entry in fs::read_dir(pkg_root).unwrap() {
        let namespace_entry = namespace_entry.unwrap();
        let namespace = namespace_entry.file_name().to_string_lossy().to_string();
        for name_entry in fs::read_dir(namespace_entry.path()).unwrap() {
            let name_entry = name_entry.unwrap();
            let name = name_entry.file_name().to_string_lossy().to_string();
            for version_entry in fs::read_dir(name_entry.path()).unwrap() {
                let version_entry = version_entry.unwrap();
                let version = version_entry.file_name().to_string_lossy().to_string();
                let version_path = version_entry.path();
                collect_files(f, &namespace, &name, &version, &version_path, &version_path, ext, macro_name);
            }
        }
    }
}

fn collect_files(
    f: &mut fs::File,
    namespace: &str,
    name: &str,
    version: &str,
    base: &Path,
    dir: &Path,
    ext: &str,
    macro_name: &str,
) {
    for entry in fs::read_dir(dir).unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();
        if path.is_dir() {
            collect_files(f, namespace, name, version, base, &path, ext, macro_name);
        } else if path.extension().map_or(false, |e| e == ext)
            || (ext == "typ" && path.file_name().map_or(false, |n| n == "typst.toml"))
        {
            let rel = path.strip_prefix(base).unwrap();
            let rel_str = rel.to_string_lossy().replace('\\', "/");
            let abs = path.canonicalize().unwrap();
            let abs_str = abs.to_string_lossy().replace('\\', "/");
            writeln!(
                f,
                "        (\"{namespace}\", \"{name}\", \"{version}\", \"{rel_str}\", {macro_name}(\"{abs_str}\")),",
            )
            .unwrap();
        }
    }
}
