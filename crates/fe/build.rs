fn main() {
    println!("cargo:rerun-if-changed=tests/fixtures");
    println!("cargo:rerun-if-changed=tests/doc_fixtures");
}
