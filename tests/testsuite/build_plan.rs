use support::registry::Package;
use support::{basic_bin_manifest, basic_manifest, main_file, project};

#[test]
fn cargo_build_plan_simple() {
    let p = project()
        .file("Cargo.toml", &basic_bin_manifest("foo"))
        .file("src/foo.rs", &main_file(r#""i am foo""#, &[]))
        .build();

    p.cargo("build --build-plan -Zunstable-options")
        .masquerade_as_nightly_cargo()
        .with_json(
            r#"
    {
        "inputs": [
            "[..]/foo/Cargo.toml"
        ],
        "invocations": [
            {
                "args": "{...}",
                "cwd": "[..]/cit/[..]/foo",
                "deps": [],
                "env": "{...}",
                "kind": "Host",
                "links": "{...}",
                "outputs": "{...}",
                "package_name": "foo",
                "package_version": "0.5.0",
                "program": "rustc",
                "target_kind": ["bin"],
                "compile_mode": "build"
            }
        ]
    }
    "#,
        ).run();
    assert!(!p.bin("foo").is_file());
}

#[test]
fn cargo_build_plan_single_dep() {
    let p = project()
        .file(
            "Cargo.toml",
            r#"
            [package]
            name = "foo"
            authors = []
            version = "0.5.0"

            [dependencies]
            bar = { path = "bar" }
        "#,
        ).file(
            "src/lib.rs",
            r#"
            extern crate bar;
            pub fn foo() { bar::bar(); }

            #[test]
            fn test() { foo(); }
        "#,
        ).file("bar/Cargo.toml", &basic_manifest("bar", "0.0.1"))
        .file("bar/src/lib.rs", "pub fn bar() {}")
        .build();
    p.cargo("build --build-plan -Zunstable-options")
        .masquerade_as_nightly_cargo()
        .with_json(
            r#"
    {
        "inputs": [
            "[..]/foo/Cargo.toml",
            "[..]/foo/bar/Cargo.toml"
        ],
        "invocations": [
            {
                "args": "{...}",
                "cwd": "[..]/cit/[..]/foo",
                "deps": [],
                "env": "{...}",
                "kind": "Host",
                "links": "{...}",
                "outputs": [
                    "[..]/foo/target/debug/deps/libbar-[..].rlib"
                ],
                "package_name": "bar",
                "package_version": "0.0.1",
                "program": "rustc",
                "target_kind": ["lib"],
                "compile_mode": "build"
            },
            {
                "args": "{...}",
                "cwd": "[..]/cit/[..]/foo",
                "deps": [0],
                "env": "{...}",
                "kind": "Host",
                "links": "{...}",
                "outputs": [
                    "[..]/foo/target/debug/deps/libfoo-[..].rlib"
                ],
                "package_name": "foo",
                "package_version": "0.5.0",
                "program": "rustc",
                "target_kind": ["lib"],
                "compile_mode": "build"
            }
        ]
    }
    "#,
        ).run();
}

#[test]
fn cargo_build_plan_build_script() {
    let p = project()
        .file(
            "Cargo.toml",
            r#"
            [project]

            name = "foo"
            version = "0.5.0"
            authors = ["wycats@example.com"]
            build = "build.rs"
        "#,
        ).file("src/main.rs", r#"fn main() {}"#)
        .file("build.rs", r#"fn main() {}"#)
        .build();

    p.cargo("build --build-plan -Zunstable-options")
        .masquerade_as_nightly_cargo()
        .with_json(
            r#"
    {
        "inputs": [
            "[..]/foo/Cargo.toml"
        ],
        "invocations": [
            {
                "args": "{...}",
                "cwd": "[..]/cit/[..]/foo",
                "deps": [],
                "env": "{...}",
                "kind": "Host",
                "links": "{...}",
                "outputs": [
                    "[..]/foo/target/debug/build/[..]/build_script_build-[..]"
                ],
                "package_name": "foo",
                "package_version": "0.5.0",
                "program": "rustc",
                "target_kind": ["custom-build"],
                "compile_mode": "build"
            },
            {
                "args": "{...}",
                "cwd": "[..]/cit/[..]/foo",
                "deps": [0],
                "env": "{...}",
                "kind": "Host",
                "links": "{...}",
                "outputs": [],
                "package_name": "foo",
                "package_version": "0.5.0",
                "program": "[..]/build-script-build",
                "target_kind": ["custom-build"],
                "compile_mode": "run-custom-build"
            },
            {
                "args": "{...}",
                "cwd": "[..]/cit/[..]/foo",
                "deps": [1],
                "env": "{...}",
                "kind": "Host",
                "links": "{...}",
                "outputs": "{...}",
                "package_name": "foo",
                "package_version": "0.5.0",
                "program": "rustc",
                "target_kind": ["bin"],
                "compile_mode": "build"
            }
        ]
    }
    "#,
        ).run();
}

#[test]
fn build_plan_with_dev_dep() {
    Package::new("bar", "0.1.0").publish();

    let p = project()
        .file(
            "Cargo.toml",
            r#"
                [project]
                name = "foo"
                version = "0.5.0"
                authors = []

                [dev-dependencies]
                bar = "*"
            "#,
        )
        .file("src/lib.rs", "")
        .build();

    p.cargo("build --build-plan -Zunstable-options")
        .masquerade_as_nightly_cargo()
        .run();
}
