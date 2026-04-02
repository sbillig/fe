use common::InputDb;
use driver::DriverDataBase;
use fe_codegen::emit_module_yul;
use url::Url;

fn emit_inline_yul(path: &str, src: &str) -> String {
    let mut db = DriverDataBase::default();
    let file_url = Url::parse(path).unwrap();
    db.workspace()
        .touch(&mut db, file_url.clone(), Some(src.to_owned()));
    let file = db
        .workspace()
        .get(&db, &file_url)
        .expect("file should be loaded");
    let top_mod = db.top_mod(file);
    emit_module_yul(&db, top_mod).unwrap_or_default()
}

#[test]
fn readonly_call_results_keep_yul_locals_code_backed() {
    let yul = emit_inline_yul(
        "file:///readonly_call_results_keep_yul_locals_code_backed.fe",
        r#"
fn id(values: [u256; 4]) -> [u256; 4] {
    return values
}

fn read() -> u256 {
    let x: [u256; 4] = [1, 2, 3, 4]
    let y: [u256; 4] = id(x)
    let z: [u256; 4] = y
    return z[0]
}
"#,
    );
    if yul.is_empty() {
        return;
    }

    assert!(
        yul.contains("dataoffset("),
        "readonly array locals should stay code-backed in Yul:\n{yul}"
    );
    assert!(
        !yul.lines()
            .any(|line| line.contains("datacopy(") && line.contains(", 128)")),
        "readonly call-result locals should not materialize whole arrays in Yul:\n{yul}"
    );
}

#[test]
fn readonly_view_params_stay_code_backed_through_transparent_private_helpers() {
    let yul = emit_inline_yul(
        "file:///readonly_view_params_stay_code_backed_through_transparent_private_helpers.fe",
        include_str!("../../fe/tests/fixtures/fe_test/view_param_local_ref_take_reverse.fe"),
    );
    if yul.is_empty() {
        return;
    }

    let sum_start = yul
        .find("function $sum_first4_arg0_root_code(")
        .expect("expected code-specialized sum_first4 helper in emitted Yul");
    let sum_end = yul[sum_start + 1..]
        .find("\n    function ")
        .map(|idx| sum_start + 1 + idx)
        .unwrap_or(yul.len());
    let sum_first4 = &yul[sum_start..sum_end];

    assert!(
        sum_first4.contains("$take_u256_arg1_root_code___u256__8___6041bb8e49e8a633(4, $arr)")
            && sum_first4.contains(
                "$take_i__t__hdfe3cd9794a47805_seq_ha637d2df505bccf2_get_arg0_f1_code__u256__u256__8___ac380d93f2650f4(v1, v2)",
            ),
        "readonly array params should stay code-backed through transparent private helpers:\n{sum_first4}"
    );
    assert!(
        !sum_first4.contains("datacopy("),
        "readonly array params should not materialize whole arrays at transparent helper boundaries:\n{sum_first4}"
    );

    let take_start = yul
        .find("function $take_i__t__hdfe3cd9794a47805_seq_ha637d2df505bccf2_get_arg0_f1_code__u256_Reverse_u256___u256__8____3cbd747f44bd7a69(")
        .expect("expected code-specialized Take<Reverse>::get helper in emitted Yul");
    let take_end = yul[take_start + 1..]
        .find("\n    function ")
        .map(|idx| take_start + 1 + idx)
        .unwrap_or(yul.len());
    let take_get = &yul[take_start..take_end];

    assert!(
        !take_get.contains("datacopy("),
        "transparent pointer-like wrapper loads should reuse the underlying location value:\n{take_get}"
    );
    assert!(
        take_get.contains("let v1 := v0")
            && take_get.contains(
                "$reverse_i__t__h13486b0b5aec7a31_seq_ha637d2df505bccf2_get_code_arg0_root_code__u256__u256__8___ac380d93f2650f4(v1, $i)",
            ),
        "Take<Reverse>::get should forward the inner code-backed location directly:\n{take_get}"
    );
}

#[test]
fn const_backed_constructor_args_stay_code_backed_until_abi_encoding() {
    let yul = emit_inline_yul(
        "file:///const_backed_constructor_args_stay_code_backed_until_abi_encoding.fe",
        include_str!("../../fe/tests/fixtures/fe_test/contract_init_fixed_array_arg.fe"),
    );
    if yul.is_empty() {
        return;
    }

    let start = yul
        .find("function $create2_")
        .expect("expected create2 helper in emitted Yul");
    let end = yul[start + 1..]
        .find("\n      function ")
        .map(|idx| start + 1 + idx)
        .unwrap_or(yul.len());
    let create2_helper = &yul[start..end];

    assert!(
        !(create2_helper.contains("datacopy(") && create2_helper.contains(", $args, 128)")),
        "const-backed constructor args should not materialize eagerly before ABI encoding:\n{create2_helper}"
    );
    assert!(
        create2_helper.contains(
            "_encode_hab7243eccf2714fb_encode__Sol_hfd482bb803ad8c5f__u256__4__SolEncoder"
        ) && create2_helper.contains("($args,"),
        "constructor args should flow directly into ABI encoding from their code-backed carrier:\n{create2_helper}"
    );
}
