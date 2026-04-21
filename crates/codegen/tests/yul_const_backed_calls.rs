use common::InputDb;
use driver::DriverDataBase;
use fe_codegen::{emit_module_yul, emit_test_module_yul};
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
    emit_module_yul(&db, top_mod).expect("Yul emission should succeed")
}

fn emit_inline_test_yul(path: &str, src: &str) -> String {
    let mut db = DriverDataBase::default();
    let file_url = Url::parse(path).unwrap();
    db.workspace()
        .touch(&mut db, file_url.clone(), Some(src.to_owned()));
    let file = db
        .workspace()
        .get(&db, &file_url)
        .expect("file should be loaded");
    let top_mod = db.top_mod(file);
    let output =
        emit_test_module_yul(&db, top_mod, None).expect("Yul test emission should succeed");
    assert_eq!(output.tests.len(), 1, "fixture should yield one test");
    output.tests[0].yul.clone()
}

fn function_body<'a>(yul: &'a str, signature: &str) -> &'a str {
    let start = yul
        .find(signature)
        .unwrap_or_else(|| panic!("expected function `{signature}` in emitted Yul:\n{yul}"));
    let tail = &yul[start..];
    let end = tail
        .find("\n      function ")
        .or_else(|| tail.find("\n    function "))
        .unwrap_or(tail.len());
    &tail[..end]
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
    let src = format!(
        "{}\nfn emit_helpers() -> u256 {{\n    let arr: [u256; 8] = [1, 2, 3, 4, 5, 6, 7, 8]\n    sum_last4(arr) + sum_first4(arr)\n}}\n",
        include_str!("../../fe/tests/fixtures/fe_test/view_param_local_ref_take_reverse.fe"),
    );
    let yul = emit_inline_yul(
        "file:///readonly_view_params_stay_code_backed_through_transparent_private_helpers.fe",
        &src,
    );
    let sum_start = yul
        .find("function $sum_first4_arg0_root_code(")
        .expect("expected code-specialized sum_first4 helper in emitted Yul");
    let sum_end = yul[sum_start + 1..]
        .find("\n    function ")
        .map(|idx| sum_start + 1 + idx)
        .unwrap_or(yul.len());
    let sum_first4 = &yul[sum_start..sum_end];

    assert!(
        sum_first4.contains("$take_u256_") && sum_first4.contains("_arg1_root_code("),
        "readonly array params should stay code-backed through transparent private helpers:\n{sum_first4}"
    );
    assert!(
        !sum_first4.contains("datacopy(")
            && sum_first4
                .lines()
                .any(|line| line.contains("$len_") && line.contains("_arg0_f1_code("))
            && sum_first4
                .lines()
                .any(|line| line.contains("$get_") && line.contains("_arg0_f1_code("))
            && !sum_first4
                .lines()
                .any(|line| line.contains("$len_") && !line.contains("_arg0_f1_code("))
            && !sum_first4.lines().any(|line| line.contains("$get_")
                && line.contains("(v2,")
                && !line.contains("_arg0_f1_code(")),
        "readonly array params should not materialize whole arrays at transparent helper boundaries:\n{sum_first4}"
    );

    let sum_last4 = function_body(&yul, "function $sum_last4_arg0_root_code(");
    assert!(
        sum_last4.contains("$take_u256_") && sum_last4.contains("_arg1_f0_code("),
        "readonly array params should preserve nested code-backed refs through wrapper materialization:\n{sum_last4}"
    );
    assert!(
        sum_last4.contains("$len_")
            && sum_last4.contains("_arg0_f1_f0_code(")
            && sum_last4.contains("$get_")
            && sum_last4.contains("_arg0_f1_f0_code("),
        "nested transparent helpers should see Take.base.Reverse.base as code-backed:\n{sum_last4}"
    );
}

#[test]
fn transparent_wrapper_get_loads_base_handle_before_inner_get() {
    let src = format!(
        "{}\nfn emit_helpers() -> u256 {{\n    let arr: [u256; 8] = [1, 2, 3, 4, 5, 6, 7, 8]\n    sum_last4(arr)\n}}\n",
        include_str!("../../fe/tests/fixtures/fe_test/view_param_local_ref_take_reverse.fe"),
    );
    let yul = emit_inline_yul(
        "file:///transparent_wrapper_get_loads_base_handle_before_inner_get.fe",
        &src,
    );
    let get_reverse = function_body(&yul, "function $get_0_arg0_f0_code(p0, p1) -> ret {");

    assert!(
        get_reverse.contains("mload(p0)") && get_reverse.contains("$get_"),
        "transparent wrapper get should load the wrapper base handle before delegating to the inner sequence get:\n{get_reverse}"
    );
}

#[test]
fn const_backed_constructor_args_stay_code_backed_until_abi_encoding() {
    let yul = emit_inline_test_yul(
        "file:///const_backed_constructor_args_stay_code_backed_until_abi_encoding.fe",
        include_str!("../../fe/tests/fixtures/fe_test/contract_init_fixed_array_arg.fe"),
    );
    let create2_signature = if yul.contains("function $create2_arg") {
        "function $create2_arg"
    } else {
        "function $create2("
    };
    let create2_helper = function_body(&yul, create2_signature);

    assert!(
        !(create2_helper.contains("datacopy(") && create2_helper.contains(", $args, 128)")),
        "const-backed constructor args should not materialize eagerly before ABI encoding:\n{create2_helper}"
    );
    assert!(
        create2_helper.contains("$encode_") && create2_helper.contains("_arg0_root_code(p2)"),
        "constructor args should flow directly into ABI encoding from their code-backed carrier:\n{create2_helper}"
    );
    let mut tail = yul.as_str();
    while let Some(start) = tail.find("function $") {
        let function_tail = &tail[start..];
        let end = function_tail
            .find("\n      function ")
            .or_else(|| function_tail.find("\n    function "))
            .unwrap_or(function_tail.len());
        let body = &function_tail[..end];
        let signature = body.lines().next().unwrap_or_default();
        if signature.contains("_arg0_root_code(")
            && body
                .lines()
                .any(|line| line.contains("$encode_field_") && !line.contains("_arg0_root_code("))
        {
            assert!(
                body.contains("datacopy("),
                "code-backed aggregate transport should not be passed into plain field encoding without first materializing a scalar:\n{body}"
            );
        }
        tail = &function_tail[end..];
    }
}

#[test]
fn code_backed_storage_array_copy_stages_code_before_storage_writes() {
    let yul = emit_inline_test_yul(
        "file:///code_backed_storage_array_copy_stages_code_before_storage_writes.fe",
        include_str!("../../fe/tests/fixtures/fe_test/code_backed_array_storage_copy.fe"),
    );
    let init = function_body(&yul, "function $__CodeArrayStore_init_eff0_stor(");

    assert!(
        init.contains("datacopy("),
        "code-backed aggregate storage copy should stage through memory before loading words:\n{init}"
    );
    assert!(
        !init.contains("mload(dataoffset("),
        "Yul cannot load code-space constants with mload(dataoffset(...)); use datacopy first:\n{init}"
    );
    assert!(
        init.contains("sstore(add(p0, 1),")
            && init.contains("sstore(add(p0, 2),")
            && init.contains("sstore(add(p0, 3),"),
        "aggregate storage copies should address word slots, not byte offsets:\n{init}"
    );
    assert!(
        !init.contains("sstore(add(p0, 32),"),
        "aggregate storage copies should not write byte offsets into storage slots:\n{init}"
    );
}
