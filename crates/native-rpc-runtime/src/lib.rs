use std::{env, ptr, slice, str};

use alloy_primitives::Address;
use serde_json::{Value, json};

const ERR_INVALID_ARG: i64 = -1;
const ERR_MISSING_URL: i64 = -2;
const ERR_INVALID_INPUT: i64 = -3;
const ERR_REQUEST: i64 = -4;
const ERR_RPC: i64 = -5;
const ERR_OUTPUT_TOO_SMALL: i64 = -6;

#[unsafe(no_mangle)]
/// # Safety
///
/// This function is called from generated native Fe code. It does not dereference
/// caller-provided pointers.
pub unsafe extern "C" fn __fe_rpc_eth_block_number_u64() -> i64 {
    match post_rpc("eth_blockNumber", json!([])).and_then(|result| parse_quantity(&result)) {
        Ok(value) => value,
        Err(status) => status,
    }
}

#[unsafe(no_mangle)]
/// # Safety
///
/// `out` must point to a writable buffer of at least `out_cap` bytes.
pub unsafe extern "C" fn __fe_rpc_eth_block_number(out: *mut u8, out_cap: usize) -> i64 {
    match post_rpc("eth_blockNumber", json!([]))
        .and_then(|result| write_output(out, out_cap, &result))
    {
        Ok(len) => len,
        Err(status) => status,
    }
}

#[unsafe(no_mangle)]
/// # Safety
///
/// `to_ptr`/`data_ptr` must point to readable UTF-8 buffers of `to_len` and
/// `data_len` bytes. `out` must point to a writable buffer of at least
/// `out_cap` bytes.
pub unsafe extern "C" fn __fe_rpc_eth_call(
    to_ptr: *const u8,
    to_len: usize,
    data_ptr: *const u8,
    data_len: usize,
    out: *mut u8,
    out_cap: usize,
) -> i64 {
    match eth_call(to_ptr, to_len, data_ptr, data_len)
        .and_then(|result| write_output(out, out_cap, &result))
    {
        Ok(len) => len,
        Err(status) => status,
    }
}

fn eth_call(
    to_ptr: *const u8,
    to_len: usize,
    data_ptr: *const u8,
    data_len: usize,
) -> Result<String, i64> {
    let to = input_str(to_ptr, to_len)?;
    let data = input_str(data_ptr, data_len)?;
    let _address: Address = to.parse().map_err(|_| ERR_INVALID_INPUT)?;
    validate_hex_data(data)?;
    post_rpc(
        "eth_call",
        json!([
            {
                "to": to,
                "data": data,
            },
            "latest"
        ]),
    )
}

fn input_str(ptr: *const u8, len: usize) -> Result<&'static str, i64> {
    if ptr.is_null() {
        return Err(ERR_INVALID_ARG);
    }
    unsafe { str::from_utf8(slice::from_raw_parts(ptr, len)) }.map_err(|_| ERR_INVALID_INPUT)
}

fn validate_hex_data(value: &str) -> Result<(), i64> {
    if value.len() < 2 || !value.starts_with("0x") || !value.len().is_multiple_of(2) {
        return Err(ERR_INVALID_INPUT);
    }
    if value[2..].bytes().all(|byte| byte.is_ascii_hexdigit()) {
        Ok(())
    } else {
        Err(ERR_INVALID_INPUT)
    }
}

fn parse_quantity(value: &str) -> Result<i64, i64> {
    let digits = value
        .strip_prefix("0x")
        .ok_or(ERR_INVALID_INPUT)?
        .trim_start_matches('0');
    let normalized = if digits.is_empty() { "0" } else { digits };
    i64::from_str_radix(normalized, 16).map_err(|_| ERR_RPC)
}

fn post_rpc(method: &str, params: Value) -> Result<String, i64> {
    let url = env::var("FE_ETH_RPC_URL").map_err(|_| ERR_MISSING_URL)?;
    let response = reqwest::blocking::Client::new()
        .post(url)
        .json(&json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": method,
            "params": params,
        }))
        .send()
        .map_err(|_| ERR_REQUEST)?;

    if !response.status().is_success() {
        return Err(ERR_RPC);
    }

    let payload: Value = response.json().map_err(|_| ERR_RPC)?;
    if payload.get("error").is_some() {
        return Err(ERR_RPC);
    }
    payload
        .get("result")
        .and_then(Value::as_str)
        .map(ToOwned::to_owned)
        .ok_or(ERR_RPC)
}

fn write_output(out: *mut u8, out_cap: usize, result: &str) -> Result<i64, i64> {
    if out.is_null() {
        return Err(ERR_INVALID_ARG);
    }
    if result.len() > out_cap {
        return Err(ERR_OUTPUT_TOO_SMALL);
    }
    unsafe {
        ptr::copy_nonoverlapping(result.as_ptr(), out, result.len());
    }
    Ok(result.len() as i64)
}
