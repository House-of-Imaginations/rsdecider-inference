//! Byte-exact port of Python `json.dumps(v, ensure_ascii=...)` with default separators (", ", ": ").
use serde_json::Value;
use std::fmt::Write;

pub fn dumps(v: &Value, ensure_ascii: bool) -> String {
    let mut out = String::new();
    write_value(&mut out, v, ensure_ascii);
    out
}

fn write_value(out: &mut String, v: &Value, ascii: bool) {
    match v {
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Number(n) => match (n.as_i64(), n.as_u64(), n.as_f64()) {
            (Some(i), _, _) => write!(out, "{i}").unwrap(),
            (_, Some(u), _) => write!(out, "{u}").unwrap(),
            (_, _, Some(f)) => out.push_str(&float_repr(f)),
            _ => unreachable!("serde_json numbers are i64, u64 or f64"),
        },
        Value::String(s) => write_str(out, s, ascii),
        Value::Array(a) => {
            out.push('[');
            for (i, x) in a.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                write_value(out, x, ascii);
            }
            out.push(']');
        }
        Value::Object(m) => {
            out.push('{');
            for (i, (k, x)) in m.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                write_str(out, k, ascii);
                out.push_str(": ");
                write_value(out, x, ascii);
            }
            out.push('}');
        }
    }
}

fn write_str(out: &mut String, s: &str, ascii: bool) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => write!(out, "\\u{:04x}", c as u32).unwrap(),
            c if ascii && !(' '..='~').contains(&c) => {
                let mut buf = [0u16; 2];
                for unit in c.encode_utf16(&mut buf) {
                    write!(out, "\\u{:04x}", unit).unwrap();
                }
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

/// Python `repr(float)`: shortest round-trip digits; scientific when the decimal exponent is < -4 or >= 16.
/// Precondition: f.is_finite()
pub(crate) fn float_repr(f: f64) -> String {
    debug_assert!(f.is_finite(), "float_repr needs a finite value, got {f}");
    let sci = format!("{:e}", f); // shortest round-trip, e.g. "-1.5e16", "1e-5", "0e0"
    let (mant, exp) = sci.split_once('e').unwrap();
    let exp: i32 = exp.parse().unwrap();
    let (sign, mant) = mant.strip_prefix('-').map_or(("", mant), |m| ("-", m));
    let digits: String = mant.chars().filter(|c| *c != '.').collect();
    let decpt = exp + 1; // value = 0.DIGITS * 10^decpt
    let body = if decpt <= -4 || decpt > 16 {
        let (first, rest) = digits.split_at(1);
        let frac = if rest.is_empty() { String::new() } else { format!(".{rest}") };
        let e = decpt - 1;
        format!("{first}{frac}e{}{:02}", if e < 0 { '-' } else { '+' }, e.abs())
    } else if decpt <= 0 {
        format!("0.{}{digits}", "0".repeat((-decpt) as usize))
    } else if decpt as usize >= digits.len() {
        format!("{digits}{}.0", "0".repeat(decpt as usize - digits.len()))
    } else {
        let (a, b) = digits.split_at(decpt as usize);
        format!("{a}.{b}")
    };
    format!("{sign}{body}")
}

#[cfg(test)]
mod tests {
    use super::*;

    // Expected strings produced by CPython 3 `json.dumps`.
    #[test]
    fn numbers_and_order_match_python() {
        let v: Value = serde_json::from_str(
            r#"{"amount": 1e16, "tiny": 1e-05, "ratio": 0.1, "whole": 100.0, "neg_zero": -0.0, "count": 3, "big": 12345678901234567890, "flag": true, "none": null, "x": 1.5e-7, "y": 123.456, "z": 1234567890123456.0, "w": 0.0001, "e": [], "o": {}}"#,
        )
        .unwrap();
        assert_eq!(
            dumps(&v, false),
            r#"{"amount": 1e+16, "tiny": 1e-05, "ratio": 0.1, "whole": 100.0, "neg_zero": -0.0, "count": 3, "big": 12345678901234567890, "flag": true, "none": null, "x": 1.5e-07, "y": 123.456, "z": 1234567890123456.0, "w": 0.0001, "e": [], "o": {}}"#
        );
    }

    #[test]
    fn float_repr_matches_python() {
        for (f, want) in [
            (1e16, "1e+16"),
            (1.5e16, "1.5e+16"),
            (1e15, "1000000000000000.0"),
            (1e-5, "1e-05"),
            (0.0001, "0.0001"),
            (0.1, "0.1"),
            (100.0, "100.0"),
            (123.456, "123.456"),
            (1234567890123456.0, "1234567890123456.0"),
            (12345678901234567.0, "1.2345678901234568e+16"),
            (-2.5e-10, "-2.5e-10"),
            (5e-324, "5e-324"),
            (1.7976931348623157e308, "1.7976931348623157e+308"),
            (3.0, "3.0"),
            (-0.0, "-0.0"),
        ] {
            assert_eq!(float_repr(f), want);
        }
    }

    #[test]
    fn strings_match_python_both_modes() {
        let v = serde_json::json!({"note": "Caf\u{e9} cr\u{e8}me \u{2014} \u{4e2d}\u{6587} \u{1F600} \"quoted\" back\\slash\ttab\u{7f}\u{1}"});
        assert_eq!(
            dumps(&v, false),
            "{\"note\": \"Caf\u{e9} cr\u{e8}me \u{2014} \u{4e2d}\u{6587} \u{1F600} \\\"quoted\\\" back\\\\slash\\ttab\u{7f}\\u0001\"}"
        );
        assert_eq!(
            dumps(&v, true),
            r#"{"note": "Caf\u00e9 cr\u00e8me \u2014 \u4e2d\u6587 \ud83d\ude00 \"quoted\" back\\slash\ttab\u007f\u0001"}"#
        );
    }
}
