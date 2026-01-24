#[macro_use]
extern crate lazy_static;
extern crate num_complex;
extern crate pest;
extern crate pest_derive;

pub mod ast;
pub mod codegen;
pub mod optimizer;
pub mod parser;

use codegen::{generate_glsl, generate_glsl_dd};
use optimizer::optimize_ast;
use parser::{MyComplex, Rule, parse_to_ast};
use pest::Parser;
use wasm_bindgen::prelude::*;

// --- WASMから呼び出される公開関数 ---

/// パニック時にコンソールに詳細なエラーを出力するための初期設定
#[wasm_bindgen]
pub fn set_panic_hook() {
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();
}

/// 数式文字列を受け取り、GLSLの式を返すメイン関数
#[wasm_bindgen]
pub fn get_glsl_output(input: &str) -> String {
    // 入力が空なら何も返さない
    if input.trim().is_empty() {
        return "".to_string();
    }

    match MyComplex::parse(Rule::calculation, input) {
        Ok(mut pairs) => {
            let expression = pairs
                .next()
                .unwrap()
                .into_inner()
                .find(|p| p.as_rule() == Rule::expr)
                .unwrap();
            let ast = parse_to_ast(expression.into_inner());
            let optimized_ast = optimize_ast(&ast);
            generate_glsl(&optimized_ast)
        }
        Err(e) => {
            // パースエラーの場合は、その内容を文字列として返す
            format!("Parse Error:\n{}", e)
        }
    }
}

#[wasm_bindgen]
pub fn get_glsl_output_dd(input: &str) -> String {
    // 入力が空なら何も返さない
    if input.trim().is_empty() {
        return "".to_string();
    }

    match MyComplex::parse(Rule::calculation, input) {
        Ok(mut pairs) => {
            let expression = pairs
                .next()
                .unwrap()
                .into_inner()
                .find(|p| p.as_rule() == Rule::expr)
                .unwrap();
            let ast = parse_to_ast(expression.into_inner());
            let optimized_ast = optimize_ast(&ast);
            generate_glsl_dd(&optimized_ast)
        }
        Err(e) => {
            // パースエラーの場合は、その内容を文字列として返す
            format!("Parse Error:\n{}", e)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_addition_optimized() {
        let input = "1.0 + 2.0i";
        let output = get_glsl_output(input);
        // Optimizer compiles this to a single vec2
        assert_eq!(output, "vec2(1.0, 2.0)");
    }

    #[test]
    fn test_multiplication_variable() {
        // Use a variable 'z' to prevent constant folding of the multiplication
        let input = "(1.0 + 1.0i) * z";
        let output = get_glsl_output(input);
        assert!(output.contains("cmul"));
    }

    #[test]
    fn test_function_call() {
        let input = "sin(z)";
        let output = get_glsl_output(input);
        assert_eq!(output, "csin(z)");
    }

    #[test]
    fn test_dd_variable() {
        // Use variable to ensure dcadd is generated
        let input = "z + 2.0i";
        let output = get_glsl_output_dd(input);
        assert!(output.contains("dcadd"));
    }
}
