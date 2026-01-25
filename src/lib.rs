#[macro_use]
extern crate lazy_static;
extern crate num_complex;
extern crate pest;
extern crate pest_derive;

pub mod ast;
pub mod codegen;
pub mod optimizer;
pub mod parser;
pub mod typing;

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

            // 1. AST生成
            let ast = parser::parse_to_ast(expression.into_inner());

            // 2. 最適化
            let optimized_ast = optimizer::optimize_ast(&ast);

            // 3. 型解決
            let typed_ast = typing::type_check(&optimized_ast);

            // 4. コード生成 (GlslBackend)
            codegen::generate_code(&typed_ast, &codegen::GlslBackend)
        }
        Err(e) => {
            // パースエラーの場合は、その内容を文字列として返す
            format!("Parse Error:\n{}", e)
        }
    }
}

#[wasm_bindgen]
pub fn get_glsl_output_dd(input: &str) -> String {
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

            // 1. AST生成
            let ast = parser::parse_to_ast(expression.into_inner()); // parserモジュールに移動したと仮定

            // 2. 最適化 (定数畳み込みなど)
            let optimized_ast = optimizer::optimize_ast(&ast);

            // 3. 型チェック & 型付与 (新規)
            let typed_ast = typing::type_check(&optimized_ast);

            // 4. コード生成 (Backend指定)
            codegen::generate_code(&typed_ast, &codegen::GlslDoubleBackend)
        }
        Err(e) => {
            format!("Parse Error:\n{}", e)
        }
    }
}

#[wasm_bindgen]
pub fn get_wgsl_output(input: &str) -> String {
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

            let ast = parser::parse_to_ast(expression.into_inner());
            let optimized_ast = optimizer::optimize_ast(&ast);
            let typed_ast = typing::type_check(&optimized_ast); // 型解決

            // WGSL Backend を指定
            codegen::generate_code(&typed_ast, &codegen::WgslBackend)
        }
        Err(e) => format!("Parse Error:\n{}", e),
    }
}

#[wasm_bindgen]
pub fn get_wgsl_output_dd(input: &str) -> String {
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
            let ast = parser::parse_to_ast(expression.into_inner());
            let optimized_ast = optimizer::optimize_ast(&ast);
            let typed_ast = typing::type_check(&optimized_ast); // 型解決

            // WGSL Double Backend を使用
            codegen::generate_code(&typed_ast, &codegen::WgslDoubleBackend)
        }
        Err(e) => format!("Parse Error:\n{}", e),
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
        let input = "sin(real(z))";
        let output = get_glsl_output(input);
        assert_eq!(output, "sin((z).x)");
    }

    #[test]
    fn test_dd_variable() {
        // Use variable to ensure dcadd is generated
        let input = "z + 2.0i";
        let output = get_glsl_output_dd(input);
        assert!(output.contains("dcadd"));
    }

    #[test]
    fn test_type_resolution_abs_dd() {
        // 仕様: abs(z) は実数(vec2)を返すべき
        // 現状の generate_glsl_dd は mat2(...) でラップしてしまうため、これが失敗するはずです
        // 期待: mat2ラップなしの "dcabs(z)"
        let input = "abs(z)";
        let output = get_glsl_output_dd(input);

        // 現状の実装だと mat2(dcabs(z), vec2(0.0)) のような形になっている
        // 将来的に型システムが入れば、ここは単なる dcabs(z) になるべき
        assert_eq!(output, "dcabs(z)");
    }

    #[test]
    fn test_mixed_arithmetic_dd() {
        // 仕様: 実数 + 複素数 の場合、実数側が複素数にキャストされる
        // input: abs(z) + z
        // abs(z) は Real(vec2), z は Complex(mat2)
        // 期待: dcadd( to_complex(dcabs(z)), z )  ※to_complexは仮のキャスト関数名
        // または、インライン展開で dcadd( mat2(dcabs(z), vec2(0.0)), z )

        let input = "abs(z) + z";
        let output = get_glsl_output_dd(input);

        // 少なくとも dcadd は使われる
        assert!(output.contains("dcadd"));
        // キャストの痕跡（mat2生成）が abs(z) の周りにあるか確認
        // (現状は常にmat2なので偶然通るかもしれませんが、意図としては「必要な時だけキャスト」です)
    }

    #[test]
    fn test_pure_real_multiplication_dd() {
        // 仕様: 実数同士の掛け算は、複素数乗算(dcmul)ではなく実数乗算(dmul)になるべき
        // input: abs(z) * 2.0
        // 期待: dmul(dcabs(z), ...)

        let input = "abs(z) * 2.0";
        let output = get_glsl_output_dd(input);

        // dcmul (複素数乗算) が使われていたら、非効率なので失敗とみなす
        assert!(output.contains("dmul"));
        assert!(!output.contains("dcmul"));
    }
    #[test]
    fn test_dd_type_resolution_abs() {
        // 仕様: abs(z) は実数(vec2)を返すべき
        // 期待: mat2(...) でラップされず、生の "dcabs(z)" が返る
        let input = "abs(z)";
        let output = get_glsl_output_dd(input);
        assert_eq!(output, "dcabs(z)");
    }

    #[test]
    fn test_dd_mixed_arithmetic() {
        // 仕様: 複素数(z) + 実数(abs(z))
        // 実数側が明示的、あるいは暗黙的に複素数(mat2)にキャストされる必要がある
        // 期待: dcadd(z, <cast>(dcabs(z))) のような形式
        // ここでは、キャスト用のコンストラクタ "mat2(" が含まれているかを確認
        let input = "z + abs(z)";
        let output = get_glsl_output_dd(input);

        assert!(output.contains("dcadd"));
        // dcabs(z)の結果(vec2)をmat2に変換しているはず
        assert!(output.contains("mat2(dcabs(z)"));
    }

    #[test]
    fn test_dd_pure_real_opt() {
        // 仕様: 実数同士の乗算 abs(z) * 2.0
        // 期待: 複素数乗算 dcmul ではなく、実数乗算 dmul が使われる
        let input = "abs(z) * 2.0";
        let output = get_glsl_output_dd(input);

        assert!(output.contains("dmul"));
        assert!(!output.contains("dcmul"));
    }

    #[test]
    fn test_wgsl_basic_arithmetic() {
        let input = "abs(z) + abs(z)";
        let output = get_wgsl_output(input);

        // abs -> length に変換され、かつ + で結合されているか
        assert!(output.contains("length(z)"));
        assert!(output.contains("+"));

        // 複素数の足し算: そのまま + (vec2対応)
        let input = "z + z"; // zは変数なのでComplex
        let output = get_wgsl_output(input);
        assert!(output.contains("z + z"));
    }

    #[test]
    fn test_wgsl_complex_multiplication() {
        // 複素数の掛け算: cmul関数になるべき (* ではない)
        let input = "z * z";
        let output = get_wgsl_output(input);
        assert!(output.contains("cmul(z, z)"));
        assert!(!output.contains("z * z")); // * は成分ごとの掛け算になってしまうためNG
    }

    #[test]
    fn test_wgsl_abs_and_cast() {
        // abs(z) -> length(z) (実数)
        // length(z) + z -> 実数+複素数 -> vec2<f32>(length(z), 0.0) + z

        let input = "abs(z) + z";
        let output = get_wgsl_output(input);

        assert!(output.contains("length(z)")); // abs -> length
        assert!(output.contains("vec2<f32>(")); // キャスト発生
    }

    #[test]
    fn test_wgsl_component_access() {
        let input = "real(z) + imag(z)";
        let output = get_wgsl_output(input);
        // WGSLでは .x, .y アクセスになる
        assert!(output.contains(".x"));
        assert!(output.contains(".y"));
    }

    #[test]
    fn test_wgsl_dd_literal_split() {
        // 円周率のような循環小数を入力して、Hi/Loに分割されているか確認
        // PI = 3.14159265...
        // Hi (f32) = 3.1415927
        // Lo (f32) = -8.74...e-8 (残差)

        let input = "3.141592653589793";
        let output = get_wgsl_output_dd(input);

        // vec2<f32>(Hi, Lo) の形式になっているか
        assert!(output.contains("vec2<f32>("));
        // Hiパートのおおよその値が含まれているか
        assert!(output.contains("3.14159"));
        // 0.0 ではない Loパートが含まれているか (分割が機能しているか)
        assert!(!output.contains(", 0.0)"));
    }

    #[test]
    fn test_wgsl_dd_complex_struct() {
        // 複素数が mat2x2<f32> になっているか
        let input = "1.0 + 2.0i";
        let output = get_wgsl_output_dd(input);

        assert!(output.contains("mat2x2<f32>"));
    }

    #[test]
    fn test_wgsl_dd_function_call() {
        let input = "sin(real(z))";
        let output = get_wgsl_output_dd(input);
        assert_eq!(output, "dsin((z[0]))");
    }
}
