#[macro_use]
extern crate lazy_static;
extern crate pest;
extern crate pest_derive;

use num_complex::Complex;

use pest::Parser;
use pest::iterators::Pairs;
use pest_derive::Parser;
// use num_complex::Complex;
use pest::pratt_parser::{Op, PrattParser};
use wasm_bindgen::prelude::*;

// Cloneトレイトを追加
#[derive(Debug, Clone)]
pub enum AstNode {
    ComplexNumber(Complex<f64>),
    Variable(String), // 変数名を保持するバリアントを追加
    FunctionCall {
        name: String,
        args: Vec<AstNode>,
    },
    UnaryOp {
        // 単項演算ノードを追加
        op: Rule,
        child: Box<AstNode>,
    },
    BinaryOp {
        op: Rule,
        lhs: Box<AstNode>,
        rhs: Box<AstNode>,
    },
}
use std::fmt;

impl fmt::Display for AstNode {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            AstNode::ComplexNumber(c) => {
                let (re, im) = (c.re, c.im);
                // 虚部が非常に小さい場合は実質的に実数とみなす
                if im.abs() < 1e-10 {
                    write!(f, "{}", re)
                }
                // 実部が非常に小さい場合は実質的に純虚数とみなす
                else if re.abs() < 1e-10 {
                    write!(f, "{}i", im)
                }
                // 一般的な複素数
                else {
                    if im > 0.0 {
                        write!(f, "({}+{}i)", re, im)
                    } else {
                        write!(f, "({}{}i)", re, im)
                    }
                }
            }
            AstNode::Variable(name) => write!(f, "{}", name),
            AstNode::FunctionCall { name, args } => {
                write!(f, "{}(", name)?;
                for (i, arg) in args.iter().enumerate() {
                    if i > 0 {
                        write!(f, ",")?;
                    }
                    write!(f, "{}", arg)?;
                }
                write!(f, ")")
            }
            AstNode::UnaryOp { op, child } => {
                let op_str = match op {
                    Rule::pos_op => "pos",
                    Rule::neg_op => "neg",
                    _ => unreachable!(),
                };
                write!(f, "{}({})", op_str, child)
            }
            AstNode::BinaryOp { op, lhs, rhs } => {
                let op_str = match op {
                    Rule::add_op => "add",
                    Rule::sub_op => "sub",
                    Rule::mul_op => "mul",
                    Rule::div_op => "div",
                    _ => unreachable!(),
                };
                write!(f, "{}({},{})", op_str, lhs, rhs)
            }
        }
    }
}
// ... (Pestのパーサー設定) ...
#[derive(Parser)]
#[grammar = "complex.pest"]
struct MyComplex;

lazy_static! {
    static ref PRATT_PARSER: PrattParser<Rule> = {
        use Rule::*;
        use pest::pratt_parser::Assoc::*;

        PrattParser::new()
            .op(Op::infix(add_op, Left) | Op::infix(sub_op, Left))
            .op(Op::infix(mul_op, Left) | Op::infix(div_op, Left))
            .op(Op::prefix(pos_op) | Op::prefix(neg_op))
    };
}

// parse_to_ast: 複素数リテラルを解釈
fn parse_to_ast(pairs: Pairs<Rule>) -> AstNode {
    PRATT_PARSER
        .map_primary(|primary| match primary.as_rule() {
            Rule::real_number => {
                AstNode::ComplexNumber(Complex::new(primary.as_str().parse().unwrap(), 0.0))
            }
            Rule::imaginary_number => {
                let s = primary.as_str();
                let val = s[..s.len() - 1].parse().unwrap();
                AstNode::ComplexNumber(Complex::new(0.0, val))
            }
            Rule::variable => match primary.as_str() {
                "i" => AstNode::ComplexNumber(Complex::i()),
                "pi" => AstNode::ComplexNumber(Complex::new(std::f64::consts::PI, 0.0)),
                "e" => AstNode::ComplexNumber(Complex::new(std::f64::consts::E, 0.0)),
                name => AstNode::Variable(name.to_string()),
            },
            Rule::function_call => {
                let mut inner = primary.into_inner();
                let name = inner.next().unwrap().as_str().to_string();
                let args = inner.map(|arg| parse_to_ast(arg.into_inner())).collect();
                AstNode::FunctionCall { name, args }
            }
            Rule::operand | Rule::expr => parse_to_ast(primary.into_inner()),
            rule => unreachable!("Primary expected, found {:?}", rule),
        })
        .map_prefix(|op, rhs| AstNode::UnaryOp {
            op: op.as_rule(),
            child: Box::new(rhs),
        })
        .map_infix(|lhs, op, rhs| AstNode::BinaryOp {
            op: op.as_rule(),
            lhs: Box::new(lhs),
            rhs: Box::new(rhs),
        })
        .parse(pairs)
}

// optimize_ast: 複素数演算に対応
fn optimize_ast(node: &AstNode) -> AstNode {
    match node {
        AstNode::ComplexNumber(_) | AstNode::Variable(_) => node.clone(),
        AstNode::FunctionCall { name, args } => {
            let optimized_args: Vec<AstNode> = args.iter().map(optimize_ast).collect();
            let all_args_are_numbers = optimized_args
                .iter()
                .all(|arg| matches!(arg, AstNode::ComplexNumber(_)));

            if all_args_are_numbers {
                let arg_values: Vec<Complex<f64>> = optimized_args
                    .into_iter()
                    .filter_map(|arg| {
                        if let AstNode::ComplexNumber(n) = arg {
                            Some(n)
                        } else {
                            None
                        }
                    })
                    .collect();

                match (name.as_str(), arg_values.as_slice()) {
                    ("sqrt", &[c]) => AstNode::ComplexNumber(c.sqrt()),
                    ("sin", &[c]) => AstNode::ComplexNumber(c.sin()),
                    ("cos", &[c]) => AstNode::ComplexNumber(c.cos()),
                    ("tan", &[c]) => AstNode::ComplexNumber(c.tan()),
                    ("exp", &[c]) => AstNode::ComplexNumber(c.exp()),
                    ("ln", &[c]) | ("log", &[c]) => AstNode::ComplexNumber(c.ln()),
                    ("asin", &[c]) | ("arcsin", &[c]) => AstNode::ComplexNumber(c.asin()),
                    ("acos", &[c]) | ("arccos", &[c]) => AstNode::ComplexNumber(c.acos()),
                    ("atan", &[c]) | ("arctan", &[c]) => AstNode::ComplexNumber(c.atan()),
                    ("pow", &[base, exponent]) => AstNode::ComplexNumber(base.powc(exponent)),
                    ("abs", &[c]) => AstNode::ComplexNumber(Complex::new(c.norm(), 0.0)),
                    ("real", &[c]) => AstNode::ComplexNumber(Complex::new(c.re, 0.0)),
                    ("imag", &[c]) => AstNode::ComplexNumber(Complex::new(c.im, 0.0)),
                    ("conj", &[c]) => AstNode::ComplexNumber(c.conj()),
                    _ => AstNode::FunctionCall {
                        name: name.clone(),
                        args: arg_values.into_iter().map(AstNode::ComplexNumber).collect(),
                    },
                }
            } else {
                AstNode::FunctionCall {
                    name: name.clone(),
                    args: optimized_args,
                }
            }
        }
        AstNode::UnaryOp { op, child } => {
            let optimized_child = optimize_ast(child);
            match op {
                Rule::pos_op => optimized_child,
                Rule::neg_op => match optimized_child {
                    AstNode::UnaryOp {
                        op: child_op,
                        child: grandchild,
                    } if child_op == Rule::neg_op => *grandchild,
                    AstNode::ComplexNumber(c) => AstNode::ComplexNumber(-c),
                    _ => AstNode::UnaryOp {
                        op: *op,
                        child: Box::new(optimized_child),
                    },
                },
                _ => unreachable!(),
            }
        }
        AstNode::BinaryOp { op, lhs, rhs } => {
            let optimized_lhs = optimize_ast(lhs);
            let optimized_rhs = optimize_ast(rhs);
            if let (AstNode::ComplexNumber(c1), AstNode::ComplexNumber(c2)) =
                (&optimized_lhs, &optimized_rhs)
            {
                let result = match op {
                    Rule::add_op => c1 + c2,
                    Rule::sub_op => c1 - c2,
                    Rule::mul_op => c1 * c2,
                    Rule::div_op => c1 / c2,
                    _ => unreachable!(),
                };
                AstNode::ComplexNumber(result)
            } else {
                AstNode::BinaryOp {
                    op: *op,
                    lhs: Box::new(optimized_lhs),
                    rhs: Box::new(optimized_rhs),
                }
            }
        }
    }
}

fn generate_glsl(node: &AstNode) -> String {
    match node {
        AstNode::ComplexNumber(c) => {
            // f64をGLSLのfloatリテラル文字列に変換するヘルパー関数
            let to_glsl_float = |val: f64| -> String {
                if val.abs() < 1e-9 {
                    return "0.0".to_string();
                }
                let s = format!("{}", val);
                if !s.contains('.') && !s.contains('e') {
                    format!("{}.0", s)
                } else {
                    s
                }
            };

            let re_str = to_glsl_float(c.re);
            let im_str = to_glsl_float(c.im);

            format!("vec2({}, {})", re_str, im_str)
        }
        AstNode::Variable(name) => name.clone(), // 変数名はそのまま使用
        AstNode::FunctionCall { name, args } => {
            let glsl_args: Vec<String> = args.iter().map(generate_glsl).collect();
            match (name.as_str(), glsl_args.as_slice()) {
                ("abs", [arg]) => format!("vec2(length({}), 0.0)", arg),
                ("real", [arg]) => format!("vec2(({}).x, 0.0)", arg),
                ("imag", [arg]) => format!("vec2(({}).y, 0.0)", arg),
                ("conj", [arg]) => format!("vec2(({}).x, -({}).y)", arg, arg),
                ("sqrt", [arg]) => format!("csqrt({})", arg),
                ("sin", [arg]) => format!("csin({})", arg),
                ("cos", [arg]) => format!("ccos({})", arg),
                ("tan", [arg]) => format!("ctan({})", arg),
                ("exp", [arg]) => format!("cexp({})", arg),
                ("ln", [arg]) | ("log", [arg]) => format!("clog({})", arg),
                ("asin", [arg]) | ("arcsin", [arg]) => format!("casin({})", arg),
                ("acos", [arg]) | ("arccos", [arg]) => format!("cacos({})", arg),
                ("atan", [arg]) | ("arctan", [arg]) => format!("catan({})", arg),
                ("pow", [base, exponent]) => format!("cpow({}, {})", base, exponent),
                _ => format!("{}({})", name, glsl_args.join(", ")),
            }
        }
        AstNode::UnaryOp { op, child } => {
            let child_glsl = generate_glsl(child);
            match op {
                Rule::neg_op => format!("(-{})", child_glsl),
                // 単項プラスは最適化で消えているはずだが、念のため
                Rule::pos_op => child_glsl,
                _ => unreachable!(),
            }
        }
        AstNode::BinaryOp { op, lhs, rhs } => {
            let lhs_glsl = generate_glsl(lhs);
            let rhs_glsl = generate_glsl(rhs);
            // 演算子の優先順位を考慮し、常に括弧で囲むのが安全
            match op {
                Rule::add_op => format!("({} + {})", lhs_glsl, rhs_glsl),
                Rule::sub_op => format!("({} - {})", lhs_glsl, rhs_glsl),
                // 複素数の乗算・除算はヘルパー関数を呼び出す
                Rule::mul_op => format!("cmul({}, {})", lhs_glsl, rhs_glsl),
                Rule::div_op => format!("cdiv({}, {})", lhs_glsl, rhs_glsl),
                _ => unreachable!(),
            }
        }
    }
}


fn generate_glsl_dd(node: &AstNode) -> String {
    match node {
        AstNode::ComplexNumber(c) => {
            // f64をGLSLのfloatリテラル文字列に変換するヘルパー関数
            let to_glsl_float = |val: f64| -> String {
                if val.abs() < 1e-9 {
                    return "0.0".to_string();
                }
                let s = format!("{}", val);
                if !s.contains('.') && !s.contains('e') {
                    format!("{}.0", s)
                } else {
                    s
                }
            };

            let re_str = to_glsl_float(c.re);
            let im_str = to_glsl_float(c.im);

            // 倍々精度複素数リテラル (mat2)として生成
            // d_real = vec2(val, 0.0), d_imag = vec2(val, 0.0)
            format!(
                "mat2(vec2({}, 0.0), vec2({}, 0.0))",
                re_str, im_str
            )
        }
        AstNode::Variable(name) => name.clone(), // 変数名はそのまま使用 (シェーダ側でmat2型と仮定)
        AstNode::FunctionCall { name, args } => {
            let glsl_args: Vec<String> = args.iter().map(generate_glsl_dd).collect();
            
            // 関数呼び出しを倍々精度複素数用の `dc...` 関数にマッピング
            match (name.as_str(), glsl_args.as_slice()) {
                // 実数値を返す関数は、結果を純粋な実複素数(mat2)に変換する
                ("abs", [arg]) => format!("mat2(dcabs({}), vec2(0.0))", arg),
                ("real", [arg]) => format!("mat2(({})[0], vec2(0.0))", arg),
                ("imag", [arg]) => format!("mat2(({})[1], vec2(0.0))", arg),
                
                // 複素数を返す関数
                ("conj", [arg]) => format!("dcconj({})", arg),
                ("sqrt", [arg]) => format!("dcsqrt({})", arg),
                ("sin", [arg]) => format!("dcsin({})", arg),
                ("cos", [arg]) => format!("dccos({})", arg),
                ("tan", [arg]) => format!("dctan({})", arg),
                ("exp", [arg]) => format!("dcexp({})", arg),
                ("ln", [arg]) | ("log", [arg]) => format!("dcln({})", arg),
                ("asin", [arg]) | ("arcsin", [arg]) => format!("dcasin({})", arg), // GLSL側に関数の実装が必要
                ("acos", [arg]) | ("arccos", [arg]) => format!("dcacos({})", arg), // GLSL側に関数の実装が必要
                ("atan", [arg]) | ("arctan", [arg]) => format!("dcatan({})", arg), // GLSL側に関数の実装が必要
                ("pow", [base, exponent]) => format!("dcpow({}, {})", base, exponent),
                
                // 不明な関数はそのまま出力
                _ => format!("{}({})", name, glsl_args.join(", ")),
            }
        }
        AstNode::UnaryOp { op, child } => {
            let child_glsl = generate_glsl_dd(child);
            match op {
                Rule::neg_op => format!("dcneg({})", child_glsl), // 単項マイナス
                Rule::pos_op => child_glsl, // 単項プラスは何もしない
                _ => unreachable!(),
            }
        }
        AstNode::BinaryOp { op, lhs, rhs } => {
            let lhs_glsl = generate_glsl_dd(lhs);
            let rhs_glsl = generate_glsl_dd(rhs);

            // 二項演算子を倍々精度複素数用の `dc...` 関数にマッピング
            match op {
                Rule::add_op => format!("dcadd({}, {})", lhs_glsl, rhs_glsl),
                Rule::sub_op => format!("dcsub({}, {})", lhs_glsl, rhs_glsl),
                Rule::mul_op => format!("dcmul({}, {})", lhs_glsl, rhs_glsl),
                Rule::div_op => format!("dcdiv({}, {})", lhs_glsl, rhs_glsl),
                _ => unreachable!(),
            }
        }
    }
}
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

    // これまでmain関数で行っていた処理を実行
    match MyComplex::parse(Rule::calculation, input) {
        Ok(mut pairs) => {
            let expression = pairs.next().unwrap().into_inner().find(|p| p.as_rule() == Rule::expr).unwrap();
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

    // これまでmain関数で行っていた処理を実行
    match MyComplex::parse(Rule::calculation, input) {
        Ok(mut pairs) => {
            let expression = pairs.next().unwrap().into_inner().find(|p| p.as_rule() == Rule::expr).unwrap();
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
