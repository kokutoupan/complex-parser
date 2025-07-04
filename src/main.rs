#[macro_use]
extern crate lazy_static;
extern crate pest;
extern crate pest_derive;

use num_complex::Complex;

use pest::Parser;
use pest::iterators::{Pair, Pairs};
use pest_derive::Parser;
// use num_complex::Complex;
use pest::pratt_parser::{Op, PrattParser};

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
            Rule::variable => {
                let name = primary.as_str();
                if name == "i" {
                    AstNode::ComplexNumber(Complex::i())
                } else {
                    AstNode::Variable(name.to_string())
                }
            }
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
                    // absは実数を返すので、虚部0の複素数としてラップする
                    ("abs", &[c]) => AstNode::ComplexNumber(Complex::new(c.norm(), 0.0)),
                    // 複素数特有の関数
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

fn main() {
    let inputs = vec![
        "5i * 5i",
        "max(2 * 5, 12)",
        "sin(0) + cos(0)",
        "foo(x + 1, 2, y)",
        "sqrt(abs(-16))",
    ];

    for input in inputs {
        println!("--------------------");
        match MyComplex::parse(Rule::calculation, input) {
            Ok(mut pairs) => {
                // `find`は `calculation` の子要素から `expr` を探す
                let expression = pairs
                    .next()
                    .unwrap()
                    .into_inner()
                    .find(|p| p.as_rule() == Rule::expr)
                    .unwrap();
                // `expr` の子要素を `parse_to_ast` に渡す
                let ast = parse_to_ast(expression.into_inner());

                println!("Input:         {}", input);
                println!("Original AST:  {}", ast);

                let optimized_ast = optimize_ast(&ast);
                println!("Optimized AST: {}", optimized_ast);
            }
            Err(e) => {
                println!("Parse failed for '{}':\n{}", input, e);
            }
        }
    }
}
