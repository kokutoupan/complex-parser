use crate::ast::{AstNode, BinaryOp, UnaryOp};
use num_complex::Complex;
use pest::iterators::Pairs;
use pest::pratt_parser::{Op, PrattParser};
use pest_derive::Parser;

#[derive(Parser)]
#[grammar = "complex.pest"]
pub struct MyComplex;

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

pub fn parse_to_ast(pairs: Pairs<Rule>) -> AstNode {
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
        .map_prefix(|op, rhs| {
            let op = match op.as_rule() {
                Rule::pos_op => UnaryOp::Pos,
                Rule::neg_op => UnaryOp::Neg,
                _ => unreachable!(),
            };
            AstNode::UnaryOp {
                op,
                child: Box::new(rhs),
            }
        })
        .map_infix(|lhs, op, rhs| {
            let op = match op.as_rule() {
                Rule::add_op => BinaryOp::Add,
                Rule::sub_op => BinaryOp::Sub,
                Rule::mul_op => BinaryOp::Mul,
                Rule::div_op => BinaryOp::Div,
                _ => unreachable!(),
            };
            AstNode::BinaryOp {
                op,
                lhs: Box::new(lhs),
                rhs: Box::new(rhs),
            }
        })
        .parse(pairs)
}
