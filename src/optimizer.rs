use crate::ast::{AstNode, BinaryOp, UnaryOp};
use num_complex::Complex;

pub fn optimize_ast(node: &AstNode) -> AstNode {
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
                UnaryOp::Pos => optimized_child,
                UnaryOp::Neg => match optimized_child {
                    AstNode::UnaryOp {
                        op: child_op,
                        child: grandchild,
                    } if child_op == UnaryOp::Neg => *grandchild,
                    AstNode::ComplexNumber(c) => AstNode::ComplexNumber(-c),
                    _ => AstNode::UnaryOp {
                        op: *op,
                        child: Box::new(optimized_child),
                    },
                },
            }
        }
        AstNode::BinaryOp { op, lhs, rhs } => {
            let optimized_lhs = optimize_ast(lhs);
            let optimized_rhs = optimize_ast(rhs);
            if let (AstNode::ComplexNumber(c1), AstNode::ComplexNumber(c2)) =
                (&optimized_lhs, &optimized_rhs)
            {
                let result = match op {
                    BinaryOp::Add => c1 + c2,
                    BinaryOp::Sub => c1 - c2,
                    BinaryOp::Mul => c1 * c2,
                    BinaryOp::Div => c1 / c2,
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
