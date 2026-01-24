use num_complex::Complex;
use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum DataType {
    Real,    // GLSL: float, Double: vec2
    Complex, // GLSL: vec2,  Double: mat2
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    Pos,
    Neg,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
}

#[derive(Debug, Clone)]
pub enum AstNode {
    ComplexNumber(Complex<f64>),
    Variable(String),
    FunctionCall {
        name: String,
        args: Vec<AstNode>,
    },
    UnaryOp {
        op: UnaryOp,
        child: Box<AstNode>,
    },
    BinaryOp {
        op: BinaryOp,
        lhs: Box<AstNode>,
        rhs: Box<AstNode>,
    },
}

impl fmt::Display for AstNode {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            AstNode::ComplexNumber(c) => {
                let (re, im) = (c.re, c.im);
                if im.abs() < 1e-10 {
                    write!(f, "{}", re)
                } else if re.abs() < 1e-10 {
                    write!(f, "{}i", im)
                } else {
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
                    UnaryOp::Pos => "pos",
                    UnaryOp::Neg => "neg",
                };
                write!(f, "{}({})", op_str, child)
            }
            AstNode::BinaryOp { op, lhs, rhs } => {
                let op_str = match op {
                    BinaryOp::Add => "add",
                    BinaryOp::Sub => "sub",
                    BinaryOp::Mul => "mul",
                    BinaryOp::Div => "div",
                };
                write!(f, "{}({},{})", op_str, lhs, rhs)
            }
        }
    }
}
