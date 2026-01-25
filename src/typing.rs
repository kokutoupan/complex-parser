use crate::ast::{AstNode, BinaryOp, UnaryOp}; // 変更: Rule を削除し Enum を追加
use num_complex::Complex;

#[derive(Debug, Clone, PartialEq, Eq, Copy)]
pub enum DataType {
    Real,
    Complex,
}

#[derive(Debug, Clone)]
pub enum TypedAst {
    Literal {
        value: Complex<f64>,
        ty: DataType,
    },
    Variable {
        name: String,
        ty: DataType,
    },
    FunctionCall {
        name: String,
        args: Vec<TypedAst>,
        ty: DataType,
    },
    UnaryOp {
        op: UnaryOp, // 変更: Rule -> UnaryOp
        child: Box<TypedAst>,
        ty: DataType,
    },
    BinaryOp {
        op: BinaryOp, // 変更: Rule -> BinaryOp
        lhs: Box<TypedAst>,
        rhs: Box<TypedAst>,
        ty: DataType,
    },
    CastToComplex {
        child: Box<TypedAst>,
    },
}

impl TypedAst {
    pub fn get_type(&self) -> DataType {
        match self {
            TypedAst::Literal { ty, .. } => *ty,
            TypedAst::Variable { ty, .. } => *ty,
            TypedAst::FunctionCall { ty, .. } => *ty,
            TypedAst::UnaryOp { ty, .. } => *ty,
            TypedAst::BinaryOp { ty, .. } => *ty,
            TypedAst::CastToComplex { .. } => DataType::Complex,
        }
    }
}

pub fn type_check(node: &AstNode) -> TypedAst {
    match node {
        AstNode::ComplexNumber(c) => {
            if c.im.abs() < 1e-16 {
                TypedAst::Literal {
                    value: *c,
                    ty: DataType::Real,
                }
            } else {
                TypedAst::Literal {
                    value: *c,
                    ty: DataType::Complex,
                }
            }
        }
        AstNode::Variable(name) => TypedAst::Variable {
            name: name.clone(),
            ty: DataType::Complex,
        },
        AstNode::FunctionCall { name, args } => {
            let typed_args: Vec<TypedAst> = args.iter().map(type_check).collect();
            // 第1引数の型を取得 (引数がない場合はComplexとしておく)
            let first_arg_type = typed_args
                .first()
                .map(|a| a.get_type())
                .unwrap_or(DataType::Complex);
            let ty = match name.as_str() {
                "abs" | "norm" | "length" | "real" | "imag" | "arg" => DataType::Real,
                // 2. 入力の型を引き継ぐ関数 (Real->Real, Complex->Complex)
                "sin" | "cos" | "tan" | "exp" | "ln" | "log" | "sqrt" | "conj" | "asin"
                | "acos" | "atan" | "arcsin" | "arccos" | "arctan" => first_arg_type,

                // 3. pow は引数のどちらかがComplexならComplex、両方RealならReal
                "pow" => {
                    if typed_args.iter().any(|a| a.get_type() == DataType::Complex) {
                        DataType::Complex
                    } else {
                        DataType::Real
                    }
                }
                // 4. それ以外はComplex
                _ => DataType::Complex,
            };
            TypedAst::FunctionCall {
                name: name.clone(),
                args: typed_args,
                ty,
            }
        }
        AstNode::UnaryOp { op, child } => {
            let typed_child = type_check(child);
            let ty = typed_child.get_type();
            TypedAst::UnaryOp {
                op: *op, // これで型が UnaryOp 同士で一致します
                child: Box::new(typed_child),
                ty,
            }
        }
        AstNode::BinaryOp { op, lhs, rhs } => {
            let t_lhs = type_check(lhs);
            let t_rhs = type_check(rhs);

            // 型の組み合わせに応じた処理
            match (t_lhs.get_type(), t_rhs.get_type()) {
                (DataType::Real, DataType::Real) => {
                    TypedAst::BinaryOp {
                        op: *op, // これで型が BinaryOp 同士で一致します
                        lhs: Box::new(t_lhs),
                        rhs: Box::new(t_rhs),
                        ty: DataType::Real,
                    }
                }
                (DataType::Complex, DataType::Complex) => TypedAst::BinaryOp {
                    op: *op,
                    lhs: Box::new(t_lhs),
                    rhs: Box::new(t_rhs),
                    ty: DataType::Complex,
                },
                (DataType::Complex, DataType::Real) => TypedAst::BinaryOp {
                    op: *op,
                    lhs: Box::new(t_lhs),
                    rhs: Box::new(TypedAst::CastToComplex {
                        child: Box::new(t_rhs),
                    }),
                    ty: DataType::Complex,
                },
                (DataType::Real, DataType::Complex) => TypedAst::BinaryOp {
                    op: *op,
                    lhs: Box::new(TypedAst::CastToComplex {
                        child: Box::new(t_lhs),
                    }),
                    rhs: Box::new(t_rhs),
                    ty: DataType::Complex,
                },
            }
        }
    }
}
