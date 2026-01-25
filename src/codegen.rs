use std::fmt::format;

use crate::ast::{BinaryOp, UnaryOp};
use crate::typing::{DataType, TypedAst}; // 変更

pub trait CodegenBackend {
    fn format_real_literal(&self, val: f64) -> String;
    fn format_complex_literal(&self, re: f64, im: f64) -> String;
    fn format_variable(&self, name: &str) -> String;
    fn format_cast_to_complex(&self, child_str: &str) -> String;

    // 引数の型を変更
    fn format_binary_op(&self, op: BinaryOp, lhs: &str, rhs: &str, ty: DataType) -> String;
    fn format_unary_op(&self, op: UnaryOp, child: &str, ty: DataType) -> String;

    fn format_function_call(
        &self,
        name: &str,
        args: &[String],
        arg_types: &[DataType],
        ty: DataType,
    ) -> String;
}

pub fn generate_code(node: &TypedAst, backend: &impl CodegenBackend) -> String {
    match node {
        TypedAst::Literal { value, ty } => match ty {
            DataType::Real => backend.format_real_literal(value.re),
            DataType::Complex => backend.format_complex_literal(value.re, value.im),
        },
        TypedAst::Variable { name, .. } => backend.format_variable(name),
        TypedAst::CastToComplex { child } => {
            let c = generate_code(child, backend);
            backend.format_cast_to_complex(&c)
        }
        TypedAst::FunctionCall { name, args, ty } => {
            let (arg_strs, arg_types): (Vec<String>, Vec<DataType>) = args
                .iter()
                .map(|arg| (generate_code(arg, backend), arg.get_type()))
                .unzip();
            backend.format_function_call(name, &arg_strs, &arg_types, *ty)
        }
        TypedAst::UnaryOp { op, child, ty } => {
            let c = generate_code(child, backend);
            backend.format_unary_op(*op, &c, *ty)
        }
        TypedAst::BinaryOp { op, lhs, rhs, ty } => {
            let l = generate_code(lhs, backend);
            let r = generate_code(rhs, backend);
            backend.format_binary_op(*op, &l, &r, *ty)
        }
    }
}

// --- GlslBackend の実装 (Single Precision) ---

pub struct GlslBackend;

impl CodegenBackend for GlslBackend {
    fn format_real_literal(&self, val: f64) -> String {
        let s = format!("{}", val);
        if !s.contains('.') && !s.contains('e') {
            format!("{}.0", s)
        } else {
            s
        }
    }

    fn format_complex_literal(&self, re: f64, im: f64) -> String {
        let re_s = self.format_real_literal(re);
        let im_s = self.format_real_literal(im);
        format!("vec2({}, {})", re_s, im_s)
    }

    fn format_variable(&self, name: &str) -> String {
        name.to_string()
    }

    fn format_cast_to_complex(&self, child_str: &str) -> String {
        format!("vec2({}, 0.0)", child_str)
    }

    fn format_binary_op(&self, op: BinaryOp, lhs: &str, rhs: &str, ty: DataType) -> String {
        match ty {
            DataType::Real => match op {
                BinaryOp::Add => format!("({} + {})", lhs, rhs),
                BinaryOp::Sub => format!("({} - {})", lhs, rhs),
                BinaryOp::Mul => format!("({} * {})", lhs, rhs),
                BinaryOp::Div => format!("({} / {})", lhs, rhs),
            },
            DataType::Complex => match op {
                BinaryOp::Add => format!("({} + {})", lhs, rhs),
                BinaryOp::Sub => format!("({} - {})", lhs, rhs),
                BinaryOp::Mul => format!("cmul({}, {})", lhs, rhs),
                BinaryOp::Div => format!("cdiv({}, {})", lhs, rhs),
            },
        }
    }

    fn format_unary_op(&self, op: UnaryOp, child: &str, ty: DataType) -> String {
        match op {
            UnaryOp::Neg => format!("(-{})", child),
            UnaryOp::Pos => child.to_string(),
        }
    }

    fn format_function_call(
        &self,
        name: &str,
        args: &[String],
        arg_types: &[DataType],
        _ty: DataType,
    ) -> String {
        let is_arg_complex = arg_types
            .first()
            .map(|t| *t == DataType::Complex)
            .unwrap_or(false);

        match name {
            "abs" | "length" | "norm" => return format!("length({})", args.join(", ")),
            "real" => return format!("({}).x", args[0]),
            "imag" => return format!("({}).y", args[0]),
            "conj" => {
                if is_arg_complex {
                    format!("cconj({})", args[0])
                } else {
                    args[0].to_string() // 実数の共役はそのまま
                }
            }
            "sin" | "cos" | "tan" | "exp" | "ln" | "log" | "sqrt" | "pow" => {
                // Realなら標準関数 (例: sin), Complexなら csin
                let base_name = if name == "ln" { "log" } else { name };
                if is_arg_complex {
                    format!("c{}({})", base_name, args.join(", "))
                } else {
                    format!("{}({})", base_name, args.join(", "))
                }
            }
            _ => format!("{}({})", name, args.join(", ")),
        }
    }
}

// --- GlslDoubleBackend の実装修正 ---

pub struct GlslDoubleBackend;

impl CodegenBackend for GlslDoubleBackend {
    fn format_real_literal(&self, val: f64) -> String {
        let s = format!("{}", val);
        let s = if !s.contains('.') && !s.contains('e') {
            format!("{}.0", s)
        } else {
            s
        };
        format!("vec2({}, 0.0)", s)
    }

    fn format_complex_literal(&self, re: f64, im: f64) -> String {
        let re_s = format!("{}", re);
        let re_s = if !re_s.contains('.') && !re_s.contains('e') {
            format!("{}.0", re_s)
        } else {
            re_s
        };
        let im_s = format!("{}", im);
        let im_s = if !im_s.contains('.') && !im_s.contains('e') {
            format!("{}.0", im_s)
        } else {
            im_s
        };
        format!("mat2(vec2({}, 0.0), vec2({}, 0.0))", re_s, im_s)
    }

    fn format_variable(&self, name: &str) -> String {
        name.to_string()
    }

    fn format_cast_to_complex(&self, child_str: &str) -> String {
        format!("mat2({}, vec2(0.0))", child_str)
    }

    fn format_binary_op(&self, op: BinaryOp, lhs: &str, rhs: &str, ty: DataType) -> String {
        match ty {
            DataType::Real => match op {
                BinaryOp::Add => format!("dadd({}, {})", lhs, rhs),
                BinaryOp::Sub => format!("dsub({}, {})", lhs, rhs),
                BinaryOp::Mul => format!("dmul({}, {})", lhs, rhs),
                BinaryOp::Div => format!("ddiv({}, {})", lhs, rhs),
            },
            DataType::Complex => match op {
                BinaryOp::Add => format!("dcadd({}, {})", lhs, rhs),
                BinaryOp::Sub => format!("dcsub({}, {})", lhs, rhs),
                BinaryOp::Mul => format!("dcmul({}, {})", lhs, rhs),
                BinaryOp::Div => format!("dcdiv({}, {})", lhs, rhs),
            },
        }
    }

    fn format_unary_op(&self, op: UnaryOp, child: &str, ty: DataType) -> String {
        match op {
            UnaryOp::Neg => match ty {
                DataType::Real => format!("dneg({})", child),
                DataType::Complex => format!("dcneg({})", child),
            },
            _ => unreachable!(),
        }
    }

    fn format_function_call(
        &self,
        name: &str,
        args: &[String],
        arg_types: &[DataType],
        _ty: DataType,
    ) -> String {
        let is_arg_complex = arg_types
            .first()
            .map(|t| *t == DataType::Complex)
            .unwrap_or(false);

        match name {
            "abs" | "length" | "norm" => return format!("dcabs({})", args.join(", ")),
            "real" => return format!("vec2(({})[0])", args[0]),
            "imag" => return format!("vec2(({})[1])", args[0]),
            "conj" => {
                if is_arg_complex {
                    format!("dcconj({})", args[0])
                } else {
                    args[0].to_string() // 実数の共役はそのまま
                }
            }
            "sin" | "cos" | "tan" | "exp" | "ln" | "log" | "sqrt" | "pow" => {
                // Realなら標準関数 (例: sin), Complexなら csin
                let base_name = if name == "ln" { "log" } else { name };
                if is_arg_complex {
                    format!("dc{}({})", base_name, args.join(", "))
                } else {
                    format!("d{}({})", base_name, args.join(", "))
                }
            }
            _ => format!("{}({})", name, args.join(", ")),
        }
    }
}

pub struct WgslBackend;

impl CodegenBackend for WgslBackend {
    fn format_real_literal(&self, val: f64) -> String {
        // WGSL: f32 (または f64 が使える環境なら f64)
        // WebGPU標準では f32 が基本なので、一旦 f32 として出力します
        // 必要なら "f32" を "f64" に置換できるようにするか、設定を持たせます
        let s = format!("{}", val);
        if !s.contains('.') && !s.contains('e') {
            format!("{}.0", s)
        } else {
            s
        }
    }

    fn format_complex_literal(&self, re: f64, im: f64) -> String {
        // WGSL: vec2<f32>(re, im)
        // エイリアス vec2f も使えますが、冗長に vec2<f32> と書くのが無難です
        let re_s = self.format_real_literal(re);
        let im_s = self.format_real_literal(im);
        format!("vec2<f32>({}, {})", re_s, im_s)
    }

    fn format_variable(&self, name: &str) -> String {
        name.to_string()
    }

    fn format_cast_to_complex(&self, child_str: &str) -> String {
        // 実数(float) -> 複素数(vec2<f32>)
        // vec2<f32>(val, 0.0)
        format!("vec2<f32>({}, 0.0)", child_str)
    }

    fn format_binary_op(&self, op: BinaryOp, lhs: &str, rhs: &str, ty: DataType) -> String {
        match ty {
            DataType::Real => match op {
                // 実数同士なら標準演算子が使える
                BinaryOp::Add => format!("({} + {})", lhs, rhs),
                BinaryOp::Sub => format!("({} - {})", lhs, rhs),
                BinaryOp::Mul => format!("({} * {})", lhs, rhs),
                BinaryOp::Div => format!("({} / {})", lhs, rhs),
            },
            DataType::Complex => match op {
                // 複素数同士はユーザー定義関数が必要 (cmulなど)
                // WGSLは演算子オーバーロードがないため関数呼び出し必須
                BinaryOp::Add => format!("({} + {})", lhs, rhs), // vec2同士の足し算は成分ごと
                BinaryOp::Sub => format!("({} - {})", lhs, rhs), // 引き算もOK
                BinaryOp::Mul => format!("cmul({}, {})", lhs, rhs), // 掛け算は関数
                BinaryOp::Div => format!("cdiv({}, {})", lhs, rhs), // 割り算も関数
            },
        }
    }

    fn format_unary_op(&self, op: UnaryOp, child: &str, ty: DataType) -> String {
        match op {
            UnaryOp::Neg => format!("(-{})", child), // vec2の反転も標準対応
            _ => unreachable!(),
        }
    }

    fn format_function_call(
        &self,
        name: &str,
        args: &[String],
        arg_types: &[DataType],
        _ty: DataType,
    ) -> String {
        let is_arg_complex = arg_types
            .first()
            .map(|t| *t == DataType::Complex)
            .unwrap_or(false);

        match name {
            "abs" | "length" | "norm" => return format!("length({})", args.join(", ")),
            "real" => return format!("({}).x", args[0]),
            "imag" => return format!("({}).y", args[0]),
            "conj" => {
                if is_arg_complex {
                    format!("cconj({})", args[0])
                } else {
                    args[0].to_string() // 実数の共役はそのまま
                }
            }
            "sin" | "cos" | "tan" | "exp" | "ln" | "log" | "sqrt" | "pow" => {
                // Realなら標準関数 (例: sin), Complexなら csin
                let base_name = if name == "ln" { "log" } else { name };
                if is_arg_complex {
                    format!("c{}({})", base_name, args.join(", "))
                } else {
                    format!("{}({})", base_name, args.join(", "))
                }
            }
            _ => format!("{}({})", name, args.join(", ")),
        }
    }
}

// ★ ヘルパー関数: f64 を 上位(Hi) と 下位(Lo) の f32 に分割する
// これにより、Rustのf64精度を損なわずにシェーダの "Emulated Double" に渡せます
fn split_f64(val: f64) -> (f32, f32) {
    if val == 0.0 {
        return (0.0, 0.0);
    }
    let hi = val as f32;
    let lo = (val - hi as f64) as f32;
    (hi, lo)
}

// WGSL用の f32 文字列生成 ("1.0", "0.5" など)
fn to_wgsl_f32(val: f32) -> String {
    let s = format!("{}", val);
    if !s.contains('.') && !s.contains('e') {
        format!("{}.0", s)
    } else {
        s
    }
}

// --- WgslDoubleBackendの実装 ---

pub struct WgslDoubleBackend;

impl CodegenBackend for WgslDoubleBackend {
    fn format_real_literal(&self, val: f64) -> String {
        // f64 -> vec2<f32>(hi, lo)
        let (hi, lo) = split_f64(val);
        format!("vec2<f32>({}, {})", to_wgsl_f32(hi), to_wgsl_f32(lo))
    }

    fn format_complex_literal(&self, re: f64, im: f64) -> String {
        // f64 complex -> mat2x2<f32>( real_vec, imag_vec )
        let (re_hi, re_lo) = split_f64(re);
        let (im_hi, im_lo) = split_f64(im);

        // WGSLの行列コンストラクタは列(Column)ごとに指定します
        // mat2x2<f32>(col0, col1) -> col0が実部, col1が虚部
        format!(
            "mat2x2<f32>(vec2<f32>({}, {}), vec2<f32>({}, {}))",
            to_wgsl_f32(re_hi),
            to_wgsl_f32(re_lo),
            to_wgsl_f32(im_hi),
            to_wgsl_f32(im_lo)
        )
    }

    fn format_variable(&self, name: &str) -> String {
        name.to_string()
    }

    fn format_cast_to_complex(&self, child_str: &str) -> String {
        // Real(vec2) -> Complex(mat2x2)
        // 虚部は 0.0 (Hi=0, Lo=0)
        format!("mat2x2<f32>({}, vec2<f32>(0.0, 0.0))", child_str)
    }

    fn format_binary_op(&self, op: BinaryOp, lhs: &str, rhs: &str, ty: DataType) -> String {
        // 関数名は GLSL Double 版 (dcaddなど) と合わせるか、WGSL用に prefix を変えるか決められます。
        // ここでは GLSL版と同じ名前の関数が WGSL でも定義されていると仮定します。
        match ty {
            DataType::Real => match op {
                BinaryOp::Add => format!("dadd({}, {})", lhs, rhs),
                BinaryOp::Sub => format!("dsub({}, {})", lhs, rhs),
                BinaryOp::Mul => format!("dmul({}, {})", lhs, rhs),
                BinaryOp::Div => format!("ddiv({}, {})", lhs, rhs),
            },
            DataType::Complex => match op {
                BinaryOp::Add => format!("dcadd({}, {})", lhs, rhs),
                BinaryOp::Sub => format!("dcsub({}, {})", lhs, rhs),
                BinaryOp::Mul => format!("dcmul({}, {})", lhs, rhs),
                BinaryOp::Div => format!("dcdiv({}, {})", lhs, rhs),
            },
        }
    }

    fn format_unary_op(&self, op: UnaryOp, child: &str, ty: DataType) -> String {
        match op {
            UnaryOp::Neg => match ty {
                DataType::Real => format!("dneg({})", child),
                DataType::Complex => format!("dcneg({})", child),
            },
            UnaryOp::Pos => child.to_string(),
        }
    }

    fn format_function_call(
        &self,
        name: &str,
        args: &[String],
        arg_types: &[DataType],
        _ty: DataType,
    ) -> String {
        let is_arg_complex = arg_types
            .first()
            .map(|t| *t == DataType::Complex)
            .unwrap_or(false);

        match name {
            "abs" | "length" | "norm" => return format!("dcabs({})", args.join(", ")),
            "real" => return format!("({}[0])", args[0]),
            "imag" => return format!("({}[1])", args[0]),
            "conj" => {
                if is_arg_complex {
                    format!("dcconj({})", args[0])
                } else {
                    args[0].to_string() // 実数の共役はそのまま
                }
            }

            "sin" | "cos" | "tan" | "exp" | "ln" | "log" | "sqrt" | "pow" => {
                let prefix = if is_arg_complex { "dc" } else { "d" };
                let base_name = if name == "ln" { "log" } else { name };
                format!("{}{}({})", prefix, base_name, args.join(", "))
            }
            _ => format!("{}({})", name, args.join(", ")),
        }
    }
}
