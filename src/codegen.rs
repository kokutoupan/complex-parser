use crate::ast::{AstNode, BinaryOp, UnaryOp};

pub fn generate_glsl(node: &AstNode) -> String {
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
                UnaryOp::Neg => format!("(-{})", child_glsl),
                // 単項プラスは最適化で消えているはずだが、念のため
                UnaryOp::Pos => child_glsl,
            }
        }
        AstNode::BinaryOp { op, lhs, rhs } => {
            let lhs_glsl = generate_glsl(lhs);
            let rhs_glsl = generate_glsl(rhs);
            // 演算子の優先順位を考慮し、常に括弧で囲むのが安全
            match op {
                BinaryOp::Add => format!("({} + {})", lhs_glsl, rhs_glsl),
                BinaryOp::Sub => format!("({} - {})", lhs_glsl, rhs_glsl),
                // 複素数の乗算・除算はヘルパー関数を呼び出す
                BinaryOp::Mul => format!("cmul({}, {})", lhs_glsl, rhs_glsl),
                BinaryOp::Div => format!("cdiv({}, {})", lhs_glsl, rhs_glsl),
            }
        }
    }
}

pub fn generate_glsl_dd(node: &AstNode) -> String {
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
            format!("mat2(vec2({}, 0.0), vec2({}, 0.0))", re_str, im_str)
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
                UnaryOp::Neg => format!("dcneg({})", child_glsl), // 単項マイナス
                UnaryOp::Pos => child_glsl,                       // 単項プラスは何もしない
            }
        }
        AstNode::BinaryOp { op, lhs, rhs } => {
            let lhs_glsl = generate_glsl_dd(lhs);
            let rhs_glsl = generate_glsl_dd(rhs);

            // 二項演算子を倍々精度複素数用の `dc...` 関数にマッピング
            match op {
                BinaryOp::Add => format!("dcadd({}, {})", lhs_glsl, rhs_glsl),
                BinaryOp::Sub => format!("dcsub({}, {})", lhs_glsl, rhs_glsl),
                BinaryOp::Mul => format!("dcmul({}, {})", lhs_glsl, rhs_glsl),
                BinaryOp::Div => format!("dcdiv({}, {})", lhs_glsl, rhs_glsl),
            }
        }
    }
}
