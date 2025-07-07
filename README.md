# Complex Parser

**`complex-parser`** は、複素数を含む数式文字列を解析し、GLSL（WebGL）で利用可能なシェーダコードに変換するRust製のWebAssemblyライブラリです。

このライブラリは、ユーザーが入力した数式に基づいて動的にフラクタル図形（マンデルブロ集合やジュリア集合など）を描画するような、インタラクティブなWebアプリケーションでの使用を主な目的としています。

## 概要 (Overview)

ウェブ上でGLSLを用いたグラフィックスプログラミングを行う際、シェーダコードは通常静的な文字列として記述されます。しかし、ユーザーの入力に応じてリアルタイムにシェーダの計算式を変更したい場合があります。

本ライブラリは、`z * z + c` のような一般的な数式文字列をWebAssembly経由で受け取り、それを解析してGLSLの `vec2` 型を用いた複素数計算のコード `cmul(z, z) + c` に変換します。これにより、JavaScript側で複雑な文字列置換やパーサーを実装することなく、安全かつ高速に動的なシェーダを生成できます。

## デモ & 実用例 (Live Demo & Example)

この `complex-parser` ライブラリは、以下のWebアプリケーションで実際に使用されています。ユーザーが入力した数式をリアルタイムでシェーダに変換する、このライブラリの核となる機能をご覧いただけます。

**[Mandelbrot & Julia Set Renderer](https://kokutoupan.github.io/mandelbrot-julia-set-renderer/)**


このアプリケーションでは、マンデルブロ集合やジュリア集合を描画する際の計算式をユーザーが自由に入力できます。例えば、`z*z + c` という式を `sin(z*z) + c` や `pow(z, 3) + c` などに変更すると、その場で描画される図形が変わります。

この「ユーザーが入力した数式を解釈し、GLSLコードに変換する処理」のすべてを `complex-parser` が担っています。

## 主な機能 (Features)

-   **数式の解析**: `pest` を用いた堅牢なパーサーで、四則演算、括弧、関数の呼び出しを含む数式を抽象構文木（AST）に変換します。
    
-   **複素数演算のサポート**: `num-complex` クレートを利用して、実数および複素数の計算に対応しています。
    
-   **定数畳み込み**: `sin(pi/2)` のような定数式を、ASTレベルでコンパイル時に計算し、最適化します。（例: `1.0` に変換）
    
-   **GLSLコード生成**: 最適化されたASTから、GLSLの `vec2` を複素数として扱うシェーダコードを生成します。
    
-   **WebAssembly対応**: `wasm-bindgen` を通じてJavaScriptから簡単に呼び出せるAPIを提供します。
    
-   **豊富な組み込み関数**: 三角関数、指数関数、対数関数など、複素数に対応した多くの数学関数をサポートします。
    

### 対応している構文

-   **数値リテラル**: `1.23`, `42`, `3.14i`
    
-   **変数**: `z`, `c` など（GLSLの `vec2` 型変数に対応）
    
-   **定数**: `pi`, `i`
    
-   **演算子**:
    
    -   加算: `+`
        
    -   減算: `-`
        
    -   乗算: `*`
        
    -   除算: `/`
        
    -   単項演算子: `+`, `-`
        
-   **関数**:
    
    -   `sqrt(c)`: 平方根
        
    -   `sin(c)`, `cos(c)`, `tan(c)`: 三角関数
        
    -   `asin(c)`, `acos(c)`, `atan(c)`: 逆三角関数
        
    -   `exp(c)`: 指数関数
        
    -   `ln(c)`, `log(c)`: 自然対数
        
    -   `pow(base, exp)`: べき乗
        
    -   `abs(c)`: 絶対値（ノルム）
        
    -   `real(c)`: 実部
        
    -   `imag(c)`: 虚部
        
    -   `conj(c)`: 共役複素数
        

## 使い方 (Usage)

このライブラリはWebAssemblyモジュールとしてビルドし、JavaScriptから利用することを想定しています。
### 1. ビルド

`wasm-pack` を使ってプロジェクトをビルドします。`Cargo.toml` の `[package]` セクションの `name` が `complex-parser` になっていることを確認してください。

Bash

```
# --target web を指定してビルド
wasm-pack build --target web
```

これにより、`pkg` ディレクトリに必要なJavaScriptバインディングと `.wasm` ファイルが生成されます。

### 2. JavaScriptからの利用

生成されたモジュールをインポートして `get_glsl_output` 関数を呼び出します。

JavaScript

```
import init, { get_glsl_output } from './pkg/complex_parser.js';

async function main() {
  // WASMモジュールの初期化
  await init();

  // 変換したい数式
  const expression1 = "sin(z*z + c)";
  const expression2 = "pow(z, 3.0) - conj(c) / abs(z)";
  const expression3 = "exp(i * pi)"; // 定数式

  // GLSLコードに変換
  const glsl1 = get_glsl_output(expression1);
  const glsl2 = get_glsl_output(expression2);
  const glsl3 = get_glsl_output(expression3);

  console.log(`'${expression1}' -> ${glsl1}`);
  // > 'sin(z*z + c)' -> csin((cmul(z, z) + c))

  console.log(`'${expression2}' -> ${glsl2}`);
  // > 'pow(z, 3.0) - conj(c) / abs(z)' -> (cpow(z, vec2(3.0, 0.0)) - cdiv(vec2((c).x, -(c).y), vec2(length(z), 0.0)))

  console.log(`'${expression3}' -> ${glsl3}`);
  // > 'exp(i * pi)' -> vec2(-1.0, 0.0)  (最適化された結果)
}

main();

```

### 3. GLSL側の準備

このライブラリが生成するGLSLコードは、複素数演算のためのヘルパー関数（`cmul`, `cdiv`, `csin`など）がGLSLシェーダ内に定義されていることを前提としています。
`src/need_glsl_head.glsl`をコピペしてください。
