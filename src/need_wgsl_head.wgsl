// =========================================================
// WGSL Complex Number Library (Single Precision)
// Represents complex number z = x + iy as vec2<f32>(x, y)
// =========================================================

const PI: f32 = 3.141592653589793;

// ---------------------------------------------
// Section 0: Basic Helpers
// ---------------------------------------------

fn sinh(x: f32) -> f32 {
    return 0.5 * (exp(x) - exp(-x));
}

fn cosh(x: f32) -> f32 {
    return 0.5 * (exp(x) + exp(-x));
}

// ---------------------------------------------
// Section 1: Core Arithmetic
// ---------------------------------------------

// Complex multiplication
fn cmul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

// Complex division
fn cdiv(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    let d = dot(b, b);
    if d == 0.0 { return vec2<f32>(0.0, 0.0); }
    return vec2<f32>(dot(a, b), a.y * b.x - a.x * b.y) / d;
}

// Complex conjugate
fn cconj(z: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(z.x, -z.y);
}

// ---------------------------------------------
// Section 2: Exponential, Logarithmic, Power
// ---------------------------------------------

// Complex natural logarithm: log(z)
fn clog(z: vec2<f32>) -> vec2<f32> {
    let r = length(z);
    if r == 0.0 { return vec2<f32>(0.0, 0.0); }
    // WGSL atan2 corresponds to GLSL atan(y, x)
    return vec2<f32>(log(r), atan2(z.y, z.x));
}

// Complex exponential: exp(z)
fn cexp(z: vec2<f32>) -> vec2<f32> {
    return exp(z.x) * vec2<f32>(cos(z.y), sin(z.y));
}

// Complex power: z^w
fn cpow(base: vec2<f32>, exponent: vec2<f32>) -> vec2<f32> {
    if length(base) == 0.0 { return vec2<f32>(0.0, 0.0); }
    return cexp(cmul(exponent, clog(base)));
}

// ---------------------------------------------
// Section 3: Trigonometric
// ---------------------------------------------

// Complex square root
fn csqrt(z: vec2<f32>) -> vec2<f32> {
    let r = length(z);
    // principal branch
    return vec2<f32>(
        sqrt(0.5 * (r + z.x)),
        sign(z.y) * sqrt(0.5 * (r - z.x))
    );
}

// Complex sine
fn csin(z: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(sin(z.x) * cosh(z.y), cos(z.x) * sinh(z.y));
}

// Complex cosine
fn ccos(z: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(cos(z.x) * cosh(z.y), -sin(z.x) * sinh(z.y));
}

// Complex tangent
fn ctan(z: vec2<f32>) -> vec2<f32> {
    return cdiv(csin(z), ccos(z));
}

// ---------------------------------------------
// Section 4: Inverse Trigonometric
// ---------------------------------------------

// Complex arcsin: -i * log(iz + sqrt(1-z^2))
fn casin(z: vec2<f32>) -> vec2<f32> {
    let i = vec2<f32>(0.0, 1.0);
    let one = vec2<f32>(1.0, 0.0);
    let t = cmul(i, z); // iz
    let root = csqrt(one - cmul(z, z)); // sqrt(1-z^2)
    let log_val = clog(t + root);
    
    // -i * (x + iy) = (y, -x)
    return vec2<f32>(log_val.y, -log_val.x);
}

// Complex arccos: pi/2 - asin(z)
fn cacos(z: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(PI * 0.5, 0.0) - casin(z);
}

// Complex arctan: (i/2) * log((i-z)/(i+z))
fn catan(z: vec2<f32>) -> vec2<f32> {
    let i = vec2<f32>(0.0, 1.0);
    let t1 = i - z;
    let t2 = i + z;
    let log_val = clog(cdiv(t1, t2));
    
    // (i/2) * (x + iy) = (0, 0.5) * (x, y) = (-0.5y, 0.5x)
    return vec2<f32>(-log_val.y * 0.5, log_val.x * 0.5);
}