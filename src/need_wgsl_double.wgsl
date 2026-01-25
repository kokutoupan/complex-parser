// =========================================================
// WGSL Double-Double Arithmetic Library
// Real: vec2<f32> (hi, lo)
// Complex: mat2x2<f32> (col0=Real, col1=Imag)
// =========================================================

const SPLITTER: f32 = 134217729.0; // 2^27 + 1

// --- Helper Functions ---

fn fast_two_sum(a: f32, b: f32) -> vec2<f32> {
    let s = a + b;
    let e = b - (s - a);
    return vec2<f32>(s, e);
}

fn two_sum(a: f32, b: f32) -> vec2<f32> {
    let s = a + b;
    let v = s - a;
    let e = (a - (s - v)) + (b - v);
    return vec2<f32>(s, e);
}

fn split(a: f32) -> vec2<f32> {
    let temp = SPLITTER * a;
    let hi = temp - (temp - a);
    let lo = a - hi;
    return vec2<f32>(hi, lo);
}

fn two_prod(a: f32, b: f32) -> vec2<f32> {
    let p = a * b;
    let a_s = split(a);
    let b_s = split(b);
    let err = ((a_s.x * b_s.x - p) + a_s.x * b_s.y + a_s.y * b_s.x) + a_s.y * b_s.y;
    return vec2<f32>(p, err);
}

// --- Double Real Arithmetic ---

fn dadd(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    let s = two_sum(a.x, b.x);
    let t = two_sum(a.y, b.y);
    let e = s.y + t.x;
    let s_fast = fast_two_sum(s.x, e);
    let e_final = s_fast.y + t.y;
    return fast_two_sum(s_fast.x, e_final);
}

fn dsub(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    let s = two_sum(a.x, -b.x);
    let t = two_sum(a.y, -b.y);
    let e = s.y + t.x;
    let s_fast = fast_two_sum(s.x, e);
    let e_final = s_fast.y + t.y;
    return fast_two_sum(s_fast.x, e_final);
}

fn dneg(a: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(-a.x, -a.y);
}

fn dmul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    let p = two_prod(a.x, b.x);
    let e = p.y + a.x * b.y + a.y * b.x;
    return fast_two_sum(p.x, e);
}

fn ddiv(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    let q1 = a.x / b.x;
    let p1 = dmul(vec2<f32>(q1, 0.0), b);
    let d1 = dsub(a, p1);
    let q2 = d1.x / b.x;
    let p2 = dmul(vec2<f32>(q2, 0.0), b);
    let d2 = dsub(d1, p2);
    let q3 = d2.x / b.x;
    let s = two_sum(q1, q2);
    return dadd(s, vec2<f32>(q3, 0.0));
}

// --- Double Real Math Functions ---

fn dsqrt(a: vec2<f32>) -> vec2<f32> {
    if a.x <= 0.0 { return vec2<f32>(0.0, 0.0); }
    let x = 1.0 / sqrt(a.x);
    var r = vec2<f32>(a.x * x, 0.0);
    r = dadd(r, ddiv(a, r)); r = dmul(r, vec2<f32>(0.5, 0.0));
    r = dadd(r, ddiv(a, r)); r = dmul(r, vec2<f32>(0.5, 0.0));
    return r;
}

fn dexp(a: vec2<f32>) -> vec2<f32> {
    let hi = exp(a.x);
    let lo = hi * a.y;
    return dadd(vec2<f32>(hi, 0.0), dmul(vec2<f32>(hi, lo), dsub(a, vec2<f32>(log(hi), 0.0))));
}

fn dlog(a: vec2<f32>) -> vec2<f32> {
    let hi = log(a.x);
    let t = dsub(a, dexp(vec2<f32>(hi, 0.0)));
    return dadd(vec2<f32>(hi, 0.0), ddiv(t, a));
}

// Struct to return both sin and cos
struct SinCos {
    s: vec2<f32>,
    c: vec2<f32>,
};

fn dsincos(a: vec2<f32>) -> SinCos {
    let s = sin(a.x);
    let c = cos(a.x);
    let dr = dsub(a, vec2<f32>(a.x, 0.0));
    let s_out = dadd(vec2<f32>(s, 0.0), dmul(dr, vec2<f32>(c, 0.0)));
    let c_out = dsub(vec2<f32>(c, 0.0), dmul(dr, vec2<f32>(s, 0.0)));
    return SinCos(s_out, c_out);
}

fn dsin(a: vec2<f32>) -> vec2<f32> { return dsincos(a).s; }
fn dcos(a: vec2<f32>) -> vec2<f32> { return dsincos(a).c; }
fn dtan(a: vec2<f32>) -> vec2<f32> {
    let sc = dsincos(a);
    return ddiv(sc.s, sc.c);
}

fn datan(a: vec2<f32>) -> vec2<f32> {
    let one = vec2<f32>(1.0, 0.0);
    // 0.5 * log((1+x)/(1-x))
    return dmul(vec2<f32>(0.5, 0.0), dlog(ddiv(dadd(one, a), dsub(one, a))));
}

fn dasin(a: vec2<f32>) -> vec2<f32> {
    let one = vec2<f32>(1.0, 0.0);
    return datan(ddiv(a, dsqrt(dsub(one, dmul(a, a)))));
}

fn dacos(a: vec2<f32>) -> vec2<f32> {
    // pi/2
    let pi_half = vec2<f32>(1.5707963267948966, 6.123233995736766e-17);
    return dsub(pi_half, dasin(a));
}

fn dpow(b: vec2<f32>, e: vec2<f32>) -> vec2<f32> {
    return dexp(dmul(e, dlog(b)));
}

// --- Double Complex Arithmetic ---

fn dcadd(a: mat2x2<f32>, b: mat2x2<f32>) -> mat2x2<f32> {
    return mat2x2<f32>(dadd(a[0], b[0]), dadd(a[1], b[1]));
}

fn dcsub(a: mat2x2<f32>, b: mat2x2<f32>) -> mat2x2<f32> {
    return mat2x2<f32>(dsub(a[0], b[0]), dsub(a[1], b[1]));
}

fn dcneg(a: mat2x2<f32>) -> mat2x2<f32> {
    return mat2x2<f32>(dneg(a[0]), dneg(a[1]));
}

fn dcmul(a: mat2x2<f32>, b: mat2x2<f32>) -> mat2x2<f32> {
    let re = dsub(dmul(a[0], b[0]), dmul(a[1], b[1]));
    let im = dadd(dmul(a[0], b[1]), dmul(a[1], b[0]));
    return mat2x2<f32>(re, im);
}

fn dcdiv(a: mat2x2<f32>, b: mat2x2<f32>) -> mat2x2<f32> {
    let denom = dadd(dmul(b[0], b[0]), dmul(b[1], b[1]));
    let re = ddiv(dadd(dmul(a[0], b[0]), dmul(a[1], b[1])), denom);
    let im = ddiv(dsub(dmul(a[1], b[0]), dmul(a[0], b[1])), denom);
    return mat2x2<f32>(re, im);
}

fn dcconj(a: mat2x2<f32>) -> mat2x2<f32> {
    return mat2x2<f32>(a[0], dneg(a[1]));
}

fn dcabs(a: mat2x2<f32>) -> vec2<f32> {
    return dsqrt(dadd(dmul(a[0], a[0]), dmul(a[1], a[1])));
}

// --- Double Complex Math Functions ---

fn dcsqrt(z: mat2x2<f32>) -> mat2x2<f32> {
    let r = dcabs(z);
    let half_v = vec2<f32>(0.5, 0.0);
    let sr = dsqrt(dmul(dadd(r, z[0]), half_v));
    var si = dsqrt(dmul(dsub(r, z[0]), half_v));
    if z[1].x < 0.0 { si = dneg(si); }
    return mat2x2<f32>(sr, si);
}

fn dcexp(z: mat2x2<f32>) -> mat2x2<f32> {
    let r = dexp(z[0]);
    let sc = dsincos(z[1]);
    return mat2x2<f32>(dmul(r, sc.c), dmul(r, sc.s));
}

fn dclog(z: mat2x2<f32>) -> mat2x2<f32> {
    let r = dcabs(z);
    let theta = datan(ddiv(z[1], z[0])); // Simple atan. TODO: Full atan2 logic
    return mat2x2<f32>(dlog(r), theta);
}

fn dcpow(b: mat2x2<f32>, e: mat2x2<f32>) -> mat2x2<f32> {
    return dcexp(dcmul(e, dclog(b)));
}

fn dcsin(z: mat2x2<f32>) -> mat2x2<f32> {
    let sc = dsincos(z[0]);
    let ey = dexp(z[1]);
    let eny = dexp(dneg(z[1]));
    let half_v = vec2<f32>(0.5, 0.0);
    let chy = dmul(dadd(ey, eny), half_v);
    let shy = dmul(dsub(ey, eny), half_v);
    return mat2x2<f32>(dmul(sc.s, chy), dmul(sc.c, shy));
}

fn dccos(z: mat2x2<f32>) -> mat2x2<f32> {
    let sc = dsincos(z[0]);
    let ey = dexp(z[1]);
    let eny = dexp(dneg(z[1]));
    let half_v = vec2<f32>(0.5, 0.0);
    let chy = dmul(dadd(ey, eny), half_v);
    let shy = dmul(dsub(ey, eny), half_v);
    return mat2x2<f32>(dmul(sc.c, chy), dneg(dmul(sc.s, shy)));
}

fn dctan(z: mat2x2<f32>) -> mat2x2<f32> {
    return dcdiv(dcsin(z), dccos(z));
}