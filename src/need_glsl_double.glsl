// =========================================================
// GLSL Double-Double Arithmetic Library (Complete)
// Real: vec2 (hi, lo)
// Complex: mat2 (col0=Real, col1=Imag)
// =========================================================

// --- Constants ---
const float SPLITTER = 134217729.0; // 2^27 + 1

// --- Helper Functions ---

vec2 fast_two_sum(float a, float b) {
    float s = a + b;
    float e = b - (s - a);
    return vec2(s, e);
}

vec2 two_sum(float a, float b) {
    float s = a + b;
    float v = s - a;
    float e = (a - (s - v)) + (b - v);
    return vec2(s, e);
}

vec2 split(float a) {
    float temp = SPLITTER * a;
    float hi = temp - (temp - a);
    float lo = a - hi;
    return vec2(hi, lo);
}

vec2 two_prod(float a, float b) {
    float p = a * b;
    vec2 a_s = split(a);
    vec2 b_s = split(b);
    float err = ((a_s.x * b_s.x - p) + a_s.x * b_s.y + a_s.y * b_s.x) + a_s.y * b_s.y;
    return vec2(p, err);
}

// --- Double Real Arithmetic ---

vec2 dadd(vec2 a, vec2 b) {
    vec2 s = two_sum(a.x, b.x);
    vec2 t = two_sum(a.y, b.y);
    s.y += t.x;
    s = fast_two_sum(s.x, s.y);
    s.y += t.y;
    return fast_two_sum(s.x, s.y);
}

vec2 dsub(vec2 a, vec2 b) {
    vec2 s = two_sum(a.x, -b.x);
    vec2 t = two_sum(a.y, -b.y);
    s.y += t.x;
    s = fast_two_sum(s.x, s.y);
    s.y += t.y;
    return fast_two_sum(s.x, s.y);
}

vec2 dneg(vec2 a) {
    return vec2(-a.x, -a.y);
}

vec2 dmul(vec2 a, vec2 b) {
    vec2 p = two_prod(a.x, b.x);
    p.y += a.x * b.y + a.y * b.x;
    return fast_two_sum(p.x, p.y);
}

vec2 ddiv(vec2 a, vec2 b) {
    float q1 = a.x / b.x;
    vec2 p = dmul(vec2(q1, 0.0), b);
    vec2 d = dsub(a, p);
    float q2 = d.x / b.x;
    p = dmul(vec2(q2, 0.0), b);
    d = dsub(d, p);
    float q3 = d.x / b.x;
    vec2 s = two_sum(q1, q2);
    return dadd(s, vec2(q3, 0.0));
}

// --- Double Real Math Functions ---

vec2 dsqrt(vec2 a) {
    if (a.x <= 0.0) return vec2(0.0);
    float x = 1.0 / sqrt(a.x);
    vec2 r = vec2(a.x * x, 0.0);
    r = dadd(r, ddiv(a, r)); r = dmul(r, vec2(0.5, 0.0));
    r = dadd(r, ddiv(a, r)); r = dmul(r, vec2(0.5, 0.0));
    return r;
}

vec2 dexp(vec2 a) {
    float hi = exp(a.x);
    float lo = hi * a.y;
    return dadd(vec2(hi, 0.0), dmul(vec2(hi, lo), dsub(a, vec2(log(hi), 0.0))));
}

vec2 dlog(vec2 a) {
    float hi = log(a.x);
    vec2 t = dsub(a, dexp(vec2(hi, 0.0)));
    return dadd(vec2(hi, 0.0), ddiv(t, a));
}

void dsincos(vec2 a, out vec2 s_out, out vec2 c_out) {
    float s = sin(a.x);
    float c = cos(a.x);
    vec2 dr = dsub(a, vec2(a.x, 0.0));
    s_out = dadd(vec2(s, 0.0), dmul(dr, vec2(c, 0.0)));
    c_out = dsub(vec2(c, 0.0), dmul(dr, vec2(s, 0.0)));
}

vec2 dsin(vec2 a) { vec2 s, c; dsincos(a, s, c); return s; }
vec2 dcos(vec2 a) { vec2 s, c; dsincos(a, s, c); return c; }
vec2 dtan(vec2 a) { vec2 s, c; dsincos(a, s, c); return ddiv(s, c); }

vec2 datan(vec2 a) {
    // atan(x) = 0.5 * log((1+x)/(1-x))
    vec2 one = vec2(1.0, 0.0);
    return dmul(vec2(0.5, 0.0), dlog(ddiv(dadd(one, a), dsub(one, a))));
}

vec2 dasin(vec2 a) {
    // asin(x) = atan(x / sqrt(1-x^2))
    vec2 one = vec2(1.0, 0.0);
    return datan(ddiv(a, dsqrt(dsub(one, dmul(a, a)))));
}

vec2 dacos(vec2 a) {
    // acos(x) = pi/2 - asin(x)
    vec2 pi_half = vec2(1.5707963267948966, 6.123233995736766e-17); 
    return dsub(pi_half, dasin(a));
}

vec2 dpow(vec2 b, vec2 e) { return dexp(dmul(e, dlog(b))); }

// --- Double Complex Arithmetic ---

mat2 dcadd(mat2 a, mat2 b) { return mat2(dadd(a[0], b[0]), dadd(a[1], b[1])); }
mat2 dcsub(mat2 a, mat2 b) { return mat2(dsub(a[0], b[0]), dsub(a[1], b[1])); }
mat2 dcneg(mat2 a) { return mat2(dneg(a[0]), dneg(a[1])); }

mat2 dcmul(mat2 a, mat2 b) {
    // (ar + ai*i) * (br + bi*i)
    vec2 re = dsub(dmul(a[0], b[0]), dmul(a[1], b[1]));
    vec2 im = dadd(dmul(a[0], b[1]), dmul(a[1], b[0]));
    return mat2(re, im);
}

mat2 dcdiv(mat2 a, mat2 b) {
    // a / b = (a * conj(b)) / |b|^2
    vec2 denom = dadd(dmul(b[0], b[0]), dmul(b[1], b[1])); // |b|^2
    vec2 re = ddiv(dadd(dmul(a[0], b[0]), dmul(a[1], b[1])), denom);
    vec2 im = ddiv(dsub(dmul(a[1], b[0]), dmul(a[0], b[1])), denom);
    return mat2(re, im);
}

mat2 dcconj(mat2 a) { return mat2(a[0], dneg(a[1])); }

vec2 dcabs(mat2 a) {
    // |z| = sqrt(x^2 + y^2)
    return dsqrt(dadd(dmul(a[0], a[0]), dmul(a[1], a[1])));
}

// --- Double Complex Math Functions ---

mat2 dcsqrt(mat2 z) {
    vec2 r = dcabs(z);
    vec2 halfd = vec2(0.5, 0.0);
    vec2 sr = dsqrt(dmul(dadd(r, z[0]), halfd));
    vec2 si = dsqrt(dmul(dsub(r, z[0]), halfd));
    if (z[1].x < 0.0) si = dneg(si);
    return mat2(sr, si);
}

mat2 dcexp(mat2 z) {
    vec2 r = dexp(z[0]);
    vec2 s, c;
    dsincos(z[1], s, c);
    return mat2(dmul(r, c), dmul(r, s));
}

mat2 dclog(mat2 z) {
    vec2 r = dcabs(z);
    vec2 theta = datan(ddiv(z[1], z[0])); // 簡易atan2. 厳密には象限判定が必要
    return mat2(dlog(r), theta);
}

mat2 dcpow(mat2 b, mat2 e) { return dcexp(dcmul(e, dclog(b))); }

mat2 dcsin(mat2 z) {
    // sin(x+iy) = sin(x)cosh(y) + i cos(x)sinh(y)
    // cosh(y) = (e^y + e^-y)/2, sinh(y) = (e^y - e^-y)/2
    vec2 sx, cx; dsincos(z[0], sx, cx);
    vec2 ey = dexp(z[1]);
    vec2 eny = dexp(dneg(z[1]));
    vec2 halfd = vec2(0.5, 0.0);
    vec2 chy = dmul(dadd(ey, eny), halfd);
    vec2 shy = dmul(dsub(ey, eny), halfd);
    return mat2(dmul(sx, chy), dmul(cx, shy));
}

mat2 dccos(mat2 z) {
    // cos(x+iy) = cos(x)cosh(y) - i sin(x)sinh(y)
    vec2 sx, cx; dsincos(z[0], sx, cx);
    vec2 ey = dexp(z[1]);
    vec2 eny = dexp(dneg(z[1]));
    vec2 halfd = vec2(0.5, 0.0);
    vec2 chy = dmul(dadd(ey, eny), halfd);
    vec2 shy = dmul(dsub(ey, eny), halfd);
    return mat2(dmul(cx, chy), dneg(dmul(sx, shy)));
}

mat2 dctan(mat2 z) { return dcdiv(dcsin(z), dccos(z)); }
