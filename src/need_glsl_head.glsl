/**
 * GLSL Helper Functions for Complex Number Operations (Single Precision)
 * A complex number z = x + iy is represented as vec2(x, y).
 */

// ---------------------------------------------
// Section 0: Basic Helpers
// ---------------------------------------------

#ifndef PI
#define PI 3.141592653589793
#endif

// Hyperbolic sine: sinh(x) = (e^x - e^-x) / 2
float sinh(float x) {
    return 0.5 * (exp(x) - exp(-x));
}

// Hyperbolic cosine: cosh(x) = (e^x + e^-x) / 2
float cosh(float x) {
    return 0.5 * (exp(x) + exp(-x));
}

// ---------------------------------------------
// Section 1: Core Arithmetic
// ---------------------------------------------

// Complex multiplication
vec2 cmul(vec2 a, vec2 b) {
    return vec2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

// Complex division
vec2 cdiv(vec2 a, vec2 b) {
    float d = dot(b, b);
    if (d == 0.0) return vec2(0.0);
    return vec2(dot(a, b), a.y * b.x - a.x * b.y) / d;
}

// Complex conjugate
vec2 cconj(vec2 z) {
    return vec2(z.x, -z.y);
}

// ---------------------------------------------
// Section 2: Exponential, Logarithmic, Power
// ---------------------------------------------

// Complex natural logarithm: log(z)
vec2 clog(vec2 z) {
    float r = length(z);
    if (r == 0.0) return vec2(0.0);
    return vec2(log(r), atan(z.y, z.x));
}

// Complex exponential: exp(z)
vec2 cexp(vec2 z) {
    return exp(z.x) * vec2(cos(z.y), sin(z.y));
}

// Complex power: z^w
vec2 cpow(vec2 base, vec2 exponent) {
    if (length(base) == 0.0) return vec2(0.0);
    return cexp(cmul(exponent, clog(base)));
}

// ---------------------------------------------
// Section 3: Trigonometric
// ---------------------------------------------

// Complex square root
vec2 csqrt(vec2 z) {
    float r = length(z);
    // principal branch
    return vec2(
        sqrt(0.5 * (r + z.x)),
        sign(z.y) * sqrt(0.5 * (r - z.x))
    );
}

// Complex sine
vec2 csin(vec2 z) {
    return vec2(sin(z.x) * cosh(z.y), cos(z.x) * sinh(z.y));
}

// Complex cosine
vec2 ccos(vec2 z) {
    return vec2(cos(z.x) * cosh(z.y), -sin(z.x) * sinh(z.y));
}

// Complex tangent
vec2 ctan(vec2 z) {
    return cdiv(csin(z), ccos(z));
}

// ---------------------------------------------
// Section 4: Inverse Trigonometric
// ---------------------------------------------

// Complex arcsin: -i * log(iz + sqrt(1-z^2))
vec2 casin(vec2 z) {
    vec2 i = vec2(0.0, 1.0);
    vec2 one = vec2(1.0, 0.0);
    vec2 t = cmul(i, z); // iz
    vec2 root = csqrt(one - cmul(z, z)); // sqrt(1-z^2)
    vec2 log_val = clog(t + root);
    return vec2(log_val.y, -log_val.x); // -i * log_val
}

// Complex arccos: pi/2 - asin(z)
vec2 cacos(vec2 z) {
    return vec2(PI * 0.5, 0.0) - casin(z);
}

// Complex arctan: (i/2) * log((i-z)/(i+z))
vec2 catan(vec2 z) {
    vec2 i = vec2(0.0, 1.0);
    vec2 one = vec2(1.0, 0.0);
    vec2 t1 = i - z;
    vec2 t2 = i + z;
    vec2 log_val = clog(cdiv(t1, t2));
    return cmul(vec2(0.0, -0.5), cmul(i, log_val)); // i/2 = 0.5i -> vec2(0, 0.5)
    // Simplified: i/2 * (x+iy) = (-y/2, x/2)
    // return vec2(-log_val.y * 0.5, log_val.x * 0.5);
}