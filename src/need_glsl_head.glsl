/**
 * GLSL Helper Functions for Complex Number Operations
 * A complex number z = x + iy is represented as vec2(x, y).
 */

// ---------------------------------------------
// Section 1: Core Arithmetic & Hyperbolic Funcs
// ---------------------------------------------

// Hyperbolic sine: sinh(x) = (e^x - e^-x) / 2
// (Required by csin, ccos)
float sinh(float x) {
    return 0.5 * (exp(x) - exp(-x));
}

// Hyperbolic cosine: cosh(x) = (e^x + e^-x) / 2
// (Required by csin, ccos)
float cosh(float x) {
    return 0.5 * (exp(x) + exp(-x));
}

// Complex multiplication: (a+bi) * (c+di) = (ac-bd) + (ad+bc)i
vec2 cmul(vec2 a, vec2 b) {
    return vec2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

// Complex division: (a+bi) / (c+di)
vec2 cdiv(vec2 a, vec2 b) {
    float d = dot(b, b); // Denominator: |b|^2
    // Return (0,0) if denominator is zero to avoid division by zero
    if (d == 0.0) {
        return vec2(0.0, 0.0);
    }
    return vec2(dot(a, b), a.y * b.x - a.x * b.y) / d;
}

// ---------------------------------------------
// Section 2: Exponential, Logarithmic, Power
// ---------------------------------------------

// Complex natural logarithm: log(z) = log|z| + i*arg(z)
vec2 clog(vec2 z) {
    // Return (0,0) if z is zero
    float r = length(z);
    if (r == 0.0) {
        return vec2(0.0, 0.0);
    }
    return vec2(log(r), atan(z.y, z.x));
}

// Complex exponential: exp(x+iy) = e^x * (cos(y) + i*sin(y))
vec2 cexp(vec2 z) {
    return exp(z.x) * vec2(cos(z.y), sin(z.y));
}

// Complex power: z^w = exp(w * log(z))
vec2 cpow(vec2 base, vec2 exponent) {
    // Return (0,0) if base is zero
    if (length(base) == 0.0) {
        return vec2(0.0, 0.0);
    }
    return cexp(cmul(exponent, clog(base)));
}

// ---------------------------------------------
// Section 3: Trigonometric & Inverse Trig Funcs
// ---------------------------------------------

// Complex square root
vec2 csqrt(vec2 z) {
    float r = length(z);
    return sqrt(r) * vec2(sqrt(0.5 * (r + z.x)), sign(z.y) * sqrt(0.5 * (r - z.x)));
}

// Complex sine: sin(x+iy) = sin(x)cosh(y) + i*cos(x)sinh(y)
vec2 csin(vec2 z) {
    return vec2(sin(z.x) * cosh(z.y), cos(z.x) * sinh(z.y));
}

// Complex cosine: cos(x+iy) = cos(x)cosh(y) - i*sin(x)sinh(y)
vec2 ccos(vec2 z) {
    return vec2(cos(z.x) * cosh(z.y), -sin(z.x) * sinh(z.y));
}

// Complex tangent: tan(z) = csin(z) / ccos(z)
vec2 ctan(vec2 z) {
    return cdiv(csin(z), ccos(z));
}

// Complex arcsin: asin(z) = -i * log(i*z + sqrt(1-z^2))
vec2 casin(vec2 z) {
    vec2 i = vec2(0.0, 1.0);
    vec2 one = vec2(1.0, 0.0);
    vec2 log_val = clog(cmul(i, z) + csqrt(one - cmul(z, z)));
    // multiply by -i (which is vec2(0, -1))
    return vec2(log_val.y, -log_val.x);
}

// Complex arccos: acos(z) = pi/2 - asin(z)
vec2 cacos(vec2 z) {
    const float PI_2 = 1.57079632679;
    return vec2(PI_2, 0.0) - casin(z);
}

// Complex arctan: atan(z) = (i/2) * log((i-z)/(i+z))
vec2 catan(vec2 z) {
    vec2 i = vec2(0.0, 1.0);
    vec2 num = i - z;
    vec2 den = i + z;
    return cmul(vec2(0.0, 0.5), clog(cdiv(num, den)));
}
