use crate::num::simd::SimdScalar;
use crate::num::simd::{
    f32s, f64s, i128s, i16s, i32s, i64s, i8s, isizes, u128s, u16s, u32s, u64s, u8s, usizes,
};

pub trait Scalar: sealed::Ops {
    const ZERO: Self;
    const ONE: Self;

    type Simd: SimdScalar<Single = Self>;

    fn into_real<R: Real>(self) -> R;
    fn clamp(self, min: Self, max: Self) -> Self;
}

pub trait Signed: Scalar {
    const NEG_ONE: Self;

    fn abs(self) -> Self;
    fn signum(self) -> Self;
    fn negate(self) -> Self;
    fn is_positive(self) -> bool;
    fn is_negative(self) -> bool;
}

pub trait Real: Signed {
    const PI: Self;
    const TAU: Self;
    const FRAC_PI_2: Self;
    const FRAC_PI_3: Self;
    const FRAC_PI_4: Self;
    const FRAC_PI_6: Self;
    const FRAC_PI_8: Self;
    const FRAC_1_PI: Self;
    const FRAC_2_PI: Self;
    const FRAC_2_SQRT_PI: Self;
    const SQRT_2: Self;
    const FRAC_1_SQRT_2: Self;
    const E: Self;
    const LOG2_10: Self;
    const LOG2_E: Self;
    const LOG10_2: Self;
    const LOG10_E: Self;
    const LN_2: Self;
    const LN_10: Self;

    fn floor(self) -> Self;
    fn ceil(self) -> Self;
    fn round(self) -> Self;
    fn trunc(self) -> Self;
    fn fract(self) -> Self;
    fn mul_add(self, a: Self, b: Self) -> Self;
    fn div_euclid(self, rhs: Self) -> Self;
    fn rem_euclid(self, rhs: Self) -> Self;
    fn powi(self, n: i32) -> Self;
    fn powr(self, n: Self) -> Self;
    fn sqrt(self) -> Self;
    fn exp(self) -> Self;
    fn exp2(self) -> Self;
    fn ln(self) -> Self;
    fn log(self, base: Self) -> Self;
    fn log2(self) -> Self;
    fn log10(self) -> Self;
    fn cbrt(self) -> Self;
    fn hypot(self, other: Self) -> Self;
    fn sin(self) -> Self;
    fn cos(self) -> Self;
    fn tan(self) -> Self;
    fn asin(self) -> Self;
    fn acos(self) -> Self;
    fn atan(self) -> Self;
    fn atan2(self, other: Self) -> Self;
    fn sin_cos(self) -> (Self, Self);
    fn exp_m1(self) -> Self;
    fn ln_1p(self) -> Self;
    fn sinh(self) -> Self;
    fn cosh(self) -> Self;
    fn tanh(self) -> Self;
    fn asinh(self) -> Self;
    fn acosh(self) -> Self;
    fn atanh(self) -> Self;
    fn to_degrees(self) -> Self;
    fn to_radians(self) -> Self;
    fn from_scalar<T: Scalar>(t: T) -> Self;
    fn approx_eq(self, other: Self, epsilon: Self, max_relative: Self) -> bool;
}

macro_rules! impl_scalar {
    ($($t:ty, $simd:ty, $zero:literal, $one:literal);*$(;)?) => {
        $(
            impl $crate::num::single::Scalar for $t {
                const ZERO: Self = $zero;
                const ONE: Self = $one;

                type Simd = $simd;

                fn into_real<R: Real>(self) -> R {
                    R::from_scalar(self)
                }

                fn clamp(self, min: Self, max: Self) -> Self {
                    if self < min {
                        min
                    } else if self > max {
                        max
                    } else {
                        self
                    }
                }
            }
        )*
    }
}

macro_rules! impl_signed {
    ($($t:ty, $neg_one:literal, $pos:ident, $neg:ident);*$(;)?) => {
        $(
            impl $crate::num::single::Signed for $t {
                const NEG_ONE: Self = $neg_one;

                fn abs(self) -> Self {
                    <$t>::abs(self)
                }

                fn signum(self) -> Self {
                    <$t>::signum(self)
                }

                fn negate(self) -> Self {
                    -self
                }

                fn is_positive(self) -> bool {
                    <$t>::$pos(self)
                }

                fn is_negative(self) -> bool {
                    <$t>::$neg(self)
                }
            }
        )*
    }
}

macro_rules! impl_real {
    ($($t:ident, $convert:ident);*$(;)?) => {
        $(
            impl $crate::num::single::Real for $t {
                const PI: Self = ::core::$t::consts::PI;
                const TAU: Self = ::core::$t::consts::TAU;
                const FRAC_PI_2: Self = ::core::$t::consts::FRAC_PI_2;
                const FRAC_PI_3: Self = ::core::$t::consts::FRAC_PI_3;
                const FRAC_PI_4: Self = ::core::$t::consts::FRAC_PI_4;
                const FRAC_PI_6: Self = ::core::$t::consts::FRAC_PI_6;
                const FRAC_PI_8: Self = ::core::$t::consts::FRAC_PI_8;
                const FRAC_1_PI: Self = ::core::$t::consts::FRAC_1_PI;
                const FRAC_2_PI: Self = ::core::$t::consts::FRAC_2_PI;
                const FRAC_2_SQRT_PI: Self = ::core::$t::consts::FRAC_2_SQRT_PI;
                const SQRT_2: Self = ::core::$t::consts::SQRT_2;
                const FRAC_1_SQRT_2: Self = ::core::$t::consts::FRAC_1_SQRT_2;
                const E: Self = ::core::$t::consts::E;
                const LOG2_10: Self = ::core::$t::consts::LOG2_10;
                const LOG2_E: Self = ::core::$t::consts::LOG2_E;
                const LOG10_2: Self = ::core::$t::consts::LOG10_2;
                const LOG10_E: Self = ::core::$t::consts::LOG10_2;
                const LN_2: Self = ::core::$t::consts::LN_2;
                const LN_10: Self = ::core::$t::consts::LN_10;

                fn floor(self) -> Self {
                    <$t>::floor(self)
                }

                fn ceil(self) -> Self {
                    <$t>::ceil(self)
                }

                fn round(self) -> Self {
                    <$t>::round(self)
                }

                fn trunc(self) -> Self {
                    <$t>::trunc(self)
                }

                fn fract(self) -> Self {
                    <$t>::fract(self)
                }

                fn mul_add(self, a: Self, b: Self) -> Self {
                    <$t>::mul_add(self, a, b)
                }

                fn div_euclid(self, rhs: Self) -> Self {
                    <$t>::div_euclid(self, rhs)
                }

                fn rem_euclid(self, rhs: Self) -> Self {
                    <$t>::rem_euclid(self, rhs)
                }

                fn powi(self, n: i32) -> Self {
                    <$t>::powi(self, n)
                }

                fn powr(self, n: Self) -> Self {
                    <$t>::powf(self, n)
                }

                fn sqrt(self) -> Self {
                    <$t>::sqrt(self)
                }

                fn exp(self) -> Self {
                    <$t>::exp(self)
                }

                fn exp2(self) -> Self {
                    <$t>::exp2(self)
                }

                fn ln(self) -> Self {
                    <$t>::ln(self)
                }

                fn log(self, base: Self) -> Self {
                    <$t>::log(self, base)
                }

                fn log2(self) -> Self {
                    <$t>::log2(self)
                }

                fn log10(self) -> Self {
                    <$t>::log10(self)
                }

                fn cbrt(self) -> Self {
                    <$t>::cbrt(self)
                }

                fn hypot(self, other: Self) -> Self {
                    <$t>::hypot(self, other)
                }

                fn sin(self) -> Self {
                    <$t>::sin(self)
                }

                fn cos(self) -> Self {
                    <$t>::cos(self)
                }

                fn tan(self) -> Self {
                    <$t>::tan(self)
                }

                fn asin(self) -> Self {
                    <$t>::asin(self)
                }

                fn acos(self) -> Self {
                    <$t>::acos(self)
                }

                fn atan(self) -> Self {
                    <$t>::atan(self)
                }

                fn atan2(self, other: Self) -> Self {
                    <$t>::atan2(self, other)
                }

                fn sin_cos(self) -> (Self, Self) {
                    <$t>::sin_cos(self)
                }

                fn exp_m1(self) -> Self {
                    <$t>::exp_m1(self)
                }

                fn ln_1p(self) -> Self {
                    <$t>::ln_1p(self)
                }

                fn sinh(self) -> Self {
                    <$t>::sinh(self)
                }

                fn cosh(self) -> Self {
                    <$t>::cosh(self)
                }

                fn tanh(self) -> Self {
                    <$t>::tanh(self)
                }

                fn asinh(self) -> Self {
                    <$t>::asinh(self)
                }

                fn acosh(self) -> Self {
                    <$t>::acosh(self)
                }

                fn atanh(self) -> Self {
                    <$t>::atanh(self)
                }

                fn to_degrees(self) -> Self {
                    <$t>::to_degrees(self)
                }

                fn to_radians(self) -> Self {
                    <$t>::to_radians(self)
                }

                fn from_scalar<T: Scalar>(t: T) -> Self {
                    <T as $crate::num::single::sealed::ToFloat>::$convert(t)
                }

                fn approx_eq(self, other: Self, epsilon: Self, max_relative: Self) -> bool {
                    if self == other {
                        true
                    } else if self.is_infinite() || other.is_infinite() || self.is_nan() || other.is_nan() {
                        false
                    } else {
                        let diff = (self - other).abs();

                        if diff <= epsilon {
                            true
                        } else {
                            let abs_self = self.abs();
                            let abs_other = other.abs();

                            let largest = Self::max(abs_self, abs_other);

                            diff < largest * max_relative
                        }
                    }


                }
            }
        )*
    }
}

impl_scalar! {
    i8, i8s, 0, 1;
    u8, u8s, 0, 1;
    i16, i16s, 0, 1;
    u16, u16s, 0, 1;
    i32, i32s, 0, 1;
    u32, u32s, 0, 1;
    i64, i64s, 0, 1;
    u64, u64s, 0, 1;
    i128, i128s, 0, 1;
    u128, u128s, 0, 1;
    isize, isizes, 0, 1;
    usize, usizes, 0, 1;
    f32, f32s, 0.0, 1.0;
    f64, f64s, 0.0, 1.0;
}

impl_signed! {
    i8, -1, is_positive, is_negative;
    i16, -1, is_positive, is_negative;
    i32, -1, is_positive, is_negative;
    i64, -1, is_positive, is_negative;
    i128, -1, is_positive, is_negative;
    isize, -1, is_positive, is_negative;
    f32, -1.0, is_sign_positive, is_sign_negative;
    f64, -1.0, is_sign_positive, is_sign_negative;
}

impl_real! {
    f32, to_f32;
    f64, to_f64;
}

mod sealed {
    use std::{
        cmp::{PartialEq, PartialOrd},
        fmt::Debug,
        ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Rem, RemAssign, Sub, SubAssign},
    };

    pub trait Ops:
        Copy
        + Clone
        + Debug
        + Add<Output = Self>
        + AddAssign
        + Div<Output = Self>
        + DivAssign
        + Mul<Output = Self>
        + MulAssign
        + Rem<Output = Self>
        + RemAssign
        + Sub<Output = Self>
        + SubAssign
        + PartialEq
        + PartialOrd
        + ToFloat
    {
    }

    pub trait ToFloat: Copy {
        fn to_f64(self) -> f64;
        fn to_f32(self) -> f32;
    }

    macro_rules! impl_ops {
        ($($t:ty),*) => {
            $(
                impl $crate::num::single::sealed::ToFloat for $t {
                    fn to_f64(self) -> f64 {
                        self as f64
                    }

                    fn to_f32(self) -> f32 {
                        self as f32
                    }
                }

                impl $crate::num::single::sealed::Ops for $t {}
            )*
        }
    }

    impl_ops! {
        i8, u8, i16, u16, i32, u32, i64, u64, i128, u128, isize, usize, f32, f64
    }
}
