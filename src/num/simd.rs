#[allow(unused_imports)]
use packed_simd::{
    f32x16, f32x2, f32x4, f32x8, f64x2, f64x4, f64x8, i128x1, i128x2, i128x4, i16x16, i16x2,
    i16x32, i16x4, i16x8, i32x16, i32x2, i32x4, i32x8, i64x2, i64x4, i64x8, i8x16, i8x2, i8x32,
    i8x4, i8x64, i8x8, isizex2, isizex4, isizex8, m128x1, m128x2, m128x4, m16x16, m16x2, m16x32,
    m16x4, m16x8, m32x16, m32x2, m32x4, m32x8, m64x2, m64x4, m64x8, m8x16, m8x2, m8x32, m8x4,
    m8x64, m8x8, msizex2, msizex4, msizex8, u128x1, u128x2, u128x4, u16x16, u16x2, u16x32, u16x4,
    u16x8, u32x16, u32x2, u32x4, u32x8, u64x2, u64x4, u64x8, u8x16, u8x2, u8x32, u8x4, u8x64, u8x8,
    usizex2, usizex4, usizex8,
};

use crate::num::single::Scalar;

pub trait SimdScalar: sealed::SimdOps {
    const LANES: usize;

    type Single: Scalar<Simd = Self>;
    type Mask: SimdMask;

    fn from_array(data: [Self::Single; Self::LANES]) -> Self;
    fn from_single(single: Self::Single) -> Self;
    fn load_unaligned(slice: &[Self::Single]) -> Self;
    unsafe fn load_unaligned_unchecked(slice: &[Self::Single]) -> Self;
    fn load_aligned(slice: &[Self::Single]) -> Self;
    unsafe fn load_aligned_unchecked(slice: &[Self::Single]) -> Self;
    fn store_unaligned(self, slice: &mut [Self::Single]);
    unsafe fn store_unaligned_unchecked(self, slice: &mut [Self::Single]);
    fn store_aligned(self, slice: &mut [Self::Single]);
    unsafe fn store_aligned_unchecked(self, slice: &mut [Self::Single]);
    fn extract(self, index: usize) -> Self::Single;
    unsafe fn extract_unchecked(self, index: usize) -> Self::Single;
    fn replace(self, index: usize, new: Self::Single) -> Self;
    unsafe fn replace_unchecked(self, index: usize, new: Self::Single) -> Self;
    fn min(self, other: Self) -> Self;
    fn max(self, other: Self) -> Self;
    fn max_element(self) -> Self::Single;
    fn min_element(self) -> Self::Single;
    fn element_sum(self) -> Self::Single;
    fn element_product(self) -> Self::Single;
    fn eq(self, other: Self) -> Self::Mask;
    fn ne(self, other: Self) -> Self::Mask;
    fn ge(self, other: Self) -> Self::Mask;
    fn le(self, other: Self) -> Self::Mask;
    fn gt(self, other: Self) -> Self::Mask;
    fn lt(self, other: Self) -> Self::Mask;
    fn select_from(self, other: Self, mask: Self::Mask) -> Self;
}

pub trait SimdMask {
    const LANES: usize;

    fn from_value(v: bool) -> Self;
    fn extract(self, index: usize) -> bool;
    unsafe fn extract_unchecked(self, index: usize) -> bool;
    fn replace(self, index: usize, new: bool) -> Self;
    unsafe fn replace_unchecked(self, index: usize, new: bool) -> Self;
    fn element_and(self) -> bool;
    fn element_or(self) -> bool;
    fn element_xor(self) -> bool;
    fn all(self) -> bool;
    fn any(self) -> bool;
    fn none(self) -> bool;
    fn eq(self, other: Self) -> Self;
    fn ne(self, other: Self) -> Self;
    fn ge(self, other: Self) -> Self;
    fn le(self, other: Self) -> Self;
    fn gt(self, other: Self) -> Self;
    fn lt(self, other: Self) -> Self;
    fn select<S: SimdScalar<Mask = Self>>(self, a: S, b: S) -> S;
}

macro_rules! create_and_impl_simd_scalars {
    ($($scalar:tt, $name:tt, $packed:tt, $mask:tt, $lanes:expr, $sum:ident, $prod:ident, enabled = [$($feat:expr),*], disabled = [$($not_feat:expr),*]);*$(;)?) => {
        $(
            #[allow(non_camel_case_types)]
            #[cfg(all($(target_feature = $feat,)* not($(target_feature = $not_feat,)*)))]
            pub type $name = $packed;

            #[cfg(all($(target_feature = $feat,)* not($(target_feature = $not_feat,)*)))]
            impl $crate::num::simd::SimdScalar for $name {
                const LANES: usize = $lanes;

                type Single = $scalar;
                type Mask = $mask;

                fn from_array(data: [Self::Single; Self::LANES]) -> Self {
                    // SAFETY: Self::LANES is always equal to $packed::lanes(), therefore
                    // the slice is always exactly as long as it needs to be. We use an
                    // unaligned load because we can not be certain that the `data` array
                    // is aligned properly.
                    unsafe {
                        $packed::from_slice_unaligned_unchecked(&data[..])
                    }
                }

                fn from_single(single: Self::Single) -> Self {
                    $packed::splat(single)
                }

                fn load_unaligned(slice: &[Self::Single]) -> Self {
                    $packed::from_slice_unaligned(slice)
                }

                unsafe fn load_unaligned_unchecked(slice: &[Self::Single]) -> Self {
                    $packed::from_slice_unaligned_unchecked(slice)
                }

                fn load_aligned(slice: &[Self::Single]) -> Self {
                    $packed::from_slice_aligned(slice)
                }

                unsafe fn load_aligned_unchecked(slice: &[Self::Single]) -> Self {
                    $packed::from_slice_aligned_unchecked(slice)
                }

                fn store_unaligned(self, slice: &mut [Self::Single]) {
                    <$packed>::write_to_slice_unaligned(self, slice)
                }

                unsafe fn store_unaligned_unchecked(self, slice: &mut [Self::Single]) {
                    <$packed>::write_to_slice_unaligned_unchecked(self, slice)
                }

                fn store_aligned(self, slice: &mut [Self::Single]) {
                    <$packed>::write_to_slice_aligned(self, slice)
                }

                unsafe fn store_aligned_unchecked(self, slice: &mut [Self::Single]) {
                    <$packed>::write_to_slice_aligned_unchecked(self, slice)
                }

                fn extract(self, index: usize) -> Self::Single {
                    <$packed>::extract(self, index)
                }

                unsafe fn extract_unchecked(self, index: usize) -> Self::Single {
                    <$packed>::extract_unchecked(self, index)
                }

                fn replace(self, index: usize, new: Self::Single) -> Self {
                    <$packed>::replace(self, index, new)
                }

                unsafe fn replace_unchecked(self, index: usize, new: Self::Single) -> Self {
                    <$packed>::replace_unchecked(self, index, new)
                }

                fn min(self, other: Self) -> Self {
                    <$packed>::min(self, other)
                }

                fn max(self, other: Self) -> Self {
                    <$packed>::max(self, other)
                }

                fn max_element(self) -> Self::Single {
                    <$packed>::max_element(self)
                }

                fn min_element(self) -> Self::Single {
                    <$packed>::min_element(self)
                }

                fn element_sum(self) -> Self::Single {
                    <$packed>::$sum(self)
                }

                fn element_product(self) -> Self::Single {
                    <$packed>::$prod(self)
                }

                fn eq(self, other: Self) -> Self::Mask {
                    <$packed>::eq(self, other)
                }

                fn ne(self, other: Self) -> Self::Mask {
                    <$packed>::ne(self, other)
                }

                fn ge(self, other: Self) -> Self::Mask {
                    <$packed>::ge(self, other)
                }

                fn le(self, other: Self) -> Self::Mask {
                    <$packed>::le(self, other)
                }

                fn gt(self, other: Self) -> Self::Mask {
                    <$packed>::gt(self, other)
                }

                fn lt(self, other: Self) -> Self::Mask {
                    <$packed>::lt(self, other)
                }

                fn select_from(self, other: Self, mask: Self::Mask) -> Self {
                    <$mask>::select(mask, self, other)
                }
            }
        )*

    };
}

macro_rules! create_and_impl_simd_masks {
    ($($name:tt, $packed:tt, $lanes:expr, enabled = [$($feat:expr),*], disabled = [$($not_feat:expr),*]);*$(;)?) => {
        $(
            #[allow(non_camel_case_types)]
            #[cfg(all($(target_feature = $feat,)* not($(target_feature = $not_feat,)*)))]
            type $name = $packed;

            #[cfg(all($(target_feature = $feat,)* not($(target_feature = $not_feat,)*)))]
            impl $crate::num::simd::SimdMask for $name {
                const LANES: usize = $lanes;

                fn from_value(v: bool) -> Self {
                    $packed::splat(v)
                }

                fn extract(self, index: usize) -> bool {
                    <$packed>::extract(self, index)
                }

                unsafe fn extract_unchecked(self, index: usize) -> bool {
                    <$packed>::extract_unchecked(self, index)
                }

                fn replace(self, index: usize, new: bool) -> Self {
                    <$packed>::replace(self, index, new)
                }

                unsafe fn replace_unchecked(self, index: usize, new: bool) -> Self {
                    <$packed>::replace_unchecked(self, index, new)
                }

                fn element_and(self) -> bool {
                    <$packed>::and(self)
                }

                fn element_or(self) -> bool {
                    <$packed>::or(self)
                }

                fn element_xor(self) -> bool {
                    <$packed>::xor(self)
                }

                fn all(self) -> bool {
                    <$packed>::all(self)
                }

                fn any(self) -> bool {
                    <$packed>::any(self)
                }

                fn none(self) -> bool {
                    <$packed>::none(self)
                }

                fn eq(self, other: Self) -> Self {
                    <$packed>::eq(self, other)
                }

                fn ne(self, other: Self) -> Self {
                    <$packed>::ne(self, other)
                }

                fn ge(self, other: Self) -> Self {
                    <$packed>::ge(self, other)
                }

                fn le(self, other: Self) -> Self {
                    <$packed>::le(self, other)
                }

                fn gt(self, other: Self) -> Self {
                    <$packed>::gt(self, other)
                }

                fn lt(self, other: Self) -> Self {
                    <$packed>::lt(self, other)
                }

                fn select<S: $crate::num::simd::SimdScalar<Mask = Self>>(self, a: S, b: S) -> S {
                    S::select_from(a, b, self)
                }
            }
        )*

    };
}

create_and_impl_simd_scalars! {
    // i8 on x86
    i8, i8s, i8x8, m8x8, 8, wrapping_sum, wrapping_product, enabled = [], disabled = ["avx"];
    i8, i8s, i8x16, m8x16, 16, wrapping_sum, wrapping_product, enabled = ["avx"], disabled = ["avx2"];
    i8, i8s, i8x32, m8x32, 32, wrapping_sum, wrapping_product, enabled = ["avx2"], disabled = ["avx512"];
    i8, i8s, i8x64, m8x64, 64, wrapping_sum, wrapping_product, enabled = ["avx512"], disabled = [];

    // u8 on x86
    u8, u8s, u8x8, m8x8, 8, wrapping_sum, wrapping_product, enabled = [], disabled = ["avx"];
    u8, u8s, u8x16, m8x16, 16, wrapping_sum, wrapping_product, enabled = ["avx"], disabled = ["avx2"];
    u8, u8s, u8x32, m8x32, 32, wrapping_sum, wrapping_product, enabled = ["avx2"], disabled = ["avx512"];
    u8, u8s, u8x64, m8x64, 64, wrapping_sum, wrapping_product, enabled = ["avx512"], disabled = [];

    // i16 on x86
    i16, i16s, i16x4, m16x4, 4, wrapping_sum, wrapping_product, enabled = [], disabled = ["avx"];
    i16, i16s, i16x8, m16x8, 8, wrapping_sum, wrapping_product, enabled = ["avx"], disabled = ["avx2"];
    i16, i16s, i16x16, m16x16, 16, wrapping_sum, wrapping_product, enabled = ["avx2"], disabled = ["avx512"];
    i16, i16s, i16x32, m16x32, 32, wrapping_sum, wrapping_product, enabled = ["avx512"], disabled = [];

    // u16 on x86
    u16, u16s, u16x4, m16x4, 4, wrapping_sum, wrapping_product, enabled = [], disabled = ["avx"];
    u16, u16s, u16x8, m16x8, 8, wrapping_sum, wrapping_product, enabled = ["avx"], disabled = ["avx2"];
    u16, u16s, u16x16, m16x16, 16, wrapping_sum, wrapping_product, enabled = ["avx2"], disabled = ["avx512"];
    u16, u16s, u16x32, m16x32, 32, wrapping_sum, wrapping_product, enabled = ["avx512"], disabled = [];

    // i32 on x86
    i32, i32s, i32x2, m32x2, 2, wrapping_sum, wrapping_product, enabled = [], disabled = ["avx"];
    i32, i32s, i32x4, m32x4, 4, wrapping_sum, wrapping_product, enabled = ["avx"], disabled = ["avx2"];
    i32, i32s, i32x8, m32x8, 8, wrapping_sum, wrapping_product, enabled = ["avx2"], disabled = ["avx512"];
    i32, i32s, i32x16, m32x16, 16, wrapping_sum, wrapping_product, enabled = ["avx512"], disabled = [];

    // u32 on x86
    u32, u32s, u32x2, m32x2, 2, wrapping_sum, wrapping_product, enabled = [], disabled = ["avx"];
    u32, u32s, u32x4, m32x4, 4, wrapping_sum, wrapping_product, enabled = ["avx"], disabled = ["avx2"];
    u32, u32s, u32x8, m32x8, 8, wrapping_sum, wrapping_product, enabled = ["avx2"], disabled = ["avx512"];
    u32, u32s, u32x16, m32x16, 16, wrapping_sum, wrapping_product, enabled = ["avx512"], disabled = [];

    // i64 on x86
    i64, i64s, i64x2, m64x2, 2, wrapping_sum, wrapping_product, enabled = [], disabled = ["avx"];
    i64, i64s, i64x2, m64x2, 2, wrapping_sum, wrapping_product, enabled = ["avx"], disabled = ["avx2"];
    i64, i64s, i64x4, m64x4, 4, wrapping_sum, wrapping_product, enabled = ["avx2"], disabled = ["avx512"];
    i64, i64s, i64x8, m64x8, 8, wrapping_sum, wrapping_product, enabled = ["avx512"], disabled = [];

    // u64 on x86
    u64, u64s, u64x2, m64x2, 2, wrapping_sum, wrapping_product, enabled = [], disabled = ["avx"];
    u64, u64s, u64x2, m64x2, 2, wrapping_sum, wrapping_product, enabled = ["avx"], disabled = ["avx2"];
    u64, u64s, u64x4, m64x4, 4, wrapping_sum, wrapping_product, enabled = ["avx2"], disabled = ["avx512"];
    u64, u64s, u64x8, m64x8, 8, wrapping_sum, wrapping_product, enabled = ["avx512"], disabled = [];

    // i128 on x86
    i128, i128s, i128x1, m128x1, 1, wrapping_sum, wrapping_product, enabled = [], disabled = ["avx"];
    i128, i128s, i128x1, m128x1, 1, wrapping_sum, wrapping_product, enabled = ["avx"], disabled = ["avx2"];
    i128, i128s, i128x2, m128x2, 2, wrapping_sum, wrapping_product, enabled = ["avx2"], disabled = ["avx512"];
    i128, i128s, i128x4, m128x4, 4, wrapping_sum, wrapping_product, enabled = ["avx512"], disabled = [];

    // u128 on x86
    u128, u128s, u128x1, m128x1, 1, wrapping_sum, wrapping_product, enabled = [], disabled = ["avx"];
    u128, u128s, u128x1, m128x1, 1, wrapping_sum, wrapping_product, enabled = ["avx"], disabled = ["avx2"];
    u128, u128s, u128x2, m128x2, 2, wrapping_sum, wrapping_product, enabled = ["avx2"], disabled = ["avx512"];
    u128, u128s, u128x4, m128x4, 4, wrapping_sum, wrapping_product, enabled = ["avx512"], disabled = [];

    // isize on x86
    isize, isizes, isizex2, msizex2, 2, wrapping_sum, wrapping_product, enabled = [], disabled = ["avx"];
    isize, isizes, isizex2, msizex2, 2, wrapping_sum, wrapping_product, enabled = ["avx"], disabled = ["avx2"];
    isize, isizes, isizex4, msizex4, 4, wrapping_sum, wrapping_product, enabled = ["avx2"], disabled = ["avx512"];
    isize, isizes, isizex8, msizex8, 8, wrapping_sum, wrapping_product, enabled = ["avx512"], disabled = [];

    // usize on x86
    usize, usizes, usizex2, msizex2, 2, wrapping_sum, wrapping_product, enabled = [], disabled = ["avx"];
    usize, usizes, usizex2, msizex2, 2, wrapping_sum, wrapping_product, enabled = ["avx"], disabled = ["avx2"];
    usize, usizes, usizex4, msizex4, 4, wrapping_sum, wrapping_product, enabled = ["avx2"], disabled = ["avx512"];
    usize, usizes, usizex8, msizex8, 8, wrapping_sum, wrapping_product, enabled = ["avx512"], disabled = [];

    // f32 on x86
    f32, f32s, f32x2, m32x2, 2, sum, product, enabled = [], disabled = ["avx"];
    f32, f32s, f32x8, m32x8, 8, sum, product, enabled = ["avx"], disabled = ["avx2"];
    f32, f32s, f32x8, m32x8, 8, sum, product, enabled = ["avx2"], disabled = ["avx512"];
    f32, f32s, f32x16, m32x16, 16, sum, product, enabled = ["avx512"], disabled = [];

    // f64 on x86
    f64, f64s, f64x2, m64x2, 2, sum, product, enabled = [], disabled = ["avx"];
    f64, f64s, f64x4, m64x4, 4, sum, product, enabled = ["avx"], disabled = ["avx2"];
    f64, f64s, f64x4, m64x4, 4, sum, product, enabled = ["avx2"], disabled = ["avx512"];
    f64, f64s, f64x8, m64x8, 8, sum, product, enabled = ["avx512"], disabled = [];
}

create_and_impl_simd_masks! {
    // m8 on x86
    m8s, m8x8, 8, enabled = [], disabled = ["avx"];
    m8s, m8x16, 16, enabled = ["avx"], disabled = ["avx2"];
    m8s, m8x32, 32, enabled = ["avx2"], disabled = ["avx512"];
    m8s, m8x64, 64, enabled = ["avx512"], disabled = [];

    // m16 on x86
    m16s, m16x4, 4, enabled = [], disabled = ["avx"];
    m16s, m16x8, 8, enabled = ["avx"], disabled = ["avx2"];
    m16s, m16x16, 16, enabled = ["avx2"], disabled = ["avx512"];
    m16s, m16x32, 32, enabled = ["avx512"], disabled = [];

    // m32 on x86
    m32s, m32x2, 2, enabled = [], disabled = ["avx"];
    m32s, m32x4, 4, enabled = ["avx"], disabled = ["avx2"];
    m32s, m32x8, 8, enabled = ["avx2"], disabled = ["avx512"];
    m32s, m32x16, 16, enabled = ["avx512"], disabled = [];

    // m32f on x86 for AVX (Special case for floating point with AVX)
    // Floating point with AVX has operations that require a 256 bit
    // mask, but usually only 128 bit masks are generated for AVX
    // (Unless AVX2 is supported). Ugh.
    m32fs, m32x8, 8, enabled = ["avx"], disabled = ["avx2"];

    // m64 on x86
    m64s, m64x2, 2, enabled = [], disabled = ["avx"];
    m64s, m64x2, 2, enabled = ["avx"], disabled = ["avx2"];
    m64s, m64x4, 4, enabled = ["avx2"], disabled = ["avx512"];
    m64s, m64x8, 8, enabled = ["avx512"], disabled = [];

    // m64f on x86 for AVX (Special case for floating point with AVX)
    // Floating point with AVX has operations that require a 256 bit
    // mask, but usually only 128 bit masks are generated for AVX
    // (Unless AVX2 is supported). Ugh.
    m64fs, m64x4, 4, enabled = ["avx"], disabled = ["avx2"];

    // m128 on x86
    m128s, m128x1, 1, enabled = [], disabled = ["avx"];
    m128s, m128x1, 1, enabled = ["avx"], disabled = ["avx2"];
    m128s, m128x2, 2, enabled = ["avx2"], disabled = ["avx512"];
    m128s, m128x4, 4, enabled = ["avx512"], disabled = [];

    // msize on x86
    msizes, msizex2, 2, enabled = [], disabled = ["avx"];
    msizes, msizex2, 2, enabled = ["avx"], disabled = ["avx2"];
    msizes, msizex4, 4, enabled = ["avx2"], disabled = ["avx512"];
    msizes, msizex8, 8, enabled = ["avx512"], disabled = [];
}

mod sealed {
    use std::{
        fmt::Debug,
        ops::{
            Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Div,
            DivAssign, Mul, MulAssign, Not, Rem, RemAssign, Sub, SubAssign,
        },
    };

    use packed_simd::{
        f32x16, f32x2, f32x4, f32x8, f64x2, f64x4, f64x8, i128x1, i128x2, i128x4, i16x16, i16x2,
        i16x32, i16x4, i16x8, i32x16, i32x2, i32x4, i32x8, i64x2, i64x4, i64x8, i8x16, i8x2, i8x32,
        i8x4, i8x64, i8x8, isizex2, isizex4, isizex8, m128x1, m128x2, m128x4, m16x16, m16x2,
        m16x32, m16x4, m16x8, m32x16, m32x2, m32x4, m32x8, m64x2, m64x4, m64x8, m8x16, m8x2, m8x32,
        m8x4, m8x64, m8x8, msizex2, msizex4, msizex8, u128x1, u128x2, u128x4, u16x16, u16x2,
        u16x32, u16x4, u16x8, u32x16, u32x2, u32x4, u32x8, u64x2, u64x4, u64x8, u8x16, u8x2, u8x32,
        u8x4, u8x64, u8x8, usizex2, usizex4, usizex8,
    };

    pub trait SimdOps:
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
    {
    }

    pub trait SimdMaskOps:
        Copy
        + Clone
        + Debug
        + BitAnd<Output = Self>
        + BitAndAssign
        + BitOr<Output = Self>
        + BitOrAssign
        + BitXor<Output = Self>
        + BitXorAssign
        + Not
    {
    }

    macro_rules! impl_simd_ops {
        ($($t:ty),*$(,)?) => {
            $(
                impl $crate::num::simd::sealed::SimdOps for $t {}
            )*
        };
    }

    impl_simd_ops! {
        i8x2, i8x4, i8x8, i8x16, i8x32, i8x64,
        u8x2, u8x4, u8x8, u8x16, u8x32, u8x64,
        i16x2, i16x4, i16x8, i16x16, i16x32,
        u16x2, u16x4, u16x8, u16x16, u16x32,
        i32x2, i32x4, i32x8, i32x16,
        u32x2, u32x4, u32x8, u32x16,
        i64x2, i64x4, i64x8,
        u64x2, u64x4, u64x8,
        i128x1, i128x2, i128x4,
        u128x1, u128x2, u128x4,
        isizex2, isizex4, isizex8,
        usizex2, usizex4, usizex8,
        f32x2, f32x4, f32x8, f32x16,
        f64x2, f64x4, f64x8,
    }

    macro_rules! impl_simd_mask_ops {
        ($($t:ty),*$(,)?) => {
            $(
                impl $crate::num::simd::sealed::SimdMaskOps for $t {}
            )*
        }
    }

    impl_simd_mask_ops! {
        m8x2, m8x4, m8x8, m8x16, m8x32, m8x64,
        m16x2, m16x4, m16x8, m16x16, m16x32,
        m32x2, m32x4, m32x8, m32x16,
        m64x2, m64x4, m64x8,
        m128x1, m128x2, m128x4,
        msizex2, msizex4, msizex8,
    }
}
