use crate::num::simd::SimdScalar;
use crate::num::single::Scalar;
use std::ptr;

#[repr(C, align(128))]
struct Align128Array<T, const N: usize>([T; N]);

unsafe fn simd_mul_aligned_slice<T: Scalar>(a: &[T], b: &[T], acc: &mut [T]) {
    let packed_a = T::Simd::load_unaligned_unchecked(a);
    let packed_b = T::Simd::load_unaligned_unchecked(b);

    let result: T::Simd = packed_a * packed_b;

    result.store_unaligned_unchecked(acc);
}

pub unsafe fn dot_product<T: Scalar, const N: usize>(a: &[T; N], b: &[T; N]) -> T
where
    [T; T::Simd::LANES]: ,
{
    let mut acc = [T::ZERO; N];

    let (chunks_a, remainder_a) = a.as_chunks::<{ T::Simd::LANES }>();
    let (chunks_b, remainder_b) = b.as_chunks::<{ T::Simd::LANES }>();
    let (chunks_acc, remainder_acc) = acc.as_chunks_mut::<{ T::Simd::LANES }>();

    let zipped = chunks_a.iter().zip(chunks_b.iter()).zip(chunks_acc);

    for ((slice_a, slice_b), slice_acc) in zipped {
        simd_mul_aligned_slice(slice_a, slice_b, slice_acc);
    }

    let remainder_len = remainder_a.len();

    if remainder_len > 0 {
        let mut lanes_a = [T::ZERO; T::Simd::LANES];
        let mut lanes_b = [T::ZERO; T::Simd::LANES];
        let mut lanes_acc = [T::ZERO; T::Simd::LANES];

        lanes_a[..remainder_len].copy_from_slice(remainder_a);
        lanes_b[..remainder_len].copy_from_slice(remainder_b);
        lanes_acc[..remainder_len].copy_from_slice(remainder_acc);

        simd_mul_aligned_slice(&lanes_a, &lanes_b, &mut lanes_acc);
    }

    let mut sum = T::ZERO;

    for elem in a {
        sum += *elem;
    }

    sum
}

// fn dot_prod_mono() {
//     let mut array_a = Align128Array([f32::ONE; 16384 * 4]);
//     let mut array_b = Align128Array([f32::ONE; 16384 * 4]);
//
//     unsafe { dot_product(&array_a.0, &array_b.0) }
// }

use std::time::Instant;

pub fn simd_dot_product() {
    let mut array_a = box Align128Array([f32::ONE; 16384 * 16]);
    let mut array_b = box Align128Array([f32::ONE; 16384 * 16]);

    let expected_result = 16384.0_f32 * 4.0;

    let real = unsafe { dot_product(&array_a.0, &array_b.0) };

    assert_eq!(real, expected_result);
}

#[cfg(test)]
pub mod simd_tests {
    use super::*;
    use std::time::Instant;

    #[test]
    pub fn simd_dot_product() {
        let array_a = box Align128Array([f32::ONE; 16384 * 16]);
        let array_b = box Align128Array([f32::ONE; 16384 * 16]);

        let expected_result = 16384.0_f32 * 16.0;

        {
            unsafe {
                dot_product(&array_a.0, &array_b.0);
                dot_product(&array_a.0, &array_b.0);
                dot_product(&array_a.0, &array_b.0);
                dot_product(&array_a.0, &array_b.0);
                dot_product(&array_a.0, &array_b.0);
                dot_product(&array_a.0, &array_b.0);
                dot_product(&array_a.0, &array_b.0);
                dot_product(&array_a.0, &array_b.0);
                dot_product(&array_a.0, &array_b.0);
                dot_product(&array_a.0, &array_b.0);
                dot_product(&array_a.0, &array_b.0);
                dot_product(&array_a.0, &array_b.0);
                dot_product(&array_a.0, &array_b.0);
                dot_product(&array_a.0, &array_b.0);
                dot_product(&array_a.0, &array_b.0);
                dot_product(&array_a.0, &array_b.0);
                dot_product(&array_a.0, &array_b.0);
                dot_product(&array_a.0, &array_b.0);
            }
        }

        let simd_start = Instant::now();
        let real = unsafe { dot_product(&array_a.0, &array_b.0) };
        let simd_elapsed = simd_start.elapsed();

        let mut sum = f32::ZERO;

        let scalar_start = Instant::now();
        for (a, b) in array_a.0.iter().zip(array_b.0.iter()) {
            sum += std::hint::black_box(*a) * std::hint::black_box(*b);
        }
        let scalar_elapsed = scalar_start.elapsed();

        eprintln!("simd = {:?}, scalar = {:?}", simd_elapsed, scalar_elapsed);

        assert_eq!(real, expected_result);
    }
}
