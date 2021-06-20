#![allow(incomplete_features)]
#![feature(const_generics)]
#![feature(const_evaluatable_checked)]
#![feature(bench_black_box)]
#![feature(slice_as_chunks)]
#![feature(box_syntax)]

use cgars::num::simd::SimdScalar;
use cgars::num::single::Scalar;
use cgars::ops::simd::dot_product;
use std::time::Instant;

#[repr(C, align(128))]
struct Align128Array<T, const N: usize>([T; N]);

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
