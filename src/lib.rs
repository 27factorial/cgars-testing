#![allow(incomplete_features)]
#![feature(const_generics)]
#![feature(const_evaluatable_checked)]
#![feature(bench_black_box)]
#![feature(slice_as_chunks)]
#![feature(box_syntax)]

pub mod num;
pub mod ops;

use num::single::Scalar;

#[repr(C, align(128))]
#[derive(Copy, Clone, Debug)]
pub struct AlignedScalarArray<T: Scalar, const N: usize>([T; N]);

impl<T: Scalar, const N: usize> AlignedScalarArray<T, { N }> {
    pub fn new(array: [T; N]) -> Self {
        Self(array)
    }

    pub fn as_array(&self) -> &[T; N] {
        &self.0
    }

    pub fn as_array_mut(&mut self) -> &mut [T; N] {
        &mut self.0
    }

    pub fn as_slice(&self) -> &[T] {
        &self.0
    }

    pub fn as_slice_mut(&mut self) -> &mut [T] {
        &mut self.0
    }

    pub fn into_inner(self) -> [T; N] {
        self.0
    }
}
