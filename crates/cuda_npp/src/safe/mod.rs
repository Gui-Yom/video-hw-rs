use core::ffi::c_void;
use cudarc::driver::sys::CUdeviceptr;
use cudarc::driver::{DevicePtr, DevicePtrMut, DeviceSlice};
use std::marker::PhantomData;
use std::mem;
use std::ptr::null_mut;

use cuda_npp_sys::{cudaFree, cudaMalloc, cudaMallocArray, NppStatus, NppiSize};

use crate::{Channel, Sample};

#[cfg(feature = "ial")]
pub mod ial;
#[cfg(feature = "icc")]
pub mod icc;
#[cfg(feature = "idei")]
pub mod idei;
#[cfg(feature = "if")]
pub mod if_;
#[cfg(feature = "ig")]
pub mod ig;
#[cfg(feature = "ist")]
pub mod ist;
#[cfg(feature = "isu")]
pub mod isu;

#[derive(Debug)]
pub enum E {
    NppError(NppStatus),
}

impl From<NppStatus> for E {
    fn from(value: NppStatus) -> Self {
        E::NppError(value)
    }
}

pub type Result<T> = std::result::Result<T, E>;

#[derive(Debug)]
pub struct Image<S: Sample, C: Channel> {
    pub width: u32,
    pub height: u32,
    /// Line step in bytes
    pub line_step: i32,
    pub data: u64,
    marker_: PhantomData<S>,
    marker__: PhantomData<C>,
}

impl<S: Sample, C: Channel> DeviceSlice<S> for Image<S, C> {
    fn len(&self) -> usize {
        (self.line_step as u32 * self.height) as usize
    }
}

impl<S: Sample, C: Channel> DevicePtr<S> for Image<S, C> {
    fn device_ptr(&self) -> &CUdeviceptr {
        &self.data
    }
}

impl<S: Sample, C: Channel> DevicePtrMut<S> for Image<S, C> {
    fn device_ptr_mut(&mut self) -> &mut CUdeviceptr {
        &mut self.data
    }
}

impl<S: Sample, C: Channel> Image<S, C> {
    pub fn size(&self) -> NppiSize {
        NppiSize {
            width: self.width as i32,
            height: self.height as i32,
        }
    }

    pub fn has_padding(&self) -> bool {
        dbg!(self.width as usize * C::NUM_SAMPLES * mem::size_of::<S>())
            != dbg!(self.line_step as usize)
    }

    #[cfg(feature = "cudarc")]
    pub fn copy_from_cpu(&mut self, data: &[S]) -> Result<()> {
        if self.has_padding() {
            for c in 0..self.height as usize {
                unsafe {
                    cudarc::driver::result::memcpy_htod_sync(
                        self.data + c as u64 * self.line_step as u64,
                        &data[c * self.width as usize * C::NUM_SAMPLES
                            ..(c + 1) * self.width as usize * C::NUM_SAMPLES],
                    )
                }
                .unwrap();
            }
            Ok(())
        } else {
            let res = unsafe { cudarc::driver::result::memcpy_htod_sync(self.data as _, data) };
            res.unwrap();
            Ok(())
        }
    }
}

#[cfg(feature = "cudarc")]
impl<S: Sample + Default + Copy, C: Channel> Image<S, C> {
    pub fn copy_to_cpu(&self) -> Result<Vec<S>> {
        let mut dst =
            vec![S::default(); self.width as usize * self.height as usize * C::NUM_SAMPLES];
        if self.has_padding() {
            for c in 0..self.height as usize {
                unsafe {
                    cudarc::driver::result::memcpy_dtoh_sync(
                        &mut dst[c * self.width as usize * C::NUM_SAMPLES
                            ..(c + 1) * self.width as usize * C::NUM_SAMPLES],
                        self.data + c as u64 * self.line_step as u64,
                    )
                }
                .unwrap();
            }
            Ok(dst)
        } else {
            let res = unsafe { cudarc::driver::result::memcpy_dtoh_sync(&mut dst, self.data as _) };
            res.unwrap();
            Ok(dst)
        }
    }
}

#[derive(Debug, Clone)]
pub struct ScratchBuffer {
    pub ptr: *mut c_void,
    pub size: usize,
}

impl ScratchBuffer {
    pub fn alloc(size: usize) -> Self {
        let mut ptr = null_mut();
        unsafe {
            cudaMalloc(&mut ptr, size);
        }
        Self { ptr, size }
    }
}

impl Drop for ScratchBuffer {
    fn drop(&mut self) {
        unsafe {
            cudaFree(self.ptr);
        }
    }
}

// #[cfg(test)]
// mod tests {
//     use crate::C;
//     use crate::safe::Image;
//     use crate::safe::isu::Malloc;
//
//     #[test]
//     fn copy_and_back() {
//         let source_bytes = &source.flatten_to_u8()[0];
//         let mut source_img =
//             Image::<u8, C<3>>::malloc(source.dimensions().0 as u32, source.dimensions().1 as u32)
//                 .unwrap();
//         source_img.copy_from_cpu(&source_bytes).unwrap();
//
//         let source_back = source_img.copy_to_cpu().unwrap();
//         assert_eq!(source_bytes, &source_back);
//     }
// }
