use crate::{sys, CuStream};
use std::borrow::Borrow;
use std::mem;
use std::mem::MaybeUninit;
use std::ops::{Deref, DerefMut};
use std::ptr::{slice_from_raw_parts_mut, NonNull};
use sys::CuResult;

/// Like [Box], but in Cuda global memory.
pub struct CuBox<T: ?Sized>(NonNull<T>);

impl<T: Sized> CuBox<[T]> {
    pub fn new(t: &[T], stream: &CuStream) -> CuResult<Self> {
        unsafe {
            let s = Self::new_uninit(t.len(), stream)?;
            sys::cuMemcpyHtoDAsync_v2(
                s.ptr(),
                t.as_ptr() as _,
                t.len() * size_of::<T>(),
                stream.raw(),
            )
            .result()?;
            Ok(s)
        }
    }

    /// Allocate an uninitialized slice in cuda global memory.
    pub fn new_uninit(len: usize, stream: &CuStream) -> CuResult<Self> {
        let mut ptr = 0;
        unsafe {
            sys::cuMemAllocAsync(&mut ptr, len * size_of::<T>(), stream.raw()).result()?;
        }
        Ok(Self(
            NonNull::new(slice_from_raw_parts_mut(ptr as _, len)).unwrap(),
        ))
    }

    /// Allocate a zeroed slice in cuda global memory.
    pub fn new_zeroed(len: usize, stream: &CuStream) -> CuResult<Self> {
        let mut s = Self::new_uninit(len, stream)?;
        s.clear(stream)?;
        Ok(s)
    }

    /// Create a CuBox by owning an external pointer.
    pub fn from_ptr_len(ptr: sys::CUdeviceptr, len: usize) -> Self {
        Self(NonNull::new(slice_from_raw_parts_mut(ptr as _, len)).unwrap())
    }

    /// Zero out memory
    pub fn clear(&mut self, stream: &CuStream) -> CuResult<()> {
        unsafe { sys::cuMemsetD8Async(self.ptr(), 0, self.len(), stream.raw()).result() }
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn cast<Target>(self) -> CuBox<[Target]> {
        let ptr = self.0.as_ptr().cast();
        let len = self.0.len();
        CuBox::from_ptr(slice_from_raw_parts_mut(
            ptr,
            len * size_of::<T>() / size_of::<Target>(),
        ))
    }
}

impl<T: Sized> CuBox<T> {
    pub fn new(t: &T, stream: &CuStream) -> CuResult<Self> {
        unsafe {
            let s = Self::new_uninit(stream)?;
            sys::cuMemcpyHtoDAsync_v2(s.ptr(), t as *const T as _, size_of::<T>(), stream.raw())
                .result()?;
            Ok(s)
        }
    }

    pub unsafe fn new_uninit(stream: &CuStream) -> CuResult<CuBox<T>> {
        unsafe {
            let mut ptr = 0;
            sys::cuMemAllocAsync(&mut ptr, size_of::<T>(), stream.raw()).result()?;
            Ok(Self(NonNull::new(ptr as _).unwrap()))
        }
    }

    pub unsafe fn new_zeroed(stream: &CuStream) -> CuResult<Self> {
        unsafe {
            let s = Self::new_uninit(stream)?;
            sys::cuMemsetD8Async(s.ptr(), 0, size_of::<T>(), stream.raw()).result()?;
            Ok(s)
        }
    }

    /// Create a CuBox by owning an external pointer.
    pub fn from_cuda_ptr(ptr: sys::CUdeviceptr) -> Self {
        Self(NonNull::new(ptr as _).unwrap())
    }

    pub fn copy_from_host(&mut self, t: &T, stream: &CuStream) -> CuResult<()> {
        unsafe {
            sys::cuMemcpyHtoDAsync_v2(self.ptr(), t as *const T as _, size_of::<T>(), stream.raw())
                .result()
        }
    }

    /// # Arguments
    ///
    /// * `t`: value to be written, no guarantees are made to its content after the function call
    /// * `stream`: The stream to order this order with
    ///
    /// # Safety
    ///
    /// There is no way to know if `t` is valid even if this function succeeds.
    pub unsafe fn copy_to_host(&mut self, t: &mut T, stream: &CuStream) -> CuResult<()> {
        unsafe {
            sys::cuMemcpyDtoHAsync_v2(t as *mut T as _, self.ptr(), size_of::<T>(), stream.raw())
                .result()
        }
    }

    /// # Arguments
    ///
    /// * `stream`: The stream to order this order with
    ///
    /// returns: an instance of `T`
    ///
    /// # Safety
    ///
    /// There is no way to know if the returned value is valid even if this function succeeds.
    pub unsafe fn copy_to_host_new(&mut self, stream: &CuStream) -> CuResult<T> {
        let mut t = MaybeUninit::<T>::uninit();
        unsafe {
            sys::cuMemcpyHtoDAsync_v2(
                self.ptr(),
                t.as_mut_ptr().cast(),
                size_of::<T>(),
                stream.raw(),
            )
            .result()?;
            Ok(t.assume_init())
        }
    }
}

impl<T: ?Sized> CuBox<T> {
    /// Create a CuBox by owning an external pointer.
    pub fn from_ptr(ptr: *mut T) -> Self {
        Self(NonNull::new(ptr).unwrap())
    }

    pub fn ptr(&self) -> sys::CUdeviceptr {
        self.0.as_ptr().cast::<sys::CUdeviceptr>() as _
    }

    pub unsafe fn drop_inner(&self, stream: &CuStream) -> CuResult<()> {
        unsafe { sys::cuMemFreeAsync(self.ptr(), stream.raw()).result() }
    }

    pub fn drop(self, stream: &CuStream) -> CuResult<()> {
        let ret = unsafe { self.drop_inner(stream) };
        mem::forget(self);
        ret
    }
}

impl<T: ?Sized> Drop for CuBox<T> {
    fn drop(&mut self) {
        unsafe { self.drop_inner(CuStream::DEFAULT_).unwrap() }
    }
}

/// No guarantees this is used correctly. Beware of pinning multiple times.
pub struct CuPin<C: Deref>(C);

impl<C: Deref> CuPin<C> {
    pub fn new(data: C) -> CuResult<Self> {
        Self::new_flags(data, 0)
    }

    pub fn new_flags(data: C, flags: u32) -> CuResult<Self> {
        unsafe {
            sys::cuMemHostRegister_v2(
                data.deref() as *const _ as _,
                size_of_val(data.deref()),
                flags,
            )
            .result()?;
        }
        Ok(Self(data))
    }

    fn drop_inner(&mut self) -> CuResult<()> {
        unsafe { sys::cuMemHostUnregister(self.0.deref() as *const _ as _).result() }
    }
}

impl<C: Deref> Drop for CuPin<C> {
    fn drop(&mut self) {
        self.drop_inner().unwrap()
    }
}

impl<T: ?Sized, C: Deref<Target = T>> Deref for CuPin<C> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.0.deref()
    }
}

impl<T: ?Sized, C: DerefMut<Target = T>> DerefMut for CuPin<C> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.0.deref_mut()
    }
}

#[cfg(test)]
mod tests {
    use crate::CuPin;
    use std::rc::Rc;

    #[test]
    fn test() {
        let pinned = CuPin::new(Box::new(45)).unwrap();
        let pinned = CuPin::new(&45).unwrap();
        let pinned = CuPin::new(Rc::new(45)).unwrap();
    }
}
