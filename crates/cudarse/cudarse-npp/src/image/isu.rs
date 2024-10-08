//! Support functions (malloc & free)

use std::marker::PhantomData;
use std::ptr::null_mut;

use crate::sys::*;

use crate::image::{Channels, Image, Sample, C, P};
use crate::{get_stream, Result};

pub trait Malloc {
    /// `malloc` a new image on device
    fn malloc(width: u32, height: u32) -> Result<Self>
    where
        Self: Sized;
}

macro_rules! malloc_impl {
    ($sample_ty:ty, $channel_ty:ty, $sample_id:ident, $channel_id:ident) => {
        impl Malloc for Image<$sample_ty, $channel_ty> {
            fn malloc(width: u32, height: u32) -> Result<Self>
            where
                Self: Sized,
            {
                let mut pitch = 0;
                let ptr = unsafe { paste::paste!([<nppi Malloc $sample_id _ $channel_id>])(width as i32, height as i32, &mut pitch) };
                if ptr.is_null() {
                    Err(Error::from(NppStatus::NPP_MEMORY_ALLOCATION_ERR))
                } else {
                    // dbg!(pitch);
                    Ok(Self {
                        width,
                        height,
                        pitch,
                        data: ptr as _,
                        marker: PhantomData,
                        marker_: PhantomData,
                    })
                }
            }
        }
    };
}

malloc_impl!(u8, C<1>, _8u, C1);
malloc_impl!(u8, C<2>, _8u, C2);
malloc_impl!(u8, C<3>, _8u, C3);
malloc_impl!(u8, C<4>, _8u, C4);

malloc_impl!(u16, C<1>, _16u, C1);
malloc_impl!(u16, C<2>, _16u, C2);
malloc_impl!(u16, C<3>, _16u, C3);
malloc_impl!(u16, C<4>, _16u, C4);

malloc_impl!(i16, C<1>, _16s, C1);
malloc_impl!(i16, C<2>, _16s, C2);
malloc_impl!(i16, C<4>, _16s, C4);

malloc_impl!(f32, C<1>, _32f, C1);
malloc_impl!(f32, C<2>, _32f, C2);
malloc_impl!(f32, C<3>, _32f, C3);
malloc_impl!(f32, C<4>, _32f, C4);

macro_rules! malloc_planar_impl {
    ($sample_ty:ty, $n:literal) => {
        impl Malloc for Image<$sample_ty, P<$n>> {
            fn malloc(width: u32, height: u32) -> Result<Self>
            where
                Self: Sized,
            {
                let mut planes = [null_mut(); $n];
                let mut pitch = 0;
                for i in 0..$n {
                    let img = Image::<$sample_ty, C<1>>::malloc(width, height)?;
                    pitch = img.pitch;
                    planes[i] = img.into_inner();
                }

                Ok(Self {
                    width,
                    height,
                    pitch,
                    data: planes,
                    marker: PhantomData,
                    marker_: PhantomData,
                })
            }
        }
    };
}

malloc_planar_impl!(f32, 3);
malloc_planar_impl!(f32, 4);

impl<S: Sample, C: Channels> Drop for Image<S, C> {
    fn drop(&mut self) {
        self.inner_drop(get_stream()).unwrap();
    }
}

#[cfg(test)]
mod tests {
    use crate::image::isu::Malloc;
    use crate::image::C;
    use crate::image::{Image, Img};
    use crate::Result;

    #[test]
    fn new_image() -> Result<()> {
        let img = Image::<f32, C<3>>::malloc(1024, 1024)?;
        dbg!(img.pitch());
        Ok(())
    }

    #[test]
    fn same_size() -> Result<()> {
        let img = Image::<f32, C<3>>::malloc(1024, 1024)?;
        let other = img.malloc_same_size::<u8, C<3>>()?;
        assert_eq!(img.size(), other.size());
        Ok(())
    }
}
