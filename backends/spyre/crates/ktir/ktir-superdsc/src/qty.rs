// SPDX-License-Identifier: Apache-2.0
//! The quantities a lowering counts in. Each is its own type, entered through one constructor and
//! left through `get`, so an element count is never added to a byte count or used as an extent.

use std::num::NonZeroU64;

/// A count of elements, or an element offset into a buffer.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Elems(u64);

impl Elems {
    pub const ZERO: Elems = Elems(0);

    pub const fn new(n: u64) -> Elems {
        Elems(n)
    }

    pub const fn get(self) -> u64 {
        self.0
    }

    /// The bytes these elements take at `bytes_per_elem` bytes each.
    pub const fn bytes(self, bytes_per_elem: NonZeroU64) -> Bytes {
        Bytes(self.0 * bytes_per_elem.get())
    }
}

impl std::ops::Add for Elems {
    type Output = Elems;
    fn add(self, o: Elems) -> Elems {
        Elems(self.0 + o.0)
    }
}

/// How many positions an axis has: never zero.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Extent(NonZeroU64);

impl Extent {
    pub const ONE: Extent = Extent(NonZeroU64::MIN);

    pub const fn new(n: NonZeroU64) -> Extent {
        Extent(n)
    }

    pub const fn get(self) -> u64 {
        self.0.get()
    }
}

/// The element distance between consecutive positions of an axis. Zero repeats one element along
/// the axis: a broadcast.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Stride(u64);

impl Stride {
    pub const BROADCAST: Stride = Stride(0);
    pub const UNIT: Stride = Stride(1);

    pub const fn new(n: u64) -> Stride {
        Stride(n)
    }

    pub const fn get(self) -> u64 {
        self.0
    }
}

/// A count of bytes, or a byte offset.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Bytes(u64);

impl Bytes {
    pub const fn new(n: u64) -> Bytes {
        Bytes(n)
    }

    pub const fn get(self) -> u64 {
        self.0
    }
}
