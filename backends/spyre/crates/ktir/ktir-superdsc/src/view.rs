// SPDX-License-Identifier: Apache-2.0
//! A tensor as KTIR states it: a strided view over a buffer's elements, its origin an affine
//! integer ([`Lin`]).
//!
//! `ktdp.construct_memory_view` states a buffer's shape and strides, an access tile takes a window
//! of it, and every shape op (`expand_shape`, `collapse_shape`, `reshape`, `transpose`,
//! `broadcast`) re-describes the same elements. A [`SymView`] is the composition of them all:
//! logical index `i` is element `origin + Σ i[a]·stride[a]`. A window whose corner moves with the
//! trip or a grid index moves the origin by the same formula, so the view says how it moves.

use std::num::NonZeroU64;

use crate::lin::{Lin, NotAffine};
use crate::program::Axis;
use crate::qty::{Extent, Stride};

/// A view whose origin may depend on the trip and the grid.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SymView {
    pub origin: Lin,
    pub axes: Vec<Axis>,
}

/// A shape op the view cannot express without moving elements.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ViewError {
    /// A reshape whose merged axes are not one strided run in the buffer.
    NotContiguous { from: Vec<Extent>, to: Vec<Extent> },
    /// A reshape between shapes of different element counts.
    Numel { from: u64, to: u64 },
    /// A permutation that is not one of the view's rank.
    Permutation { rank: usize, perm: Vec<usize> },
    /// A window that reads past an axis.
    Window { axis: usize, corner: i64, extent: Extent, of: Extent },
    /// A window corner that is not affine in the trip and the grid.
    Corner(NotAffine),
    /// An op naming an axis the view does not have.
    Rank { rank: usize, axis: usize },
}

fn numel(e: &[Extent]) -> u64 {
    e.iter().map(|x| x.get()).product()
}

impl SymView {
    /// `ktdp.construct_memory_view`: the buffer's own shape and strides, at `origin`.
    pub fn of_memory(origin: Lin, shape: &[Extent], strides: &[Stride]) -> SymView {
        SymView {
            origin,
            axes: shape
                .iter()
                .zip(strides)
                .map(|(&extent, &stride)| Axis { extent, stride })
                .collect(),
        }
    }

    /// A row-major view of `shape` at `origin`.
    pub fn row_major(origin: Lin, shape: &[Extent]) -> SymView {
        let mut stride = 1u64;
        let mut axes: Vec<Axis> = shape
            .iter()
            .rev()
            .map(|&extent| {
                let a = Axis {
                    extent,
                    stride: Stride::new(stride),
                };
                stride *= extent.get();
                a
            })
            .collect();
        axes.reverse();
        SymView { origin, axes }
    }

    pub fn shape(&self) -> Vec<Extent> {
        self.axes.iter().map(|a| a.extent).collect()
    }

    /// An access tile: the window of `extents` at `corner`. A constant corner is checked against
    /// the axis here; a moving corner is checked by the walk or the grid that moves it, which
    /// knows how far it moves.
    pub fn window(&self, corner: &[Lin], extents: &[Extent]) -> Result<SymView, ViewError> {
        let mut v = self.clone();
        for (a, (c, &e)) in corner.iter().zip(extents).enumerate() {
            let axis = v.axes[a];
            if let Some(k) = c.as_const()
                && (k < 0 || k as u64 + e.get() > axis.extent.get())
            {
                return Err(ViewError::Window {
                    axis: a,
                    corner: k,
                    extent: e,
                    of: axis.extent,
                });
            }
            v.origin = v
                .origin
                .add(c.times(Lin::constant(axis.stride.get() as i64)).map_err(ViewError::Corner)?);
            v.axes[a].extent = e;
        }
        Ok(v)
    }

    /// `linalg.transpose` / `tl.trans`: result axis `i` is view axis `perm[i]`.
    pub fn transpose(&self, perm: &[usize]) -> Result<SymView, ViewError> {
        let rank = self.axes.len();
        let mut seen = vec![false; rank];
        for &p in perm {
            if p >= rank || std::mem::replace(&mut seen[p], true) {
                return Err(ViewError::Permutation {
                    rank,
                    perm: perm.to_vec(),
                });
            }
        }
        if perm.len() != rank {
            return Err(ViewError::Permutation {
                rank,
                perm: perm.to_vec(),
            });
        }
        Ok(SymView {
            origin: self.origin,
            axes: perm.iter().map(|&p| self.axes[p]).collect(),
        })
    }

    /// A broadcast to `to`: each axis of extent 1 spreads to `to`'s extent at stride 0, and
    /// leading axes `to` adds are broadcast too.
    pub fn broadcast(&self, to: &[Extent]) -> Result<SymView, ViewError> {
        let lead = to.len().saturating_sub(self.axes.len());
        let mut axes: Vec<Axis> = to[..lead]
            .iter()
            .map(|&extent| Axis {
                extent,
                stride: Stride::BROADCAST,
            })
            .collect();
        for (a, (&axis, &e)) in self.axes.iter().zip(&to[lead..]).enumerate() {
            axes.push(if axis.extent == e {
                axis
            } else if axis.extent == Extent::ONE {
                Axis {
                    extent: e,
                    stride: Stride::BROADCAST,
                }
            } else {
                return Err(ViewError::Rank {
                    rank: self.axes.len(),
                    axis: a,
                });
            });
        }
        Ok(SymView {
            origin: self.origin,
            axes,
        })
    }

    /// A reshape that moves no element: each run of old axes merged into new axes must be one
    /// strided run in the buffer. Axes of extent 1 carry stride 0.
    pub fn reshape(&self, to: &[Extent]) -> Result<SymView, ViewError> {
        let from = self.shape();
        if numel(&from) != numel(to) {
            return Err(ViewError::Numel {
                from: numel(&from),
                to: numel(to),
            });
        }
        let not_contiguous = || ViewError::NotContiguous {
            from: from.clone(),
            to: to.to_vec(),
        };
        let old: Vec<Axis> = self
            .axes
            .iter()
            .copied()
            .filter(|a| a.extent != Extent::ONE)
            .collect();
        let new: Vec<usize> = (0..to.len()).filter(|&i| to[i] != Extent::ONE).collect();
        let mut strides = vec![Stride::BROADCAST; to.len()];
        let (mut oi, mut ni) = (0usize, 0usize);
        while ni < new.len() && oi < old.len() {
            let (mut oj, mut nj) = (oi + 1, ni + 1);
            let mut op = old[oi].extent.get();
            let mut np = to[new[ni]].get();
            while op != np {
                if np < op {
                    np *= to[new[nj]].get();
                    nj += 1;
                } else {
                    op *= old[oj].extent.get();
                    oj += 1;
                }
            }
            for k in oi..oj - 1 {
                if old[k].stride.get() != old[k + 1].extent.get() * old[k + 1].stride.get() {
                    return Err(not_contiguous());
                }
            }
            let mut s = old[oj - 1].stride.get();
            for k in (ni..nj).rev() {
                strides[new[k]] = Stride::new(s);
                s *= to[new[k]].get();
            }
            oi = oj;
            ni = nj;
        }
        Ok(SymView {
            origin: self.origin,
            axes: to
                .iter()
                .zip(strides)
                .map(|(&extent, stride)| Axis { extent, stride })
                .collect(),
        })
    }

    /// How many elements the view's last position is past its origin, plus one: the span a
    /// buffer must hold from the origin.
    pub fn span(&self) -> u64 {
        1 + self
            .axes
            .iter()
            .map(|a| (a.extent.get() - 1) * a.stride.get())
            .sum::<u64>()
    }
}

/// An extent from a KTIR dimension: a dimension of zero positions is no tensor at all.
pub fn extent(n: i64) -> Option<Extent> {
    u64::try_from(n).ok().and_then(NonZeroU64::new).map(Extent::new)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lin::Sym;

    fn ex(v: &[u64]) -> Vec<Extent> {
        v.iter().map(|&n| extent(n as i64).unwrap()).collect()
    }

    #[test]
    fn a_window_at_the_trip_moves_the_origin_by_its_stride() {
        // `mask_stack` `[NUM_BLOCKS * P, BLOCK]`, the tile at `[i * P, 0]`, P = 4, BLOCK = 64.
        let mem = SymView::row_major(Lin::constant(0), &ex(&[32, 64]));
        let corner = [Lin::sym(Sym::Trip).scale(4), Lin::constant(0)];
        let tile = mem.window(&corner, &ex(&[4, 64])).unwrap();
        assert_eq!(tile.origin.trip, 4 * 64);
        assert_eq!(tile.shape(), ex(&[4, 64]));
    }

    #[test]
    fn a_reshape_splits_and_merges_contiguous_runs_only() {
        let v = SymView::row_major(Lin::constant(0), &ex(&[6, 8]));
        let split = v.reshape(&ex(&[2, 3, 8])).unwrap();
        assert_eq!(split.axes.iter().map(|a| a.stride.get()).collect::<Vec<_>>(), vec![24, 8, 1]);
        let t = v.transpose(&[1, 0]).unwrap();
        assert!(matches!(t.reshape(&ex(&[48])), Err(ViewError::NotContiguous { .. })));
    }

    #[test]
    fn a_broadcast_spreads_unit_axes_at_stride_zero() {
        let v = SymView::row_major(Lin::constant(0), &ex(&[1, 8]));
        let b = v.broadcast(&ex(&[3, 4, 8])).unwrap();
        assert_eq!(b.axes.iter().map(|a| a.stride.get()).collect::<Vec<_>>(), vec![0, 0, 1]);
    }
}
