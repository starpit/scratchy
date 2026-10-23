// SPDX-License-Identifier: Apache-2.0
//! ⭐⭐⭐⭐⭐ THE V KERNEL'S ROW EXTENT NAMES ITS PLANE — as a type, not a number. The value leg's
//! shared kernel reads one of exactly two V planes: the resident page plane (and the gathered
//! scratch's byte-for-byte copy of it) or the new-token block's unrestickified wide stream. The
//! row extent used to travel as a bare `u32` fed the HEAD DIM at the new-block call — a head-dim
//! fact standing in a plane-rows slot, coincident only where `hd` is one stick.
//!
//! THE LAW'S TEETH: the gathered value leg refuses every plane but the page (attn.rs's arm door),
//! because its scratch rows ARE page rows — a window's or a stream's extent there once declared a
//! stick-group stride the buffer does not have. These pins hold the doors themselves.

use ktir_superdsc::sdsc_abstract::{PagedKvPool, VKernelRows};

#[test]
fn the_page_planes_rows_are_the_pools_own_page_slots() {
    assert_eq!(
        VKernelRows::Page.rows(),
        PagedKvPool::PAGE_SLOTS as u32,
        "the resident V plane is `[PAGE_SLOTS, hd]` — its rows are the pool's own fact"
    );
}

#[test]
fn the_new_block_stream_carries_the_head_dim_it_declares() {
    for hd in [64u32, 128] {
        assert_eq!(
            VKernelRows::of_new_block_stream(hd).rows(),
            hd,
            "the wide stream's per-head slice is declared `hd` rows, at every head dim"
        );
    }
}

/// ⭐ THE IMPOSTOR SEAT THE ENUM REMOVES: a new-block stream whose row count HAPPENS to equal the
/// page's slots is still not the page plane. As bare `u32`s the two pairings were indistinguishable
/// — 256 == 256 — which is exactly the shape of every swap defect on this path: equal at every
/// value the tests pass, divergent at the first geometry that differs.
#[test]
fn a_stream_of_the_pages_row_count_is_still_not_the_page_plane() {
    assert_ne!(
        VKernelRows::Page,
        VKernelRows::of_new_block_stream(PagedKvPool::PAGE_SLOTS as u32),
        "the plane a kernel reads is a fact about WHICH buffer, not how many rows it happens to have"
    );
    // And each door reads back its own fact from its own slot.
    assert_eq!(
        VKernelRows::of_new_block_stream(PagedKvPool::PAGE_SLOTS as u32).rows(),
        PagedKvPool::PAGE_SLOTS as u32
    );
}
