//! Re-ported from the C++ authority. See crustify-scheduler/AGENT-BRIEF.md.
//!
//! `util/foldManager/mapWithFMHelper.h` — e037_MapWithFMHelper, and why this module declares no
//! type of its own.
//!
//! The class has ONE data member: `std::map<Dkey, FoldManager<Dval>>& key_val_` (`:830-831`), a
//! REFERENCE to a map that some other type owns, bound in the constructor (`:29-30`). Of its 43
//! member functions, 37 are `key_val_.at(key).<a FoldManager method>` or a loop over `key_val_`
//! doing the same to every value; the six that are not — `numKeys` (`:40`), `getAllKeys` (`:51`),
//! `getRef` (`:433`), `removeKey` (`:702`), `isKeyPresent` (`:714`) and `getVal` (`:722`) — are
//! `std::map` operations spelled through the reference. It holds no state and decides nothing.
//!
//! ⛔ SO THE UNIT'S ANCHORS ARE FILLED AT THE OWNER OF THE MAP, NOT HERE:
//! [`TransferPadInfo`](crate::schedule::dsc2::TransferPadInfo) in `schedule/dsc2.rs` carries
//! `Replaces: e037_MapWithFMHelper`, and its two fields carry `Field: e037_MapWithFMHelper.key_val_`
//! — one per binding, because the authority binds the single member twice, once per end
//! (`dsc/dsc2.h:758-759`, `:808-809`). A borrowed Rust facade over a map that is a field of the same
//! object would be a second `&mut` into storage the owner already reaches, answering questions on
//! its behalf — the shape RULE 1 forbids — and it would have no caller: the five methods the
//! authority actually reaches through these two instances are already there.
//!
//! ## The whole in-scope surface, and where each of the five landed
//!
//! `MapWithFMHelper` is instantiated in exactly two places in the authority that this campaign
//! scopes, `transferPadFrontSizeHelper` and `transferPadBackSizeHelper` (`dsc/dsc2.h:808-809`), and
//! five of its 43 methods are ever called on them:
//!
//! | authority | reached from | landed as |
//! |---|---|---|
//! | `addKeyBuildFoldSpace` (`:115-133`) | `buildTransferFoldDim` (`dsc/dsc2.cpp:4677-4681`) | `PadSizeFold::new` |
//! | `insertAlphaForKey` (`:282-285`) | `buildPadSizes` (`:4624`, `:4630`) | `PadSizeFold::new`'s `alphas` |
//! | `insertBetaForKey` (`:307-310`) | `buildPadSizes` (`:4627`, `:4633`) | `PadSizeFold::new`'s `betas` |
//! | `getDataForKey` (`:253-258` → `:206-209`) | both readers (`:4706`, `:4730`) | `PadSizeFold::data` |
//! | `getAllKeys` (`:51-55`) | `getPadFrontOrBackDimsSet` (`dsc/dsc2.h:783-788`) | [`TransferPadInfo::pad_dims`](crate::schedule::dsc2::TransferPadInfo::pad_dims) |
//!
//! Every other use of the class is in `perfdsc/` or `dsm/workOptimizer/` — `dsm/` is
//! `out_of_scope.paths` and `perfdsc/` is in no `impl_files` entry — so the other 38 methods have no
//! in-scope caller to port them for.
//!
//! ## ⛔ AND THE CROSS-KEY METHODS REFUSE THE STATE THE IN-SCOPE BUILDER MAKES
//!
//! This is what makes the dissolution load-bearing rather than cosmetic. The production builder
//! calls `buildPadFrontSizes` once per padded dim with that dim's own `numWkSlice` and `numChunks`
//! (`dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:5388-5451`), so two dims routinely end up with
//! DIFFERENT fold extents in one helper. Measured on the authority, with `X` built `{2, 3}` and `Y`
//! built `{3, 2}` through `buildTransferFoldDim`'s exact sequence:
//!
//! * `getFoldSpaceSize` (`:649-662`) — `DT_ERROR` at `:656-658`, "fold dimenionality cannot differ
//!   across keys in key_val_";
//! * `getCommonFlattenedCoordinates` (`:559-583`) — `DT_ERROR` through
//!   `getNumUniqueCoordsInEachFold` (`:517-552`), because `{2,3}` and `{3,2}` make both
//!   `is_super_set_curr` and `is_super_set_old` true (`:543-546`);
//! * `getNumDims` (`:669-682`) — answers 2 while the dims agree, `DT_ERROR`s as soon as one key has
//!   a different NUMBER of folded dims;
//! * `getDataForKey` — answers each key from that key's OWN extents throughout.
//!
//! A single shared fold space, which is what a ported facade would have had to represent to keep
//! those readers meaningful, cannot hold this state. A [`BTreeMap`](std::collections::BTreeMap) of
//! per-dim folds can, which is what `TransferPadInfo` uses.
//!
//! ⛔ `isLegal` (`:502-515`) CANNOT BE INSTANTIATED AT ALL, so there is no behaviour of it to port.
//! It declares `std::vector<int> foldDimSize` and assigns `FoldManager::getFoldSpaceSize()` to it
//! (`:506`), which returns `std::vector<int64_t>` (`foldInfrastructure.h:2457`) — no viable `=`.
//! Being a member of a class template it is never instantiated, and it has zero callers tree-wide;
//! naming it in a call is a hard compile error, measured. Its body is wrong a second time
//! independently: `if (foldDimSize == kv.second.getFoldSpaceSize()) return false;` (`:508`) calls
//! AGREEING fold spaces illegal, the inverse of the `DT_ERROR` commented out beneath it (`:509-511`).
//!
//! ## ⛔ THE 25 SCHEDULED FIELD ANCHORS NAME ONE FIELD
//!
//! Named here so that removing the other 24 is not silent. `key_val_` is the only declared member
//! (`:830-831`), and it entered the census through the subscript expression `key_val_[new_key];`
//! (`:70`) rather than its declaration, which the `&` in the declared type excludes. The rest are
//! function-LOCALS, one function PARAMETER, a `break;` statement and a `std::endl` insertion:
//!
//! | anchor | `mapWithFMHelper.h` | what it is |
//! |---|---|---|
//! | `allDimProp` | `:412` | local of `cloneEntryForNewKey` |
//! | `all_const` | `:771` | local of `isAllFoldsConstant`, and of `isALLKeyConstFolded` (`:783`) |
//! | `break` | `:739` | the `break;` statement in `hasZeroFoldDim` |
//! | `count` | `:68` | the `.count(` in `key_val_.count(new_key)`, plus `printFoldProp`'s local (`:686`) |
//! | `data_and_coord` | `:220` | local of `getAllDataForKey` |
//! | `endl` | `:616` | `std::endl` inserted by `print` |
//! | `foldDimSize` | `:503` | local of the uninstantiable `isLegal`, and of `getFoldSpaceSize` (`:650`) |
//! | `foldFuncQ` | `:416` | local of `cloneEntryForNewKey` |
//! | `foldPropQ` | `:415` | local of `cloneEntryForNewKey` |
//! | `folddim_coord` | `:597` | local of `print` |
//! | `idx` | `:522` | loop counter, five functions |
//! | `is_super_set_curr` | `:531` | local of `getNumUniqueCoordsInEachFold` |
//! | `is_super_set_old` | `:532` | local of `getNumUniqueCoordsInEachFold` |
//! | `is_zero_fold` | `:735` | local of `hasZeroFoldDim` |
//! | `key_set` | `:52` | local of `getAllKeys` |
//! | `my_map` | `:475` | local of `getMapData` |
//! | `num_dims` | `:670` | local of `getNumDims` |
//! | `repeat_factor` | `:574` | local of `getCommonFlattenedCoordinates` |
//! | `retSet` | `:223` | local of `getAllDataForKey` |
//! | `retVal` | `:190` | local of `rebuildDim` |
//! | `so2` | `:596` | local of `print` |
//! | `total_count` | `:566` | local of `getCommonFlattenedCoordinates` |
//! | `unique_data_coords` | `:518` | the out-PARAMETER of `getNumUniqueCoordsInEachFold` |
//! | `wasCompressed` | `:803` | local of `compressMapToConst`, and of `compressMapToConstForAllDims` (`:821`) |
//!
//! ## The differing fold DIMENSIONALITY of two keys is unspellable here
//!
//! Measured, two keys built with one and two folded dims leave `getDataForKey` answering both while
//! `getNumDims` (`:669-682`) and `getFoldSpaceSize` (`:649-662`) `DT_ERROR` — a state reachable
//! there because `addKeyBuildFoldSpace` takes a run-time-sized `fm_dim_prop` (`:115`). Here the
//! arity is an array length, `FoldDimPosition::COUNT`, so that state is E0308 and the two readers
//! have nothing to refuse. The control is what makes the `compile_fail` case evidence, since stable
//! rustdoc does not check the annotated error code:
//!
//! ```compile_fail,E0308
//! use deeptools::schedule::dims::PrimaryDimTypes;
//! use deeptools::schedule::dsc2::{PadEnd, PadSize, TransferPadInfo};
//! use deeptools::schedule::fold::FoldDimSize;
//!
//! let mut info = TransferPadInfo::default();
//! // A third folded dimension, which `addKeyBuildFoldSpace` accepts there.
//! info.build_pad_sizes(
//!     PadEnd::Front,
//!     PrimaryDimTypes::X,
//!     [FoldDimSize(2), FoldDimSize(3), FoldDimSize(4)],
//!     [PadSize(0), PadSize(0)],
//!     [PadSize(0), PadSize(0)],
//! );
//! ```
//! ```
//! use deeptools::schedule::dims::PrimaryDimTypes;
//! use deeptools::schedule::dsc2::{PadEnd, PadSize, TransferPadInfo};
//! use deeptools::schedule::fold::FoldDimSize;
//!
//! let mut info = TransferPadInfo::default();
//! assert_eq!(
//!     info.build_pad_sizes(
//!         PadEnd::Front,
//!         PrimaryDimTypes::X,
//!         [FoldDimSize(2), FoldDimSize(3)],
//!         [PadSize(0), PadSize(0)],
//!         [PadSize(0), PadSize(0)],
//!     ),
//!     Some(())
//! );
//! ```

#[cfg(test)]
mod equivalence {
    use crate::schedule::dims::PrimaryDimTypes;
    use crate::schedule::dsc2::{
        ChunkIdx, ChunkOffset, ChunkSizePadded, FoldDimPosition, NumChunks, PadEnd, PadSize,
        TransferPadInfo, WkSliceIdx,
    };
    use crate::schedule::fold::FoldDimSize;

    /// `X` gets 2 work slices and 3 chunks, `Y` gets 3 and 2 — which is what the production builder
    /// produces, one `buildPadFrontSizes` per padded dim with that dim's own counts
    /// (`dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:5388-5451`).
    fn two_dims_one_end() -> TransferPadInfo {
        let mut info = TransferPadInfo::default();
        for (dim, sizes, alphas, betas) in [
            (
                PrimaryDimTypes::X,
                [FoldDimSize(2), FoldDimSize(3)],
                [PadSize(-40), PadSize(-10)],
                [PadSize(25), PadSize(0)],
            ),
            (
                PrimaryDimTypes::Y,
                [FoldDimSize(3), FoldDimSize(2)],
                [PadSize(-100), PadSize(-7)],
                [PadSize(50), PadSize(0)],
            ),
        ] {
            assert_eq!(
                info.build_pad_sizes(PadEnd::Front, dim, sizes, alphas, betas),
                Some(())
            );
        }
        info
    }

    /// `A.numKeys = 2`, `A.allKeys = 4 5`,
    /// `A.x.w0.{c0,c1,c2} = 25, 15, 5`, `A.x.w1.{c0,c1,c2} = -15, -25, -35`,
    /// `A.y.w0.{c0,c1} = 50, 43`, `A.y.w1.{c0,c1} = -50, -57`, `A.y.w2.{c0,c1} = -150, -157`,
    /// `A.x.w2.c0_past_x_extent = THROW`, `A.y.w2.c0_legal_for_y = -150`,
    /// `A.x.w0.c2_legal_for_x = 5`, `A.y.w0.c2_past_y_extent = THROW`, `A.unknown_key = THROW`.
    ///
    /// ⛔ THE SAME COORDINATE IS LEGAL FOR ONE DIM AND A THROW FOR THE OTHER, AT ONE END, ON ONE
    /// OBJECT — in BOTH directions. That is the fact a ported `MapWithFMHelper` could not have
    /// represented: `getDataForKey` reaches `key_val_.at(key)` (`:208`) and the range test runs
    /// against THAT manager's `dim_prop_` (`foldInfrastructure.h:1677`), never a shared one.
    ///
    /// The negatives read 0 here rather than `-15`: `getTransferPadSizeFrontOrBack` wraps every
    /// query in `std::max(.., 0)` (`dsc/dsc2.cpp:4729-4731`) and is the only public reader, so the
    /// raw fold value is observable exactly where it is positive.
    #[test]
    fn e037_each_dims_fold_extents_bound_only_that_dim() {
        let info = two_dims_one_end();

        assert_eq!(
            info.pad_dims(PadEnd::Front).collect::<Vec<_>>(),
            [PrimaryDimTypes::X, PrimaryDimTypes::Y],
            "`A.numKeys = 2`, `A.allKeys = 4 5`"
        );

        // `max(getDataForKey(..), 0)` alone: the cap is the identity at `i32::MAX`.
        let raw = |dim, w: i64, c: i64| {
            info.transfer_pad_size(
                PadEnd::Front,
                dim,
                WkSliceIdx(w),
                ChunkIdx(c),
                ChunkSizePadded(i32::MAX),
            )
        };

        for (chunk, x) in [(0, 25), (1, 15), (2, 5)] {
            assert_eq!(raw(PrimaryDimTypes::X, 0, chunk), Some(PadSize(x)));
            // `A.x.w1.*` are all negative, so all three clamp to 0.
            assert_eq!(raw(PrimaryDimTypes::X, 1, chunk), Some(PadSize(0)));
        }
        for (chunk, y) in [(0, 50), (1, 43)] {
            assert_eq!(raw(PrimaryDimTypes::Y, 0, chunk), Some(PadSize(y)));
            assert_eq!(raw(PrimaryDimTypes::Y, 1, chunk), Some(PadSize(0)));
            assert_eq!(raw(PrimaryDimTypes::Y, 2, chunk), Some(PadSize(0)));
        }

        // Work slice 2 is past X's extent of 2 and inside Y's of 3.
        assert_eq!(
            raw(PrimaryDimTypes::X, 2, 0),
            None,
            "`A.x.w2.c0_past_x_extent`"
        );
        assert_eq!(
            raw(PrimaryDimTypes::Y, 2, 0),
            Some(PadSize(0)),
            "`A.y.w2.c0_legal_for_y = -150`"
        );
        // Chunk 2 is inside X's extent of 3 and past Y's of 2 — the crossover the other way.
        assert_eq!(
            raw(PrimaryDimTypes::X, 0, 2),
            Some(PadSize(5)),
            "`A.x.w0.c2_legal_for_x`"
        );
        assert_eq!(
            raw(PrimaryDimTypes::Y, 0, 2),
            None,
            "`A.y.w0.c2_past_y_extent`"
        );
        assert_eq!(raw(PrimaryDimTypes::Kij, 0, 0), None, "`A.unknown_key`");
    }

    /// `walk.x.w0.nc3 = 25`, `walk.x.w1.nc3 = 0`, `walk.y.w0.nc2 = 20`, `walk.y.w1.nc2 = 0`,
    /// `walk.y.w2.nc2 = 0`, `walk.y.w0.nc3_xs_count = THROW`, `walk.x.w0.nc2_ys_count = 20`.
    ///
    /// ⛔ `numChunks` IS THE CALLER'S, NOT THE FOLD'S, AND THE TWO DIMS DISAGREE ABOUT IT: the walk
    /// takes it per dim from `dsc/dsc2.cpp:4827` and `getDataForKey` range-tests it against the
    /// queried key's own `chunk_index` extent. Feeding `Y` the chunk count `X` was built with is
    /// therefore a throw on the third iteration, while `X` under `Y`'s smaller count simply runs out
    /// of loop — so the count cannot be recovered from any shared fold space either.
    #[test]
    fn e037_the_wk_slice_walks_chunk_count_is_range_tested_per_dim() {
        let info = two_dims_one_end();
        let walk = |dim, w: i64, num_chunks: u32| {
            info.wk_slice_pad_size(
                PadEnd::Front,
                dim,
                WkSliceIdx(w),
                NumChunks(num_chunks),
                ChunkOffset(10),
                ChunkSizePadded(10),
            )
        };

        // X: chunks 0 and 1 are fully padded (25, 15 >= 10), chunk 2 is partial at 5 — `10 * 2 + 5`.
        assert_eq!(walk(PrimaryDimTypes::X, 0, 3), Some(PadSize(25)));
        assert_eq!(walk(PrimaryDimTypes::X, 1, 3), Some(PadSize(0)));
        // Y: both chunks are fully padded (50, 43 >= 10), so the loop ends with no partial.
        assert_eq!(walk(PrimaryDimTypes::Y, 0, 2), Some(PadSize(20)));
        assert_eq!(walk(PrimaryDimTypes::Y, 1, 2), Some(PadSize(0)));
        assert_eq!(walk(PrimaryDimTypes::Y, 2, 2), Some(PadSize(0)));

        assert_eq!(
            walk(PrimaryDimTypes::Y, 0, 3),
            None,
            "`walk.y.w0.nc3_xs_count`"
        );
        assert_eq!(
            walk(PrimaryDimTypes::X, 0, 2),
            Some(PadSize(20)),
            "`walk.x.w0.nc2_ys_count`"
        );
    }

    /// `B.allKeys_out_of_order_insert = 0 4 9`, `B.getRef_order = 0:0 4:4 9:9`.
    ///
    /// ⛔ `getAllKeys` BUILDS A `std::set<Dkey>` (`:51-55`) AND `getRef` HANDS BACK THE `std::map`
    /// (`:433`), so both are ascending KEY order and neither is insertion order. The one caller
    /// `std::set_union`s the two ends (`dsc/dsc2.cpp:4814-4820`) and depends on it. Each dim's beta
    /// is its own discriminant here, so the assertion also proves the key-to-fold association
    /// survived the reordering rather than only the key sequence.
    #[test]
    fn e037_get_all_keys_is_ascending_key_order_not_insertion_order() {
        let mut info = TransferPadInfo::default();
        // Inserted in neither key order nor reverse key order: 9, 0, 4.
        for dim in [PrimaryDimTypes::Ki, PrimaryDimTypes::In, PrimaryDimTypes::X] {
            assert_eq!(
                info.build_pad_sizes(
                    PadEnd::Front,
                    dim,
                    [FoldDimSize(1); FoldDimPosition::COUNT],
                    [PadSize(0); FoldDimPosition::COUNT],
                    [PadSize(dim as i32), PadSize(0)],
                ),
                Some(())
            );
        }

        assert_eq!(
            info.pad_dims(PadEnd::Front).collect::<Vec<_>>(),
            [PrimaryDimTypes::In, PrimaryDimTypes::X, PrimaryDimTypes::Ki],
            "`B.allKeys_out_of_order_insert = 0 4 9`"
        );
        for dim in info.pad_dims(PadEnd::Front).collect::<Vec<_>>() {
            assert_eq!(
                info.transfer_pad_size(
                    PadEnd::Front,
                    dim,
                    WkSliceIdx(0),
                    ChunkIdx(0),
                    ChunkSizePadded(i32::MAX)
                ),
                Some(PadSize(dim as i32)),
                "`B.getRef_order = 0:0 4:4 9:9`"
            );
        }
    }
}

#[cfg(test)]
mod unit_tests {
    use crate::schedule::dims::PrimaryDimTypes;
    use crate::schedule::dsc2::{
        ChunkIdx, ChunkSizePadded, PadEnd, PadSize, TransferPadInfo, WkSliceIdx,
    };
    use crate::schedule::fold::FoldDimSize;

    /// A DELIBERATE NON-EQUIVALENCE, and the authority's side is a `DT_ERROR`.
    ///
    /// Measured on the authority over this exact object — `X` built `{2, 3}` and `Y` built `{3, 2}`
    /// through `buildTransferFoldDim`'s sequence (`dsc/dsc2.cpp:4655-4682`):
    /// `C.getFoldSpaceSize_differing = THROW` (`:649-662`) and
    /// `F.getCommonFlattenedCoordinates = THROW` (`:559-583`, via `getNumUniqueCoordsInEachFold`'s
    /// `is_super_set_curr && is_super_set_old` at `:543-546`), while `getDataForKey` answers both
    /// keys throughout. So the facade's own cross-key readers refuse the state the facade's in-scope
    /// BUILDER makes and its in-scope READERS query, and the refusal is unreachable only because no
    /// in-scope caller names them.
    ///
    /// The divergence: there is no cross-key reader here to refuse it, because per-dim folds are
    /// separate values in a [`BTreeMap`](std::collections::BTreeMap). This pins that the state stays
    /// constructible and queryable, which is the property the `equivalence` cases above rest on.
    #[test]
    fn e037_a_per_dim_fold_space_has_no_cross_key_reader_to_refuse_it() {
        let mut info = TransferPadInfo::default();
        assert_eq!(
            info.build_pad_sizes(
                PadEnd::Front,
                PrimaryDimTypes::X,
                [FoldDimSize(2), FoldDimSize(3)],
                [PadSize(-40), PadSize(-10)],
                [PadSize(25), PadSize(0)],
            ),
            Some(())
        );
        assert_eq!(
            info.build_pad_sizes(
                PadEnd::Front,
                PrimaryDimTypes::Y,
                [FoldDimSize(3), FoldDimSize(2)],
                [PadSize(-100), PadSize(-7)],
                [PadSize(50), PadSize(0)],
            ),
            Some(()),
            "C refuses to report a fold space for this object at all; building it is not refused"
        );

        // Both dims answer at their own maximal legal coordinate, which no single fold space
        // spanning the two could describe: `X` has 2 x 3 and `Y` has 3 x 2.
        let at = |dim, w: i64, c: i64| {
            info.transfer_pad_size(
                PadEnd::Front,
                dim,
                WkSliceIdx(w),
                ChunkIdx(c),
                ChunkSizePadded(i32::MAX),
            )
        };
        assert!(at(PrimaryDimTypes::X, 1, 2).is_some());
        assert!(at(PrimaryDimTypes::Y, 2, 1).is_some());
        assert!(at(PrimaryDimTypes::X, 2, 1).is_none());
        assert!(at(PrimaryDimTypes::Y, 1, 2).is_none());
    }
}
