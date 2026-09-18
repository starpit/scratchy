//! Re-ported from the C++ authority. See crustify-scheduler/AGENT-BRIEF.md.

use sys_arch_spec::arch_enums::SenComponent;
use sys_arch_spec::{CoreId, CoreletId, RowId};

/// Replaces: e001_FailedAlloc
///
/// The memory-tracker key an allocation would not fit in — `ddc/ddc_metadata.h:24-29`.
///
/// ⭐ THE FOUR FIELDS ARE `getTracker`'S ARGUMENT LIST, in its order
/// (`sys-arch-spec/memtracker/mem_track_bundle.h:34`): `Ddc::allocAllMem` fills one the instant
/// `checkAndAddDs` answers `DOESNT_FIT`, naming the tracker that refused (`ddc/ddcv1.cpp:344-369`).
/// ⛔ NO METHODS AND NO INITIALIZERS TO PORT. `ddc/ddcv1.cpp:363` default-constructs the struct and
/// assigns all four fields on the next four lines; the header declares no member function and no
/// default, and no other file in the authority names the type. Its only reader is
/// `failedAllocs.size() == 0` (`ddc/ddcv1.cpp:378`, `:436`).
/// ⭐ `Copy` because the fill site pushes it by value (`ddc/ddcv1.cpp:368`).
///
/// Transposing the core and the corelet is `E0308`, not a silently wrong tracker:
/// ```compile_fail
/// use deeptools::schedule::metadata::FailedAlloc;
/// use sys_arch_spec::arch_enums::SenComponent;
/// use sys_arch_spec::{CoreId, CoreletId, RowId};
/// let _ = FailedAlloc {
///     comp: SenComponent::Lx,
///     core: CoreletId(0),
///     corelet: CoreId(3),
///     row: RowId(0),
/// };
/// ```
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct FailedAlloc {
    /// The memory that refused — a key of `metadata.newAllocations_` (`ddc/ddcv1.cpp:184`, `:364`).
    pub comp: SenComponent,
    /// The core it refused on: an element of `coreIdsUsed_` (`dsc/designSpaceConfig.h:75`) for LX, L0
    /// and L0_SCALE, else that vector's `front()` standing proxy for all of them
    /// (`ddc/ddcv1.cpp:189-198`, `:365`).
    pub core: CoreId,
    /// The corelet it refused on: `0` alone, and the rest up to `numCoreletsUsed_DSC2_` only when the
    /// arch is newer than RCUDD1A and the component is L0 or L0_SCALE (`ddc/ddcv1.cpp:200-210`,
    /// `:366`).
    pub corelet: CoreletId,
    /// The register-file row it refused on. ⛔ ALWAYS `0` FROM THIS FILL SITE — `allocAllMem` fixes
    /// `rows` at the single proxy row (`ddc/ddcv1.cpp:211`, `:367`), so the field carries the tracker
    /// key's third component without this caller ever varying it.
    pub row: RowId,
}

#[cfg(test)]
mod unit_tests {
    use super::*;

    /// `ddc/ddcv1.cpp:362-369` — the LX tracker of a used core refuses, with the corelet and row at
    /// the proxy `0` that `corelets(1, 0)` and `rows(1, 0)` fix for it (`:205`, `:211`). Every field
    /// is part of the key, so moving the corelet alone names a different tracker.
    #[test]
    fn carries_the_tracker_key_that_refused() {
        let failed = FailedAlloc {
            comp: SenComponent::Lx,
            core: CoreId(3),
            corelet: CoreletId(0),
            row: RowId(0),
        };
        assert_eq!(failed.comp, SenComponent::Lx);
        assert_eq!(failed.core, CoreId(3));
        assert_eq!(failed.corelet, CoreletId(0));
        assert_eq!(failed.row, RowId(0));
        assert_ne!(
            failed,
            FailedAlloc {
                corelet: CoreletId(1),
                ..failed
            }
        );
    }
}
