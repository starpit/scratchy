//! THE WIRE VIEW — the port's [`Dsc`]/[`ScheduleView`] traits over the parsed scheduled wire
//! (`crate::wire`), so the lowering walks a real `sdsc.json` instead of its own fixtures.
//!
//! # The two seams this file exists against
//!
//! ⛔⛔ THE LEAVES ARE BUILT **INSIDE** [`ScheduleView::roots`], against `UnitHandles` the driver
//! minted one step earlier — which is why the statement's leaf payloads are owned and its closure
//! seams are boxed (see [`TransferStatement`]'s note). A view cannot pre-build them against its own
//! `'c` wire data.
//!
//! ⛔⛔ AND THE LATCH MAP IS WALK STATE. A `ComputeInput::Latch` names its latch by id and the
//! [`Val`] only exists once an earlier statement of the same unit latched it — see
//! [`Handlers::latches`], the reference's `global_latch_map`.
