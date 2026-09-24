//! Round-trip pin for [`sys_arch_spec::arch_enums::SenComponent`]'s spelling table.
//!
//! The compile-time assert in `arch_enums.rs` pins injectivity; this pins that every spelling
//! parses back to the variant that printed it — the property `stringToSenComponents`
//! (`sys-arch-spec/arch_enums.cpp:121-122`) has for free by being a flipped map, and a hand-written
//! `match` only has by test.

use sys_arch_spec::arch_enums::SenComponent;

#[test]
fn every_spelling_round_trips() {
    for c in SenComponent::ALL {
        let sp = c.spelling();
        assert_eq!(SenComponent::from_spelling(sp), Some(c), "{sp} does not parse back");
    }
}

#[test]
fn an_unknown_spelling_is_none() {
    assert_eq!(SenComponent::from_spelling("not_a_unit"), None);
}

#[test]
fn the_table_is_the_cpp_tables_size() {
    // 107 entries in `senComponentsToString` (`sys-arch-spec/arch_enums.cpp:11-119`).
    assert_eq!(SenComponent::ALL.len(), 107);
}
