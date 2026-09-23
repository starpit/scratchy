//===-- SentientTypes.td -- SentientTypes Dialect Operation Definitions -*- tablegen ----==//
//
// IBM Confidential
// (C) Copyright IBM Corp. 2021, 2025
//
// =============================================================================
//
// This file defines the sentient types in the MLIR operation set.
//
//===----------------------------------------------------------------------===//

#ifndef DCC_SRC_DIALECT_SENTIENT_TYPES_TD
#define DCC_SRC_DIALECT_SENTIENT_TYPES_TD

#ifndef OP_BASE
include "mlir/IR/OpBase.td"
#endif // OP_BASE
include "mlir/IR/EnumAttr.td"

include "mlir/Dialect/LLVMIR/LLVMOpBase.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"


def Sentient_Dialect : Dialect {
  let name = "sentient";
  let cppNamespace = "::mlir::sentient";
  let useDefaultAttributePrinterParser = 1;
}

// Base class for Sentient dialect operations. This operation inherits from the
// base `Op` class in OpBase.td, and provides:
//   * The parent dialect of the operation.
//   * The mnemonic for the operation, or the name without the dialect prefix.
//   * A list of traits for the operation.
class Sentient_Op<string mnemonic, list<Trait> traits = []> :
    Op<Sentient_Dialect, mnemonic, traits> {
  // For every standard op, there needs to be a:
  //   * void print(OpAsmPrinter &p, ${C++ class of Op} op)
  //   * LogicalResult verify(${C++ class of Op} op)
  //   * ParseResult parse${C++ class of Op}(OpAsmParser &parser,
  //                                         OperationState &result)
  // functions.
}

// Based on https://discourse.llvm.org/t/psa-stop-using-strenumattr-do-use-enumattr/5710
//Precision attributes
def SentientINT1 : I32EnumAttrCase<"int1", 0>;
def SentientINT2 : I32EnumAttrCase<"int2", 1>;
def SentientINT4 : I32EnumAttrCase<"int4", 2>;
def SentientINT8 : I32EnumAttrCase<"int8", 3>;
def SentientINT16 : I32EnumAttrCase<"int16", 4>;
def SentientINT24 : I32EnumAttrCase<"int24", 5>;
def SentientINT32 : I32EnumAttrCase<"int32", 6>;
def SentientMXFP4 : I32EnumAttrCase<"mxfp4", 8>;
def SentientMXFP8 : I32EnumAttrCase<"mxfp8", 9>;
def SentientMXINT4 : I32EnumAttrCase<"mxint4", 10>;
def SentientFP4 : I32EnumAttrCase<"fp4", 11>;
def SentientFP8 : I32EnumAttrCase<"fp8", 12>;
def SentientFP16 : I32EnumAttrCase<"fp16", 13>;
def SentientBF16 : I32EnumAttrCase<"bf16", 14>;
def SentientIEEEFP16 : I32EnumAttrCase<"ieee_fp16", 15>;
def SentientFP24 : I32EnumAttrCase<"fp24", 16>;
def SentientFP32 : I32EnumAttrCase<"fp32", 17>;
def SentientIEEEINT64 : I32EnumAttrCase<"int64", 18>;
def SentientNOPRECISION : I32EnumAttrCase<"none", 19>;

def SentientPrecision : I32EnumAttr<"SentientPrecision",
    "built-in precisions for sentient.",
    [
      SentientINT1,
      SentientINT2,
      SentientINT4,
      SentientINT8,
      SentientINT16,
      SentientINT24,
      SentientINT32,
      SentientMXFP4,
      SentientMXFP8,
      SentientMXINT4,
      SentientFP4,
      SentientFP8,
      SentientFP16,
      SentientBF16,
      SentientIEEEFP16,
      SentientFP24,
      SentientFP32,
      SentientIEEEINT64,
      SentientNOPRECISION
    ]> {
        let genSpecializedAttr = 0;
    }

def SentientPrecisionAttr : EnumAttr<Sentient_Dialect, SentientPrecision, "precision">;

//Compute port attributes
def SentientNone : I32EnumAttrCase<"none", 0>;
def SentientZero : I32EnumAttrCase<"zero", 1>;
def SentientOne : I32EnumAttrCase<"one", 2>;
def SentientTwo : I32EnumAttrCase<"two", 3>;
def SentientThree : I32EnumAttrCase<"three", 4>;
def SentientWest : I32EnumAttrCase<"west", 5>;
def SentientNorth : I32EnumAttrCase<"north", 6>;
def SentientEast : I32EnumAttrCase<"east", 7>;
def SentientSouth : I32EnumAttrCase<"south", 8>;
def SentientXRF : I32EnumAttrCase<"xrf", 9>;
def SentientIRF0 : I32EnumAttrCase<"irf0", 10>;
def SentientIRF1 : I32EnumAttrCase<"irf1", 11>;
def SentientLRF0 : I32EnumAttrCase<"lrf0", 12>;
def SentientLRF1 : I32EnumAttrCase<"lrf1", 13>;
def SentientLRF2 : I32EnumAttrCase<"lrf2", 14>;
def SentientLRF3 : I32EnumAttrCase<"lrf3", 15>;
def SentientLRF4 : I32EnumAttrCase<"lrf4", 16>;
def SentientLRF5 : I32EnumAttrCase<"lrf5", 17>;
def SentientLRF6 : I32EnumAttrCase<"lrf6", 18>;
def SentientLRF7 : I32EnumAttrCase<"lrf7", 19>;
def SentientLRF8 : I32EnumAttrCase<"lrf8", 20>;
def SentientLRF9 : I32EnumAttrCase<"lrf9", 21>;
def SentientLRF10 : I32EnumAttrCase<"lrf10", 22>;
def SentientLRF11 : I32EnumAttrCase<"lrf11", 23>;
def SentientLRF12 : I32EnumAttrCase<"lrf12", 24>;
def SentientLRF13 : I32EnumAttrCase<"lrf13", 25>;
def SentientLRF14 : I32EnumAttrCase<"lrf14", 26>;
def SentientLRF15 : I32EnumAttrCase<"lrf15", 27>;
def SentientLatch : I32EnumAttrCase<"latch", 28>;
def SentientOpA : I32EnumAttrCase<"opA", 29>;
def SentientOpB : I32EnumAttrCase<"opB", 30>;
def SentientOpC : I32EnumAttrCase<"opC", 31>;
def SentientResult : I32EnumAttrCase<"result", 32>;
def SentientPT : I32EnumAttrCase<"pt", 33>;
def SentientPE : I32EnumAttrCase<"pe", 35>;
def SentientSFP : I32EnumAttrCase<"sfp", 36>;
def SentientSFPRing : I32EnumAttrCase<"sfpring", 37>;
def SentientLX : I32EnumAttrCase<"lx", 38>;
def SentientL0 : I32EnumAttrCase<"l0", 39>;
def SentientNfwd0 : I32EnumAttrCase<"nfwd0", 40>;
def SentientNfwd2 : I32EnumAttrCase<"nfwd2", 41>;
def SentientNbrSlice : I32EnumAttrCase<"nbrslice", 42>;
def SentientISTATE0 : I32EnumAttrCase<"istate0", 43>;
def SentientISTATE1 : I32EnumAttrCase<"istate1", 44>;
def SentientISTATE2 : I32EnumAttrCase<"istate2", 45>;
def SentientISTATE3 : I32EnumAttrCase<"istate3", 46>;
def SentientCrossPTNorthLink : I32EnumAttrCase<"crossptnlink", 47>;
def SentientLRF16 : I32EnumAttrCase<"lrf16", 48>;
def SentientLRF17 : I32EnumAttrCase<"lrf17", 49>;
def SentientLRF18 : I32EnumAttrCase<"lrf18", 50>;
def SentientLRF19 : I32EnumAttrCase<"lrf19", 51>;
def SentientLRF20 : I32EnumAttrCase<"lrf20", 52>;
def SentientLRF21 : I32EnumAttrCase<"lrf21", 53>;
def SentientLRF22 : I32EnumAttrCase<"lrf22", 54>;
def SentientLRF23 : I32EnumAttrCase<"lrf23", 55>;
def SentientLRF24 : I32EnumAttrCase<"lrf24", 56>;
def SentientLRF25 : I32EnumAttrCase<"lrf25", 57>;
def SentientLRF26 : I32EnumAttrCase<"lrf26", 58>;
def SentientLRF27 : I32EnumAttrCase<"lrf27", 59>;
def SentientLRF28 : I32EnumAttrCase<"lrf28", 60>;
def SentientLRF29 : I32EnumAttrCase<"lrf29", 61>;
def SentientLRF30 : I32EnumAttrCase<"lrf30", 62>;
def SentientLRF31 : I32EnumAttrCase<"lrf31", 63>;

def SentientComputePort : I32EnumAttr<"SentientComputePort",
    "Built-in ports in compute units of Sentient",
    [
      SentientNone,
      SentientZero,
      SentientOne,
      SentientTwo,
      SentientThree,
      SentientLRF0,
      SentientLRF1,
      SentientLRF2,
      SentientLRF3,
      SentientLRF4,
      SentientLRF5,
      SentientLRF6,
      SentientLRF7,
      SentientLRF8,
      SentientLRF9,
      SentientLRF10,
      SentientLRF11,
      SentientLRF12,
      SentientLRF13,
      SentientLRF14,
      SentientLRF15,
      SentientLRF16,
      SentientLRF17,
      SentientLRF18,
      SentientLRF19,
      SentientLRF20,
      SentientLRF21,
      SentientLRF22,
      SentientLRF23,
      SentientLRF24,
      SentientLRF25,
      SentientLRF26,
      SentientLRF27,
      SentientLRF28,
      SentientLRF29,
      SentientLRF30,
      SentientLRF31,
      SentientXRF,
      SentientIRF0,
      SentientIRF1,
      SentientLatch,
      SentientLX,
      SentientL0,
      SentientNbrSlice,
      SentientPE,
      SentientPT,
      SentientSFP,
      SentientSFPRing,
      SentientNfwd0,
      SentientNfwd2,
      SentientOpA,
      SentientOpB,
      SentientOpC,
      SentientResult,
      SentientEast,
      SentientSouth,
      SentientWest,
      SentientNorth,
      SentientISTATE0,
      SentientISTATE1,
      SentientISTATE2,
      SentientISTATE3,
      SentientCrossPTNorthLink
    ]> {
        let genSpecializedAttr = 0;
    }

def SentientComputePortAttr : EnumAttr<Sentient_Dialect, SentientComputePort, "compute_port">;

def SentientComputePortArrayAttr : TypedArrayAttrBase<
    SentientComputePortAttr, "Sentient compute port array attribute">;

//FMA/FNMS mode operations
def SentientFMA : I32EnumAttrCase<"fused_mul_add", 0>;
def SentientFNMS : I32EnumAttrCase<"fused_neg_mul_sub", 1>;

def SentientFMAmode : I32EnumAttr<"SentientFMAmode",
    "built-in FMA/FNMS mode for sentient.",
    [
      SentientFMA,
      SentientFNMS
    ]> {
        let genSpecializedAttr = 0;
    }

def SentientFMAmodeAttr : EnumAttr<Sentient_Dialect, SentientFMAmode, "fma_mode_op">;

//Register types
def SentientRegUnknown : I32EnumAttrCase<"unknown", 0>;
def SentientRegImm : I32EnumAttrCase<"imm", 1>;
def SentientRegJCR : I32EnumAttrCase<"jcr", 2>;
def SentientRegLCCR : I32EnumAttrCase<"lccr", 3>;
def SentientRegLRF : I32EnumAttrCase<"lrf", 4>;
def SentientRegXRFRDPTR : I32EnumAttrCase<"xrfrdptr", 5>;
def SentientRegXRFWRPTR : I32EnumAttrCase<"xrfwrptr", 6>;
def SentientRegLAR : I32EnumAttrCase<"lar", 7>;
def SentientRegLBR : I32EnumAttrCase<"lbr", 8>;
def SentientRegEAR : I32EnumAttrCase<"ear", 9>;
def SentientRegEBR : I32EnumAttrCase<"ebr", 10>;
def SentientRegGTR : I32EnumAttrCase<"gtr", 11>;
def SentientRegMVR : I32EnumAttrCase<"mvr", 12>;
def SentientRegUnrelated : I32EnumAttrCase<"unrelated", 13>;

def SentientRegType : I32EnumAttr<"SentientRegType",
    "built-in register types for sentient.",
    [
      SentientRegUnknown,
      SentientRegImm,
      SentientRegJCR,
      SentientRegLCCR,
      SentientRegLRF,
      SentientRegXRFRDPTR,
      SentientRegXRFWRPTR,
      SentientRegLAR,
      SentientRegLBR,
      SentientRegEAR,
      SentientRegEBR,
      SentientRegGTR,
      SentientRegMVR,
      SentientRegUnrelated
    ]> {
        let genSpecializedAttr = 0;
    }

def SentientRegTypeAttr : EnumAttr<Sentient_Dialect, SentientRegType, "reg_type">;

def SentientRegTypeArrayAttr : TypedArrayAttrBase<
    SentientRegTypeAttr, "Sentient Register Type extension array attribute">;

//XRF mode operations
def XRFPtrAdd : I32EnumAttrCase<"add", 0>;
def XRFPtrCopy : I32EnumAttrCase<"copy", 1>;

def SentientXRFAddmode : I32EnumAttr<"SentientXRFAddmode",
    "built-in xrfaccess mode for sentient.",
    [
      XRFPtrAdd,
      XRFPtrCopy
    ]> {
        let genSpecializedAttr = 0;
    }

def SentientXRFAddmodeAttr : EnumAttr<Sentient_Dialect, SentientXRFAddmode, "xrf_mode_op">;

// Fold modes for PE+SFP
def SentientFold_N : I32EnumAttrCase<"none", 0>;
def SentientFold_A : I32EnumAttrCase<"fold_A", 1>;
def SentientFold_B : I32EnumAttrCase<"fold_B", 2>;
def SentientFold_AB_A : I32EnumAttrCase<"fold_AB_A", 3>;
def SentientFold_AB_B : I32EnumAttrCase<"fold_AB_B", 4>;
def SentientFold_AB_Both : I32EnumAttrCase<"fold_AB_Both", 5>;

def SentientFoldMode : I32EnumAttr<"SentientFoldMode",
    "Fold A/B/AB modes for PE+SFP",
    [
      SentientFold_N,
      SentientFold_A,
      SentientFold_B,
      SentientFold_AB_A,
      SentientFold_AB_B,
      SentientFold_AB_Both
    ]> {
        let genSpecializedAttr = 0;
    }

def SentientFoldModeAttr : EnumAttr<Sentient_Dialect, SentientFoldMode, "fold_mode">;

//Vector Binary operations
// and and or are reserved keyword in c++
def SentientBinaryAnd : I32EnumAttrCase<"and0", 0>;
def SentientBinaryOr : I32EnumAttrCase<"or0", 1>;
def SentientBinaryXnor : I32EnumAttrCase<"xnor", 2>;
def SentientBinaryAndNot : I32EnumAttrCase<"and_not", 3>;
def SentientBinaryMin : I32EnumAttrCase<"min", 4>;
def SentientBinaryMax : I32EnumAttrCase<"max", 5>;
def SentientBinaryAbsMin : I32EnumAttrCase<"abs_min", 6>;
def SentientBinaryAbsMax : I32EnumAttrCase<"abs_max", 7>;
def SentientBinaryAdd : I32EnumAttrCase<"add", 8>;
def SentientBinaryMul : I32EnumAttrCase<"mul", 9>;
def SentientBinarySub : I32EnumAttrCase<"sub", 10>;
def SentientBinaryMulDiv2 : I32EnumAttrCase<"mul_div2", 11>;
def SentientGCVTImm0 : I32EnumAttrCase<"gcvt_imm0", 12>;
def SentientGCVTImm4 : I32EnumAttrCase<"gcvt_imm4", 13>;
def SentientGCVTImm24 : I32EnumAttrCase<"gcvt_imm24", 14>;
def SentientGCVTImm28 : I32EnumAttrCase<"gcvt_imm28", 15>;
def SentientFCVTImm2 : I32EnumAttrCase<"fcvt_imm2", 16>;
def SentientFCVTImm3 : I32EnumAttrCase<"fcvt_imm3", 17>;
def SentientFCVTImm4 : I32EnumAttrCase<"fcvt_imm4", 18>;
def SentientFCVTImm7 : I32EnumAttrCase<"fcvt_imm7", 19>;
def SentientMerge8l : I32EnumAttrCase<"merge8l", 101>;
def SentientMerge8h : I32EnumAttrCase<"merge8h", 102>;
def SentientMerge16l : I32EnumAttrCase<"merge16l", 103>;
def SentientMerge16h : I32EnumAttrCase<"merge16h", 104>;
def SentientMerge32l : I32EnumAttrCase<"merge32l", 105>;
def SentientMerge32h : I32EnumAttrCase<"merge32h", 106>;
def SentientMerge64l : I32EnumAttrCase<"merge64l", 107>;
def SentientMerge64h : I32EnumAttrCase<"merge64h", 108>;
def SentientPack0 : I32EnumAttrCase<"pack0", 109>;
def SentientPack1 : I32EnumAttrCase<"pack1", 110>;
def SentientPack2 : I32EnumAttrCase<"pack2", 111>;
def SentientPack3 : I32EnumAttrCase<"pack3", 112>;
def SentientPack4 : I32EnumAttrCase<"pack4", 113>;
def SentientPack5 : I32EnumAttrCase<"pack5", 114>;
def SentientPack6 : I32EnumAttrCase<"pack6", 115>;
def SentientPack7 : I32EnumAttrCase<"pack7", 116>;
def SentientPack8 : I32EnumAttrCase<"pack8", 117>;
def SentientPack9 : I32EnumAttrCase<"pack9", 118>;
def SentientPack12 : I32EnumAttrCase<"pack12", 119>;
def SentientPack13 : I32EnumAttrCase<"pack13", 120>;
def SentientPack14 : I32EnumAttrCase<"pack14", 121>;
def SentientPack15 : I32EnumAttrCase<"pack15", 122>;
def SentientPack16 : I32EnumAttrCase<"pack16", 123>;
def SentientPack17 : I32EnumAttrCase<"pack17", 124>;
def SentientPack18 : I32EnumAttrCase<"pack18", 125>;
def SentientPack19 : I32EnumAttrCase<"pack19", 126>;
def SentientPack20 : I32EnumAttrCase<"pack20", 127>;
def SentientPack21 : I32EnumAttrCase<"pack21", 128>;
def SentientPack22 : I32EnumAttrCase<"pack22", 129>;
def SentientPack23 : I32EnumAttrCase<"pack23", 130>;
def SentientPack24 : I32EnumAttrCase<"pack24", 131>;
def SentientPack25 : I32EnumAttrCase<"pack25", 132>;
def SentientPack26 : I32EnumAttrCase<"pack26", 133>;
def SentientPack27 : I32EnumAttrCase<"pack27", 134>;
def SentientBinaryCompareNeq : I32EnumAttrCase<"fcmp_neq", 135>;
def SentientBinaryCompareEq : I32EnumAttrCase<"fcmp_eq", 136>;
def SentientBinaryCompareLt : I32EnumAttrCase<"fcmp_lt", 137>;
def SentientBinaryCompareLe : I32EnumAttrCase<"fcmp_le", 138>;

def SentientBinaryOperator : I32EnumAttr<"SentientBinaryOperator",
    "built-in operators supported by Sentient binary ops.",
    [
      SentientBinaryAnd,
      SentientBinaryOr,
      SentientBinaryXnor,
      SentientBinaryAndNot,
      SentientBinaryMin,
      SentientBinaryMax,
      SentientBinaryAbsMin,
      SentientBinaryAbsMax,
      SentientBinaryAdd,
      SentientBinaryMul,
      SentientBinarySub,
      SentientBinaryMulDiv2,
      SentientGCVTImm0,
      SentientGCVTImm4,
      SentientGCVTImm24,
      SentientGCVTImm28,
      SentientFCVTImm2,
      SentientFCVTImm3,
      SentientFCVTImm4,
      SentientFCVTImm7,
      SentientMerge8l,
      SentientMerge8h,
      SentientMerge16l,
      SentientMerge16h,
      SentientMerge32l,
      SentientMerge32h,
      SentientMerge64l,
      SentientMerge64h,
      SentientPack0,
      SentientPack1,
      SentientPack2,
      SentientPack3,
      SentientPack4,
      SentientPack5,
      SentientPack6,
      SentientPack7,
      SentientPack8,
      SentientPack9,
      SentientPack12,
      SentientPack13,
      SentientPack14,
      SentientPack15,
      SentientPack16,
      SentientPack17,
      SentientPack18,
      SentientPack19,
      SentientPack20,
      SentientPack21,
      SentientPack22,
      SentientPack23,
      SentientPack24,
      SentientPack25,
      SentientPack26,
      SentientPack27,
      SentientBinaryCompareNeq,
      SentientBinaryCompareEq,
      SentientBinaryCompareLt,
      SentientBinaryCompareLe,
    ]> {
        let genSpecializedAttr = 0;
    }

def SentientBinaryOperatorAttr : EnumAttr<Sentient_Dialect, SentientBinaryOperator, "binary_operator">;

//Vector Ternary operations
def SentientTernarySelect : I32EnumAttrCase<"select", 0>;
def SentientTernaryOperator : I32EnumAttr<"SentientTernaryOperator",
    "built-in operators supported by Sentient Ternary ops.",
    [
      SentientTernarySelect
    ]> {
        let genSpecializedAttr = 0;
    }

def SentientTernaryOperatorAttr : EnumAttr<Sentient_Dialect, SentientTernaryOperator, "ternary_operator">;

//Comparison attributes for sentient.if
def CMPI_P_EQ  : I32EnumAttrCase<"eq", 0>;
def CMPI_P_NE  : I32EnumAttrCase<"ne", 1>;
def CMPI_P_SLT : I32EnumAttrCase<"slt", 2>;
def CMPI_P_SLE : I32EnumAttrCase<"sle", 3>;
def CMPI_P_SGT : I32EnumAttrCase<"sgt", 4>;
def CMPI_P_SGE : I32EnumAttrCase<"sge", 5>;


def CmpIPredicate : I32EnumAttr<
    "CmpIPredicate", "",
    [CMPI_P_EQ, CMPI_P_NE, CMPI_P_SLT, CMPI_P_SLE, CMPI_P_SGT,
     CMPI_P_SGE]> {
  //let cppNamespace = "::mlir";
  let genSpecializedAttr = 0;
}

def CmpIPredicateAttr : EnumAttr<Sentient_Dialect, CmpIPredicate ,"cmp_predicate">;

def SentientUnrollx1 : I32EnumAttrCase<"x1", 0>;
def SentientUnrollx2 : I32EnumAttrCase<"x2", 1>;
def SentientUnrollx3 : I32EnumAttrCase<"x3", 2>;
def SentientUnrollx4 : I32EnumAttrCase<"x4", 3>;
def SentientUnrollx8 : I32EnumAttrCase<"x8", 4>;

def SentientUnrollFactor : I32EnumAttr<"SentientUnrollFactor",
    "built-in unroll factor options.",
    [
      SentientUnrollx1,
      SentientUnrollx2,
      SentientUnrollx3, // used only for reduction
      SentientUnrollx4,
      SentientUnrollx8, // used for non-reduction
    ]> {
        let genSpecializedAttr = 0;
    }

def SentientUnrollFactorAttr : EnumAttr<Sentient_Dialect, SentientUnrollFactor, "unroll_factor">;

def SentientNoShuffle : I32EnumAttrCase<"noshuffle", 0>;
def SentientShuffleSplat : I32EnumAttrCase<"splat", 1>;
def SentientShuffleRotate : I32EnumAttrCase<"rotate", 2>;
def SentientSplat2B : I32EnumAttrCase<"splat2b", 3>;
def SentientSplat4B : I32EnumAttrCase<"splat4b", 4>;
def SentientSplat16B : I32EnumAttrCase<"splat16b", 5>;
def SentientZeroPad16B : I32EnumAttrCase<"zpad16b", 6>;
def SentientMasked2B : I32EnumAttrCase<"masked2b", 7>;
def SentientMasked16B : I32EnumAttrCase<"masked16b", 8>;

def SentientShuffleMode : I32EnumAttr<"SentientShuffleMode",
    "built-in shuffle options for loads and stores.",
    [
      SentientNoShuffle,
      SentientShuffleSplat,
      SentientShuffleRotate,
      SentientSplat2B,
      SentientSplat4B,
      SentientSplat16B,
      SentientZeroPad16B,
      SentientMasked2B,
      SentientMasked16B
    ]> {
        let genSpecializedAttr = 0;
    }

def SentientShuffleModeAttr : EnumAttr<Sentient_Dialect, SentientShuffleMode, "shuffle_mode">;

def SentientSyncModeSend : I32EnumAttrCase<"send", 0>;
def SentientSyncModeRecv : I32EnumAttrCase<"recv", 1>;
def SentientSyncModeSendRecv : I32EnumAttrCase<"sendrecv", 2>;

def SentientSyncMode : I32EnumAttr<"SentientSyncMode",
    "built-in sync tag attribute options",
    [
       SentientSyncModeSend,
       SentientSyncModeRecv,
       SentientSyncModeSendRecv,
    ]> {
        let genSpecializedAttr = 0;

    }

def SentientSyncModeAttr : EnumAttr<Sentient_Dialect, SentientSyncMode, "sync_mode">;

def SentientUnitSfpAttr : I32EnumAttrCase<"sfp", 0>;
def SentientUnitPeAttr : I32EnumAttrCase<"pe", 1>;
def SentientUnitL0Attr : I32EnumAttrCase<"l0", 2>;
def SentientUnitPtAttr : I32EnumAttrCase<"pt", 3>;
def SentientUnitL0luAttr : I32EnumAttrCase<"l0lu", 4>;
def SentientUnitL0suAttr : I32EnumAttrCase<"l0su", 5>;
def SentientUnitLxluAttr : I32EnumAttrCase<"lxlu", 6>;
def SentientUnitLxluNAttr : I32EnumAttrCase<"lxluN", 7>;
def SentientUnitLxlu0Attr : I32EnumAttrCase<"lxlu0", 8>;
def SentientUnitLxlu1Attr : I32EnumAttrCase<"lxlu1", 9>;
def SentientUnitLxsuAttr : I32EnumAttrCase<"lxsu", 10>;
def SentientUnitLxsuNAttr : I32EnumAttrCase<"lxsuN", 11>;
def SentientUnitLxsu0Attr : I32EnumAttrCase<"lxsu0", 12>;
def SentientUnitLxsu1Attr : I32EnumAttrCase<"lxsu1", 13>;
def SentientUnitL3luAttr : I32EnumAttrCase<"l3lu", 14>;
def SentientUnitL3suAttr : I32EnumAttrCase<"l3su", 15>;

def SentientConsumer : I32EnumAttr<"SentientLoadConsumer",
    "built-in consumer options for load.",
    [
      SentientUnitSfpAttr,
      SentientUnitPeAttr,
      SentientUnitL0Attr,
      SentientUnitPtAttr,
      SentientUnitL0luAttr,
      SentientUnitL0suAttr,
      SentientUnitLxluAttr,
      SentientUnitLxluNAttr,
      SentientUnitLxlu0Attr,
      SentientUnitLxlu1Attr,
      SentientUnitLxsuAttr,
      SentientUnitLxsuNAttr,
      SentientUnitLxsu0Attr,
      SentientUnitLxsu1Attr,
      SentientUnitL3luAttr,
      SentientUnitL3suAttr,
    ]> {
        let genSpecializedAttr = 0;
    }

def SentientConsumerAttr : EnumAttr<Sentient_Dialect, SentientConsumer, "consumer_unit">;
def SentientConsumerArrayAttr : TypedArrayAttrBase<
    SentientConsumerAttr, "Sentient consumer unit type extension array attribute">;

def SentientSplatPadNone : I32EnumAttrCase<"none", 0>;
def SentientSplatPadLeft : I32EnumAttrCase<"left", 1>;

def SentientSplatPad : I32EnumAttr<"SentientSplatPad",
    "padding attribute for the splat operation",
    [
      SentientSplatPadNone,
      SentientSplatPadLeft
    ]> {
        let genSpecializedAttr = 0;
    }
def SentientSplatPadAttr : EnumAttr<Sentient_Dialect, SentientSplatPad, "splat_pad">;

def SentientEstExpA : I32EnumAttrCase<"exp_a", 0>;
def SentientEstExpB : I32EnumAttrCase<"exp_b", 1>;
def SentientEstRec : I32EnumAttrCase<"rec", 2>;
def SentientEstLn : I32EnumAttrCase<"ln", 3>;
def SentientEstRsqrt : I32EnumAttrCase<"rsqrt", 4>;
def SentientEstSigmS : I32EnumAttrCase<"sigm_slope", 5>;
def SentientEstSigmO : I32EnumAttrCase<"sigm_offset", 6>;
def SentientEstTanhS : I32EnumAttrCase<"tanh_slope", 7>;
def SentientEstTanhO : I32EnumAttrCase<"tanh_offset", 8>;
def SentientFloor : I32EnumAttrCase<"floor", 9>;
def SentientReductionAdd : I32EnumAttrCase<"reduction_add", 10>;
def SentientReductionMin : I32EnumAttrCase<"reduction_min", 11>;
def SentientReductionMax : I32EnumAttrCase<"reduction_max", 12>;
def SentientReductionAbsMin : I32EnumAttrCase<"reduction_abs_min", 13>;
def SentientReductionAbsMax : I32EnumAttrCase<"reduction_abs_max", 14>;
def SentientFastExp : I32EnumAttrCase<"fast_exp", 15>;
def SentientGCVTImm1 : I32EnumAttrCase<"gcvt_imm1", 16>;
def SentientGCVTImm2 : I32EnumAttrCase<"gcvt_imm2", 17>;
def SentientGCVTImm5 : I32EnumAttrCase<"gcvt_imm5", 18>;
def SentientGCVTImm6 : I32EnumAttrCase<"gcvt_imm6", 19>;
def SentientGCVTImm8 : I32EnumAttrCase<"gcvt_imm8", 20>;
def SentientGCVTImm16 : I32EnumAttrCase<"gcvt_imm16", 21>;
def SentientGCVTImm17 : I32EnumAttrCase<"gcvt_imm17", 22>;
def SentientFCVTImm0 : I32EnumAttrCase<"fcvt_imm0", 23>;
def SentientFCVTImm1 : I32EnumAttrCase<"fcvt_imm1", 24>;
def SentientFCVTImm5 : I32EnumAttrCase<"fcvt_imm5", 25>;
def SentientFCVTImm6 : I32EnumAttrCase<"fcvt_imm6", 26>;

def SentientUnary : I32EnumAttr<"SentientUnary",
    "Attribute for unary op",
    [
      SentientEstExpA,
      SentientEstExpB,
      SentientEstRec,
      SentientEstLn,
      SentientEstRsqrt,
      SentientEstSigmS,
      SentientEstSigmO,
      SentientEstTanhS,
      SentientEstTanhO,
      SentientFloor,
      SentientReductionAdd,
      SentientReductionMin,
      SentientReductionMax,
      SentientReductionAbsMin,
      SentientReductionAbsMax,
      SentientFastExp,
      SentientGCVTImm1,
      SentientGCVTImm2,
      SentientGCVTImm5,
      SentientGCVTImm6,
      SentientGCVTImm8,
      SentientGCVTImm16,
      SentientGCVTImm17,
      SentientFCVTImm0,
      SentientFCVTImm1,
      SentientFCVTImm5,
      SentientFCVTImm6
    ]> {
      let genSpecializedAttr = 0;
    }
def SentientUnaryAttr : EnumAttr<Sentient_Dialect, SentientUnary, "unary">;

def SentientRoutingDirPseudoRandom : I32EnumAttrCase<"PseudoRandom", 0>;
def SentientRoutingDirCounterClockwise : I32EnumAttrCase<"CounterClockwise", 1>;
def SentientRoutingDirClockwise : I32EnumAttrCase<"Clockwise", 2>;
def SentientRoutingDirBoth : I32EnumAttrCase<"BothWays", 3>;

def SentientRoutingDirection : I32EnumAttr<"SentientRoutingDirection",
    "Sentient routing direction:",
    [
      SentientRoutingDirPseudoRandom,
      SentientRoutingDirCounterClockwise,
      SentientRoutingDirClockwise,
      SentientRoutingDirBoth
    ]> {
    let genSpecializedAttr = 0;
    let cppNamespace = "::mlir::sentient";
}

def SentientRoutingDirectionAttr : EnumAttr<Sentient_Dialect, SentientRoutingDirection, "direction">; 

#endif // DCC_SRC_DIALECT_SENTIENT_TYPES_TD
