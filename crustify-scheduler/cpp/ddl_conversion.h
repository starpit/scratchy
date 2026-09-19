/************************************************************
 * IBM Confidential
 * (C) Copyright IBM Corp. 2023, 2025
 ************************************************************/

/**
 * @file ddl_conversion.h
 * @brief DDL (Deep Learning Description Language) to DSC (Design Space
 * Configuration) bidirectional conversion
 *
 * This header defines the core infrastructure for converting between DDL (an
 * MLIR-based intermediate representation) and DSC (the internal design space
 * configuration format). It provides template mapping, dimension tracking, type
 * definitions, and conversion logic for various neural network operations
 * across different hardware architectures.
 *
 * Key Components:
 * - DdlArch: Maps operations to architecture-specific DDL templates
 * - DdlInterface: Tracks DDL-to-DSC mappings (dimensions, types, tensors,
 * operations)
 * - DdlConversion: Main conversion class handling bidirectional translation
 *
 * Supported Architectures:
 * - RCUDD1A_ISA (RCU DD1A)
 * - MPW4_ISA (MPW4)
 * - SEN1P5_ISA (Sentient 1.5)
 *
 * @see ddc/ddl/ddl.h for DDL MLIR operations
 * @see dsc/designSpaceConfig.h for DSC data structures
 */

#ifndef DDL_CONVERSION_H_
#define DDL_CONVERSION_H_

#include <dsc/designSpaceConfig.h>
#include <dsc/superdsc.h>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>

#include "../ddc_metadata.h"
#include "ddl.h"

using namespace mlir;
using namespace mlir::ddl;

// conversion functions from/to DDL language
namespace ddc {

/**
 * @struct DdlArch
 * @brief Associates a DDL template file with an optional dedicated hardware
 * architecture
 *
 * Used to map operation types to their corresponding DDL template
 * implementations, with optional architecture-specific variants.
 */
struct DdlArch {
  std::string filename;  ///< DDL template filename (e.g., "bmm.ddl")
  std::optional<IsaCoreGen>
      dedicatedArch;  ///< Optional architecture constraint
};

namespace {  // anonymous namespace for local ("static") variables and functions

/**
 * @brief Operation function to DDL template mapping
 *
 * Maps each OpFuncs enum value to one or more DDL template files, with optional
 * architecture-specific variants. This enables automatic template selection
 * based on operation type and target hardware.
 *
 * Template Categories:
 * - Matrix Operations: bmm.ddl, bmm_dd1.ddl, bmm_sen1p5.ddl
 * - Unary Operations: unary_parallel.ddl, unary_pipeline.ddl
 * - Broadcast Operations: broadcast_ops.ddl
 * - Reduction Operations: summeanmaxexx2.ddl, summeanmaxexx2_fp32.ddl
 * - Quantization: quantization_*.ddl variants
 * - Convolution: convolution2d*.ddl variants
 * - Normalization: layernorm*.ddl variants
 * - Pooling: pooling.ddl
 * - Special Operations: rope.ddl, topk.ddl, lstmactp2.ddl
 */
const std::map<OpFuncs, std::vector<DdlArch>> opFuncToDdlTemplate = {
    {OpFuncs::MATMUL_FWD,
     {{"bmm.ddl", RCUDD1A_ISA},
      {"bmm_dd1.ddl", MPW4_ISA},
      {"bmm_sen1p5.ddl", SEN1P5_ISA}}},
    {OpFuncs::MATMUL_FP8_FWD,
     {{"bmm.ddl", RCUDD1A_ISA},
      {"bmm_dd1.ddl", MPW4_ISA},
      {"bmm_sen1p5.ddl", SEN1P5_ISA}}},
    {OpFuncs::MATMUL_INT8_FWD,
     {{"bmm.ddl", RCUDD1A_ISA},
      {"bmm_dd1.ddl", MPW4_ISA},
      {"bmm_sen1p5.ddl", SEN1P5_ISA}}},
    {OpFuncs::MATMUL_INT4_FWD,
     {{"bmm.ddl", RCUDD1A_ISA},
      {"bmm_dd1.ddl", MPW4_ISA},
      {"bmm_sen1p5.ddl", SEN1P5_ISA}}},
    {OpFuncs::BATCHMATMUL_FWD,
     {{"bmm.ddl", RCUDD1A_ISA},
      {"bmm_dd1.ddl", MPW4_ISA},
      {"bmm_sen1p5.ddl", SEN1P5_ISA}}},
    {OpFuncs::BATCHMATMUL_MXFP4W_FWD, {{"bmm_sen1p5.ddl", SEN1P5_ISA}}},
    {OpFuncs::BATCHMATMUL_FP8_FWD,
     {{"bmm.ddl", RCUDD1A_ISA},
      {"bmm_dd1.ddl", MPW4_ISA},
      {"bmm_sen1p5.ddl", SEN1P5_ISA}}},
    {OpFuncs::BATCHMATMUL_FP8_FWD_MB,
     {{"bmm.ddl", RCUDD1A_ISA},
      {"bmm_dd1.ddl", MPW4_ISA},
      {"bmm_sen1p5.ddl", SEN1P5_ISA}}},
    {OpFuncs::BATCHMATMUL_INT8_FWD,
     {{"bmm.ddl", RCUDD1A_ISA},
      {"bmm_dd1.ddl", MPW4_ISA},
      {"bmm_sen1p5.ddl", SEN1P5_ISA}}},
    {OpFuncs::BATCHMATMUL_INT8_FWD_MBKG3,
     {{"bmm.ddl", RCUDD1A_ISA},
      {"bmm_dd1.ddl", MPW4_ISA},
      {"bmm_sen1p5.ddl", SEN1P5_ISA}}},
    {OpFuncs::BATCHMATMUL_INT4_FWD,
     {{"bmm.ddl", RCUDD1A_ISA},
      {"bmm_dd1.ddl", MPW4_ISA},
      {"bmm_sen1p5.ddl", SEN1P5_ISA}}},
    {OpFuncs::BATCHMATMUL_XRF_FWD,
     {{"bmm.ddl", RCUDD1A_ISA},
      {"bmm_dd1.ddl", MPW4_ISA},
      {"bmm_sen1p5.ddl", SEN1P5_ISA}}},
    {OpFuncs::BATCHMATMUL_XRF_FP8_FWD,
     {{"bmm.ddl", RCUDD1A_ISA},
      {"bmm_dd1.ddl", MPW4_ISA},
      {"bmm_sen1p5.ddl", SEN1P5_ISA}}},
    {OpFuncs::BATCHMATMUL_XRF_INT8_FWD,
     {{"bmm.ddl", RCUDD1A_ISA},
      {"bmm_dd1.ddl", MPW4_ISA},
      {"bmm_sen1p5.ddl", SEN1P5_ISA}}},
    {OpFuncs::BATCHMATMUL_XRF_INT4_FWD,
     {{"bmm.ddl", RCUDD1A_ISA},
      {"bmm_dd1.ddl", MPW4_ISA},
      {"bmm_sen1p5.ddl", SEN1P5_ISA}}},
    {OpFuncs::BATCHMATMUL_MXFP8_FWD, {{"bmm_sen1p5.ddl", SEN1P5_ISA}}},
    {OpFuncs::SCALED_GROUP_MATMUL_FP4_FWD, {{"bmm_sen1p5.ddl", SEN1P5_ISA}}},
    {OpFuncs::RECIPROCAL, {{"unary_parallel.ddl"}}},
    {OpFuncs::LAYERNORM_SCALE,
     {{"unary_parallel.ddl"}, {"layernormscale_32.ddl"}}},
    {OpFuncs::TANH_FWD, {{"unary_parallel.ddl"}}},
    {OpFuncs::TANH_BWD, {{"gelu_bwd.ddl"}}},
    {OpFuncs::LEAKYRELU_FWD, {{"unary_pipeline.ddl"}}},
    {OpFuncs::RELU_FWD, {{"unary_parallel.ddl"}}},
    {OpFuncs::RELU6_FWD, {{"unary_pipeline.ddl"}}},
    {OpFuncs::CLIP_FWD, {{"unary_pipeline.ddl"}, {"unary_parallel.ddl"}}},
    {OpFuncs::FAST_EXP_FWD, {{"unary_pipeline.ddl"}, {"unary_parallel.ddl"}}},
    {OpFuncs::GELU_FWD, {{"unary_parallel.ddl"}}},
    {OpFuncs::GELU_BWD, {{"gelu_bwd.ddl"}}},
    {OpFuncs::SIGMOID_FWD, {{"unary_parallel.ddl"}}},
    {OpFuncs::FAST_SIGMOID_FWD, {{"unary_pipeline.ddl"}}},
    {OpFuncs::SILU_FWD, {{"unary_parallel.ddl"}}},
    {OpFuncs::EXP_FWD, {{"unary_parallel.ddl"}, {"unary_pipeline.ddl"}}},
    {OpFuncs::LOG_FWD, {{"unary_pipeline.ddl"}}},
    {OpFuncs::LAYERNORM_NORM,
     {{"layernormnorm.ddl"}, {"layernormnorm_fp32.ddl"}}},
    {OpFuncs::LAYERNORM_BWDNORM, {{"layernormbackwardnorm.ddl"}}},
    {OpFuncs::MAXPOOL_FWD, {{"pooling.ddl"}}},
    {OpFuncs::AVGPOOL_FWD, {{"pooling.ddl"}}},
    {OpFuncs::ADD, {{"broadcast_ops.ddl"}}},
    {OpFuncs::ADD_I32_TO_I32, {{"broadcast_ops.ddl"}}},
    {OpFuncs::ADD_I64_TO_I64, {{"broadcast_ops.ddl"}}},
    {OpFuncs::MUL_I32_TO_I32, {{"broadcast_ops.ddl"}}},
    {OpFuncs::SUB, {{"broadcast_ops.ddl"}}},
    {OpFuncs::MUL, {{"broadcast_ops.ddl"}}},
    {OpFuncs::REVSUB, {{"broadcast_ops.ddl"}}},
    {OpFuncs::REALDIV, {{"broadcast_ops.ddl"}}},
    {OpFuncs::BIASADD, {{"broadcast_ops.ddl"}}},
    {OpFuncs::STRIDED_ADD, {{"broadcast_ops.ddl"}}},
    {OpFuncs::BATCHNORM_FWD, {{"broadcast_ops.ddl"}}},
    {OpFuncs::FNMS, {{"broadcast_ops.ddl"}}},
    {OpFuncs::WHERE3, {{"broadcast_ops.ddl"}}},
    {OpFuncs::MAXIMUM, {{"broadcast_ops.ddl"}}},
    {OpFuncs::MINIMUM, {{"broadcast_ops.ddl"}}},
    {OpFuncs::SINKCORRECTIONFACTOR, {{"broadcast_ops.ddl"}}},
    {OpFuncs::SUM, {{"summeanmaxexx2.ddl"}, {"summeanmaxexx2_fp32.ddl"}}},
    {OpFuncs::SUM_NONSTICK,
     {{"summeanmaxexx2.ddl"}, {"summeanmaxexx2_fp32.ddl"}}},
    {OpFuncs::MEAN, {{"summeanmaxexx2.ddl"}, {"summeanmaxexx2_fp32.ddl"}}},
    {OpFuncs::MEAN_NONSTICK,
     {{"summeanmaxexx2.ddl"}, {"summeanmaxexx2_fp32.ddl"}}},
    {OpFuncs::MAX, {{"summeanmaxexx2.ddl"}, {"summeanmaxexx2_fp32.ddl"}}},
    {OpFuncs::MAX_NONSTICK,
     {{"summeanmaxexx2.ddl"}, {"summeanmaxexx2_fp32.ddl"}}},
    {OpFuncs::ABSMAX, {{"summeanmaxexx2.ddl"}, {"summeanmaxexx2_fp32.ddl"}}},
    {OpFuncs::ABSMAX_NONSTICK,
     {{"summeanmaxexx2.ddl"}, {"summeanmaxexx2_fp32.ddl"}}},
    {OpFuncs::MIN, {{"summeanmaxexx2.ddl"}, {"summeanmaxexx2_fp32.ddl"}}},
    {OpFuncs::MIN_NONSTICK,
     {{"summeanmaxexx2.ddl"}, {"summeanmaxexx2_fp32.ddl"}}},
    {OpFuncs::PROD_NONSTICK,
     {{"summeanmaxexx2.ddl"}, {"summeanmaxexx2_fp32.ddl"}}},
    {OpFuncs::EXX2,
     {{"summeanmaxexx2.ddl"}, {"summeanmaxexx2_fp32.ddl"}, {"exx2_32.ddl"}}},
    {OpFuncs::EXX2_ZEROMEAN,
     {{"summeanmaxexx2.ddl"}, {"summeanmaxexx2_fp32.ddl"}}},
    {OpFuncs::CSQ_INT8_WT, {{"quantization_no_pad.ddl"}}},
    {OpFuncs::Q_FP8, {{"quantization_single_pad.ddl"}}},
    {OpFuncs::Q_FP8_CH, {{"quantization_double_pad.ddl"}}},
    {OpFuncs::Q_FP8_WT, {{"quantization_no_pad.ddl"}}},
    {OpFuncs::Q_FP8_MB, {{"quantization_single_pad.ddl"}}},
    // {OpFuncs::CSQ_INT4, {{"quantization_double_pad.ddl"}}},
    {OpFuncs::CSQ_INT4, {{"quantization_double_pad.ddl"}}},
    {OpFuncs::CSQ_INT4_WT, {{"quantization_no_pad.ddl"}}},
    {OpFuncs::CSQ_INT8, {{"quantization_single_pad.ddl"}}},
    {OpFuncs::CSQ_INT8_CH, {{"quantization_double_pad.ddl"}}},
    {OpFuncs::CSQ_INT8_MB, {{"quantization_single_pad.ddl"}}},
    {OpFuncs::CSQ_INT8_MB_V2, {{"quantization_single_pad_v2.ddl"}}},
    {OpFuncs::CSQ_INT8_V2, {{"quantization_single_pad_v2.ddl"}}},
    {OpFuncs::QUANT_SCALE_PER_TOKEN, {{"quant_scale_per_token.ddl"}}},
    {OpFuncs::QUANT_SCALE_PER_TOKEN_FP8, {{"quant_scale_per_token.ddl"}}},
    {OpFuncs::CONV2D_INT8_FWD,
     {{"convolution2d.ddl", RCUDD1A_ISA}, {"convolution2d_dd1.ddl", MPW4_ISA}}},
    {OpFuncs::CONV2D_FWD,
     {{"convolution2d.ddl", RCUDD1A_ISA}, {"convolution2d_dd1.ddl", MPW4_ISA}}},
    {OpFuncs::CONV2D_INT4_FWD,
     {{"convolution2d.ddl", RCUDD1A_ISA}, {"convolution2d_dd1.ddl", MPW4_ISA}}},
    {OpFuncs::CONV2D_FP8_FWD,
     {{"convolution2d.ddl", RCUDD1A_ISA}, {"convolution2d_dd1.ddl", MPW4_ISA}}},
    {OpFuncs::CONV2D_FWD_GEN_OS1, {{"convolution2d_os1.ddl"}}},
    {OpFuncs::CONV2D_FWD_OS1, {{"convolution2d_os1.ddl"}}},
    {OpFuncs::CONV2D_INT8_FWD_OS1, {{"convolution2d_os1.ddl"}}},
    {OpFuncs::CONV2D_XRF_INT8_FWD_OS1, {{"convolution2d_os1.ddl"}}},
    {OpFuncs::DEPTHWISE_CONV_FWD, {{"depthwise_conv_fwd.ddl"}}},
    {OpFuncs::ROPE64P1_FWD, {{"rope.ddl"}}},
    {OpFuncs::ROPE64P2_FWD, {{"rope.ddl"}}},
    {OpFuncs::SQRT_FWD, {{"unary_parallel.ddl"}}},
    {OpFuncs::RSQRT, {{"unary_parallel.ddl"}}},
    {OpFuncs::IDENTITY, {{"unary_parallel.ddl"}}},
    {OpFuncs::SHUFFLE, {{"unary_parallel.ddl"}}},
    {OpFuncs::AVGPOOL_NMAP_FWD, {{"pooling.ddl"}}},
    {OpFuncs::LSTMACTP2_FWD, {{"lstmactp2.ddl"}}},
    {OpFuncs::MISH_FWD, {{"unary_pipeline.ddl"}}},
    {OpFuncs::ABS, {{"unary_parallel.ddl"}}},
    {OpFuncs::NEG, {{"unary_parallel.ddl"}}},
    {OpFuncs::GREATEREQUAL, {{"broadcast_ops.ddl"}}},
    {OpFuncs::LESSEREQUAL, {{"broadcast_ops.ddl"}}},
    {OpFuncs::GREATERTHAN, {{"broadcast_ops.ddl"}}},
    {OpFuncs::LESSERTHAN, {{"broadcast_ops.ddl"}}},
    {OpFuncs::EQUAL, {{"broadcast_ops.ddl"}}},
    {OpFuncs::NOTEQUAL, {{"broadcast_ops.ddl"}}},
    {OpFuncs::DL16TOFP32, {{"quantization_double_pad.ddl"}}},
    {OpFuncs::FP32TODL16, {{"quantization_double_pad.ddl"}}},
    {OpFuncs::INTERSLICETRANSPOSE_FP16, {{"inter_slice_transpose.ddl"}}},
    {OpFuncs::INTERSLICETRANSPOSE_FP8, {{"inter_slice_transpose.ddl"}}},
    {OpFuncs::FP8TODL16, {{"quantization_double_pad.ddl"}}},
    {OpFuncs::DL16TOBF16, {{"quantization_no_pad.ddl"}}},
    {OpFuncs::TOPK_VALUE, {{"topk.ddl"}}},
    {OpFuncs::TOPK_INDEX, {{"topk.ddl"}}},
    {OpFuncs::MASK_BY_INDEX, {{"topk.ddl"}}},
    {OpFuncs::SOFTPLUS, {{"unary_pipeline.ddl"}}},
    {OpFuncs::ReStickifyOpLx,
     {{"restickify.ddl", RCUDD1A_ISA},
      {"restickify.ddl", MPW4_ISA},
      {"restickify_sen1p5.ddl", SEN1P5_ISA}}},
    {OpFuncs::ReStickifyOpHBM,
     {{"restickify.ddl", RCUDD1A_ISA},
      {"restickify.ddl", MPW4_ISA},
      {"restickify_sen1p5.ddl", SEN1P5_ISA}}},
    {OpFuncs::FLOOR, {{"unary_parallel.ddl"}}},
    {OpFuncs::INT32IDXTOADDR, {{"unary_parallel.ddl"}}},
    {OpFuncs::STZ_LATCH, {{"stz_latch.ddl"}}},
};

}  // namespace

/**
 * @struct DdlInterface
 * @brief Maintains bidirectional mappings between DDL MLIR values and DSC
 * entities
 *
 * This structure tracks all associations needed for DDL <-> DSC conversion:
 * - Dimension mappings (DDL dimensions to primary dimension types)
 * - Type definitions (data formats and bit sizes)
 * - Tensor properties (LDS indices)
 * - Operation properties (compute op indices, core/corelet conditions)
 * - Synchronization definitions
 * - Loop labels and control flow structures
 */
struct DdlInterface {
  /**
   * @struct DimProp
   * @brief Properties of a dimension in the DDL representation
   *
   * Tracks dimension type, padding status, and candidate dimension assignments.
   * Supports both primary dimensions (I, J, K, etc.) and meta-dimensions
   * (batch, channel, spatial dimensions).
   */
  struct DimProp {
    PrimaryDimTypes dim_ =
        PrimaryDimTypes::PrimaryDimTypesCount;  ///< Assigned primary dimension
                                                ///< (invalid if unset)
    std::set<PrimaryDimTypes>
        dimCandidates_;               ///< Possible dimension assignments
    int numRefsInGlobalLayouts_ = 0;  ///< Reference count in global layouts
    bool dropDim_ = false;  ///< True if dimension not needed for current DSC
    Value nonPaddedDim;  ///< Reference to unpadded version (if this is padded)

    DimProp() {
      // initialize candidate list with all dimensions (excluding IJ, KIJ)
      for (auto i = 0; i < PrimaryDimTypes::PrimaryDimTypesCount; i++) {
        if (!is_any_of(i, IJ, KIJ)) dimCandidates_.insert(PrimaryDimTypes(i));
      }
    }

    void setPrimaryDim(PrimaryDimTypes inDim) { dim_ = inDim; }
    PrimaryDimTypes getPrimaryDim() { return dim_; }

    bool setMetaDimKind(llvm::StringRef inDimKind) {
      if (!EnumsConversion::stringToMetaDimKind.count(inDimKind.str())) {
        return false;
      }
      metaDimKind_ = EnumsConversion::stringToMetaDimKind.at(inDimKind.str());
      return true;
    }

    void setMetaDimKind(MetaDimKind pDimKind) { metaDimKind_ = pDimKind; }
    MetaDimKind getMetaDimKind() const { return metaDimKind_; }
    bool isUnpadded() const { return metaDimKind_ == MetaDimKind::Unpadded; }
    bool isPadded() const { return metaDimKind_ == MetaDimKind::Padded; }
    bool isMetaDim() const {
      return metaDimKind_ != MetaDimKind::Unpadded &&
             metaDimKind_ != MetaDimKind::Padded &&
             metaDimKind_ != MetaDimKind::Count;
    }
    void dump(std::string msg = "") const;

   private:
    MetaDimKind metaDimKind_ =
        MetaDimKind::Unpadded;  ///< Dimension kind (unpadded, padded, or
                                ///< meta-dimension type)
  };

  std::unordered_map<Value, DimProp>
      dim_association_;  ///< Maps DDL dimension values to their properties

  /**
   * @brief Get dimension properties, resolving to unpadded version if needed
   * @param ddlDim DDL dimension value
   * @return Reference to dimension properties (unpadded version if input is
   * padded)
   */
  DimProp& getNonPaddedDimProp(Value ddlDim) {
    auto& dimProp = dim_association_[ddlDim];
    if (!dimProp.isPadded()) return dimProp;
    return dim_association_[dimProp.nonPaddedDim];
  }

  /**
   * @struct TypeDefinition
   * @brief Data type information for DDL values
   */
  struct TypeDefinition {
    DataFormats dataFormat_ =
        DataFormats::INVALID;  ///< Data format (FP32, INT8, etc.)
    int bitSize_ = -1;         ///< Bit width of the type
  };
  std::unordered_map<Value, TypeDefinition>
      type_definition_;  ///< Maps DDL values to type definitions

  /**
   * @struct TensorProp
   * @brief Tensor properties in DSC representation
   */
  struct TensorProp {
    int ldsIdx_ = -1;  ///< Labeled data structure index in DSC
  };
  std::unordered_map<Value, TensorProp>
      tensor_definition_;  ///< Maps DDL tensor values to properties
  const TensorProp& getTensorProp(Value tensorSSA);

  /**
   * @struct OperationProp
   * @brief Operation properties for compute operations
   */
  struct OperationProp {
    int computeOpIdx_ = -1;  ///< Compute operation index in DSC
    std::map<int, std::set<int>>
        coreClCond_;  ///< Core/corelet conditions (empty = all cores)
  };
  std::unordered_map<Value, OperationProp>
      operation_definition_;  ///< Maps DDL operation values to properties

  std::unordered_map<Value, int>
      datastage_definition_;  ///< Maps DDL values to datastage keys

  std::unordered_map<Value, int>
      ext_constant_definition_;  ///< Maps DDL values to constant info indices

  std::unordered_map<Value, dsc2::AllocateNode*>
      alloc_storage_;  ///< Maps DDL allocation SSA values to DSC allocate nodes

  /// Record SSA values in access_pattern_dim for DSC2DDL conversion
  std::map<const dsc2::TransferNode*, std::vector<mlir::Value>>
      transfer_acc_pat_dims_;

  std::unordered_map<Value, SenComponents>
      operand_constant_tensor_;  ///< Maps OperandConstantOp values to
                                 ///< components

  std::unordered_map<std::string, dsc2::LoopNode*>
      loop_labels_;                    ///< Maps loop labels to DSC loop nodes
  std::string core_chunk_loop_label_;  ///< Label for core chunk loop
  std::unordered_map<mlir::Region*, dsc2::BlockNode*>
      region2blocks_;  ///< Maps MLIR regions to DSC block nodes

  /**
   * @struct CondProp
   * @brief Properties of conditional expressions
   */
  struct CondProp {
    bool resolvedValue_, isResolvedToBool_ = false;  ///< Resolved boolean value
    dsc2::LoopCondComposite loopCond_;  ///< Loop condition composite
    std::map<int, std::set<int>>
        coreClCond_;  ///< Core/corelet conditions for "then" region
  };
  std::unordered_map<Value, CondProp>
      resolvedConditions_;  ///< Maps condition values to properties

  /**
   * @struct SyncProp
   * @brief Synchronization properties
   */
  struct SyncProp {
    struct SendRecv {
      std::vector<dsc2::SyncNode*> senders_,
          receivers_;  ///< Sender and receiver sync nodes
    };
    std::map<int, SendRecv> syncsPerCl_;  ///< Synchronization per corelet
    bool separateCorelets_ = false;       ///< Whether corelets are separated
  };
  std::unordered_map<std::string, SyncProp>
      sync_definitions_;  ///< Maps sync labels to properties

  /**
   * @struct CoreToCore
   * @brief Core-to-core communication properties
   */
  struct CoreToCore {
    PrimaryDimTypes dim_ = PrimaryDimTypesCount;  ///< Communication dimension
    FoldManager<int64_t> nextCore_,
        prevCore_;  ///< Next and previous core indices
  };
  std::unordered_map<mlir::Operation*, CoreToCore>
      coreToCore_definitions_;  ///< Maps operations to core-to-core properties

  /**
   * @brief Clear all interface data and reinitialize
   */
  void clear() {
    this->~DdlInterface();
    new (this) DdlInterface();
  }
};

/**
 * @class DdlConversion
 * @brief Main conversion class for bidirectional DDL<->DSC translation
 *
 * Handles:
 * - DDL template selection based on operation type and architecture
 * - Parsing DDL MLIR into DSC internal representation
 * - Converting DSC back to DDL MLIR
 * - Dimension inference and type propagation
 * - Access pattern processing
 * - Control flow and synchronization handling
 *
 * Conversion Flow:
 * 1. selectAndParseDdlTemplate() - Select appropriate DDL template
 * 2. parseDdl2Dsc() - Parse DDL MLIR and build DSC structures
 * 3. matchDdl2Dsc() - Match and validate DDL against DSC
 * 4. convertDsc2Ddl() - Convert DSC back to DDL (reverse direction)
 */
class DdlConversion {
  bool matchDdl2Dsc();
  void parseDdl2Dsc();
  void processRegion(::mlir::Region& myRegion, dsc2::BlockNode* currParent);
  void processTransformations(::mlir::Region& myRegion);
  void processPaddedDimensionOp(const mlir::Value& dimVal);
  DdlInterface::DimProp& processDimensionOp(const mlir::Value& dimVal);
  void checkMetaDimensions();

  template <typename T>
  bool checkAccessPattern(T& op, std::string dimArgName,
                          mlir::Operation::operand_range dims,
                          std::string accPatternAttrName,
                          std::optional<mlir::ArrayAttr> opAccessPatternStyles);

  template <typename OpT, typename AccPatT>
  void processAccessPatterns(OpT& op, mlir::Operation::operand_range dims,
                             mlir::ArrayAttr opAccessPatternStyles,
                             std::map<PrimaryDimTypes, AccPatT>& result);

  const DdlInterface::CondProp& processCondition(mlir::Value cond);
  std::pair<dsc2::BlockNode*, std::vector<int>> processOp(
      Operation& op, dsc2::BlockNode* currParent);
  float processExpression(std::string expr);
  LabeledDsInfo& addInternalTensor(const LabeledDsInfo& refLds,
                                   int computeOpIdx);
  std::vector<const DdlInterface::TypeDefinition*> processTypes(
      mlir::OperandRange supportedTypes, mlir::Operation* userOp);

  DesignSpaceConfig& dsc;  ///< Target DSC being built/modified
  const SuperDsc& sdsc;    ///< Source SuperDSC for context
  const DesignSpaceConfigGlobal& dscGlobal_;  ///< Global DSC configuration
  DdlInterface ddlInterface;                  ///< DDL↔DSC mapping interface
  DdlModuleOp ddlParser_;                     ///< DDL MLIR parser and context
  Metadata& metadata_;                        ///< Conversion metadata
  int verbose_;                               ///< Verbosity level for logging

 public:
  Value getTensor(SenComponents unit, const dsc2::DataInfo dtinfo) const;
  std::pair<Value, Value> getTensorAndAllocation(
      llvm::DenseMap<AllocateOp, const dsc2::AllocateNode*>& allocations,
      SenComponents comp, const dsc2::DataInfo dtinfo) const;

  /**
   * @brief Convert DSC to DDL MLIR representation
   * @param outputDdl Output stream for generated DDL
   */
  void convertDsc2Ddl(std::ostream& outputDdl);

  /**
   * @brief Select and parse appropriate DDL template based on operation and
   * architecture
   * @return true if template successfully selected and parsed
   */
  bool selectAndParseDdlTemplate();

  /**
   * @brief Verify DDL constraints and consistency
   */
  void verifyDdlConstraints();

  /**
   * @brief Construct DdlConversion instance
   * @param dscGlobal Global DSC configuration
   * @param mySdsc Source SuperDSC
   * @param myDsc Target DSC to build/modify
   * @param metadata Conversion metadata
   * @param verbose Verbosity level (0=quiet, higher=more verbose)
   */
  DdlConversion(const DesignSpaceConfigGlobal& dscGlobal,
                const SuperDsc& mySdsc, DesignSpaceConfig& myDsc,
                Metadata& metadata, int verbose)
      : dscGlobal_(dscGlobal),
        sdsc(mySdsc),
        dsc(myDsc),
        metadata_(metadata),
        verbose_(verbose) {}
};

}  // namespace ddc
#endif
