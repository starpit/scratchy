/************************************************************
 * IBM Confidential
 * (C) Copyright IBM Corp. 2023, 2025
 ************************************************************/

#ifndef L3_DLOPS_SCHEDULER_
#define L3_DLOPS_SCHEDULER_

#include <dsc/designSpaceConfig.h>
#include <dsc/dims.h>
#include <dsc/dsc2.h>
#include <dsc/superdsc.h>
#include <sys-arch-spec/memtracker/mem_track_bundle.h>

#include <algorithm>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "util/dt_exception.hpp"

class CrossCoreReductionGroup {
 public:
  using GroupType = std::vector<int>;
  GroupType coreIds;

  void addCore(const int coreId, const int slice) {
    coreIds.resize(slice + 1, -1);
    coreIds.at(slice) = coreId;
  }
  GroupType getCores() const { return coreIds; }
  int getStartCoreAtCorelet(int coreletId) const {
    DT_CHECK(!coreIds.empty());
    if (coreletId == 0)
      return coreIds.front();
    else if (coreletId == 1)
      return coreIds.back();
    else
      DT_ERROR("Unknown corelet id.");
  }
  int getEndCoreAtCorelet(int coreletId) const {
    DT_CHECK(!coreIds.empty());
    if (coreletId == 0)
      return coreIds.back();
    else if (coreletId == 1)
      return coreIds.front();
    else
      DT_ERROR("Unknown corelet id.");
  }
  bool isEmpty() const { return coreIds.empty(); }
};

class L3DlOpsScheduler {
 public:
  enum class LxBufferTypeMode {
    AUTO,              // Automatically determine based on heuristics
    FORCE_DOUBLE,      // Force DOUBLE buffering
    FORCE_SPATIAL_DOUBLE  // Force SPATIAL_DOUBLE buffering
  };

  L3DlOpsScheduler(const DesignSpaceConfigGlobal &dscGlobal,
                   MemTrackBundle *memTrackers, std::vector<int> phases,
                   int verbose = 0,
                   const LxBufferTypeMode lxBufferTypeMode = LxBufferTypeMode::AUTO)
      : dscGlobal(dscGlobal),
        memTrackers(memTrackers),
        exphases(std::move(phases)),
        verbose(verbose),
        lxBufferTypeMode(lxBufferTypeMode) {
    DT_CHECK(!exphases.empty());
  }
  void run(SuperDsc &mySDsc);

  // Memory trackers.
  //  - key: coreid
  const std::vector<int> exphases;
  MemTrackBundle *memTrackers;

  static inline bool isDimensionCoreletSplit(const DesignSpaceConfig &dsc,
                                             PrimaryDimTypes dim);

  // Initializes memory tracker with lx allocations done by upstream.
  void initLxMemtracker(SuperDsc &sdsc, MemTrackBundle *memTrackers);

 private:
  enum ScheduleDimTypes : unsigned {
    ELEMENTWISE = 0,
    BROADCAST,
    REDUCTION,
    WINDOW_PADDED,
    REUSE,
    ScheduleDimTypesCount
  };

  using ScheduleDimMapType =
      std::map<ScheduleDimTypes, std::vector<PrimaryDimTypes>>;
  using ScheduleDimTableType = std::unordered_map<int, ScheduleDimMapType>;
  using DscParamCandidatesType =
      std::vector<std::unordered_map<PrimaryDimTypes, std::vector<long>>>;
  using DscParamCandidateIndicesType =
      std::vector<std::unordered_map<PrimaryDimTypes, unsigned>>;

  enum BufferType {
    DOUBLE,
    SPATIAL_DOUBLE,
    BUFFER_TYPE_COUNT
  };

  struct Metadata {
    struct Datastage {
      struct Constraints {
        bool mustBeMultiple_ = false;
        std::optional<float> min_, max_;
        std::optional<std::set<float>> values_;
        inline void updateMin(float newVal) {
          min_ = min_ ? std::max(*min_, newVal) : newVal;
        }
        inline void updateMax(float newVal) {
          max_ = max_ ? std::min(*max_, newVal) : newVal;
        }
        inline void updateValues(std::set<float> newVals) {
          values_ = values_ ? set_intersect(*values_, newVals) : newVals;
        }
      };
      // int key is reference datastage, -1 for absolute constraints
      // set key contains the dimensions the constraints apply to
      std::unordered_map<int, std::map<std::set<PrimaryDimTypes>, Constraints>>
          constraints_;
      bool strategyMinimize_ = true;  // false == maximize
      std::unordered_map<PrimaryDimTypes, int> relevantDimsAndNumerator_;
      int nearestNumeratorIdx_ = -1;
    };
    std::unordered_map<int, Datastage> datastages_;

    typedef std::pair<PadType, PadType> TransferAccessPatternType;
    typedef std::map<PrimaryDimTypes, TransferAccessPatternType>
        TransferAccessPatternPerDimType;

    struct DataTransfer {
     public:
      bool apply_row_offset_ = false;
      int force_num_elements_ = -1;
      void setAccessPattern(PrimaryDimTypes dimVal,
                            TransferAccessPatternType accessPattern) {
        accessPatternPerDim_[dimVal] = accessPattern;
      }
      TransferAccessPatternType getAccessPattern(PrimaryDimTypes dimVal);
      std::string getAccessPatternAsStr(PrimaryDimTypes dimVal);
      TransferAccessPatternPerDimType &getMutableAccessPatternList() {
        return accessPatternPerDim_;
      };
      void dump();

     private:
      TransferAccessPatternPerDimType accessPatternPerDim_;
    };
    std::unordered_map<const dsc2::TransferNode *, DataTransfer> datatransfers_;

    struct Allocation {
      std::map<int, dsc2::AllocateNode *> ldsIdxAndAllocNode;
      std::map<int, dsc2::AllocateNode *> consIdAndAllocNode;
      std::unordered_map<dsc2::ComputeNode *, dsc2::AllocateNode *>
          compAndAllocNode;
    };
    std::unordered_map<SenComponents, Allocation> newAllocations_;

    struct ExternalTransfer {
      std::unique_ptr<dsc2::TransferNode> transfer_;
      std::unique_ptr<dsc2::AllocateNode> allocate_;
      ExternalTransfer(dsc2::TransferNode *transferNode,
                       dsc2::AllocateNode *allocateNode)
          : transfer_(transferNode), allocate_(allocateNode) {}
    };
    std::vector<ExternalTransfer> externalTransfers_;
    std::set<const dsc2::ScheduleNode *> externalNodes_;

    struct DataConnect {
      std::unordered_set<const dsc2::LoopNode *> producers_;
      std::unordered_set<const dsc2::LoopNode *> consumers_;
    };
    std::unordered_map<std::string, DataConnect> dataConnects_;

    struct OpaqueOp {
      std::unordered_map<std::string, dsc2::AllocateNode *> inOutRegAllocs_;
      std::vector<std::string> internalRegs_;
      dsc2::AllocateNode *internalRegAlloc_ = nullptr;
      int max_unroll_ = 1;
    };
    std::unordered_map<dsc2::ComputeNode *, OpaqueOp> opaqueOps_;

    int core_dstgid = -1;
    int chunk_dstgid = -1;
    PrimaryDimTypes rowSplitDim = PrimaryDimTypesCount;

    void clear() { *this = {}; }  // reinitialize structure
  };

  const DesignSpaceConfigGlobal &dscGlobal;
  // DSC to metadata map.
  // TODO: For now, the dscMetadata is only used for memory allocation.
  // But it can stores more information than that. We may explore more
  // use cases when needed.
  std::unordered_map<int, Metadata> dscMetadata;

  // Print debug information.
  int verbose;

  static const int dataStageCoreIdx;
  static const int dataStageChunkIdx;
  static const std::string lxBelowBlockNodeName;
  static const std::vector<std::vector<double>> burstEfficiency;
  size_t gtrCurrGroupName = 0;
  // A unique set of cores to its groupName map.
  // It is a data structure that records information for computing the
  // gtr_->groupName_ and gtr_->shares. gtr_->groupName is a unique id for each
  // sharing group. Each unique sharing group contains the same set of cores.
  std::map<std::set<int>, size_t> coresSetToGtrGroupNameMap;
  const LxBufferTypeMode lxBufferTypeMode;
  BufferType lxBufferType = BufferType::DOUBLE;
  int dataStageIbrIdx = -1;
  int dataStageOnePageIdx = -1;
  int dataStageSuperChunkIdx = -1;

  std::string scheduleDimTypeToString(ScheduleDimTypes type);
  bool isValidDimParam(const double param) { return param > 0.0; }
  bool isOutputLabeledDs(const int ldsIdx, const DesignSpaceConfig &dsc) const {
    return (ldsIdx == dsc.labeledDs_.size() - 1);
  }
  bool hasDimensionReuse(const DesignSpaceConfig &dsc);
  int getStickSize(const DesignSpaceConfig &dsc, DsTypes dsType,
                   PrimaryDimTypes dim);
  std::unordered_set<PrimaryDimTypes> getCoreSplitDimensions(
      const SuperDsc &mySDsc) const;
  void getLabeledDsWithDsType(std::vector<int> &indices, DesignSpaceConfig &dsc,
                              DsTypes dsType);
  bool isLabeledDsLXNeighbor(const SuperDsc &mySDsc, const int dscIndex,
                             const LabeledDsInfo &lds) const;
  std::unordered_set<int> getAllLabeledDsIndicesSet(
      const DesignSpaceConfig &dsc) const;
  std::unordered_set<int> getHbmPinnedLabeledDsIndicesSet(
      const DesignSpaceConfig &dsc) const;
  std::unordered_set<int> getLxNeighborLabeledDsIndicesSet(
      const SuperDsc &mySDsc, const DesignSpaceConfig &dsc,
      const int dscIdx) const;
  dsc2::BlockNode *computeLdsAllocateSiblingLoopNode(
      const SuperDsc &mySDsc, const int dscIdx, const LabeledDsInfo &lds,
      const dsc2::BlockNode *startNode,
      const std::vector<const dsc2::LoopNode *> &parentInnerToOuterLoopNodes)
      const;
  dsc2::BlockNode* computeLdsTransferSiblingLoopNode(
      const SuperDsc& mySDsc, const int dscIdx, const LabeledDsInfo& lds,
      const dsc2::BlockNode* startNode,
      const std::vector<const dsc2::LoopNode*>& parentInnerToOuterLoopNodes)
      const;
  std::vector<const dsc2::LoopNode*> getParentLoopNodes(
      const dsc2::ScheduleNode& node, const DesignSpaceConfig& dsc) const;

  dsc2::AllocateNode *createAllocateNode(
      DesignSpaceConfig &dsc, const int ldsIdx, enum SenComponents component,
      const int numBuffers, const std::string &name, const int dscIdx);
  dsc2::TransferNode *createTransferNode(
      const SenComponents srcUnit, const SenComponents srcStorage,
      const std::vector<SenComponents> &dstUnits,
      const std::vector<SenComponents> &dstStorage, const int srcLdsIndex,
      const std::vector<int> &dstLdsIndices, const std::string &name);
  dsc2::LoopNode *createLoopNode(const DesignSpaceConfig &dsc,
                                 const std::vector<PrimaryDimTypes> &dims,
                                 const int numeratorId, const int denominatorId,
                                 const std::string &name);
  dsc2::BlockNode* createBlockNode(const std::string& name);
  dsc2::SyncNode* createSyncNode(const std::unordered_set<SenComponents>& units,
                                 const std::string& name,
                                 const bool isReceive = false,
                                 const bool isSoft = false) const;

  OpFuncs getOpFuncName(const DesignSpaceConfig &dsc) const;
  std::string getOpFuncDataFormat(const DesignSpaceConfig &dsc);
  void addOrUpdateDataStageParam(DesignSpaceConfig &dsc,
                                 const DataStructDims &ssParam,
                                 const std::string &ssName,
                                 const DataStructDims &elParam,
                                 const std::string &elName, const int index);
  bool hasComputeOp(const DesignSpaceConfig &dsc) const {
    return !dsc.computeOp_.empty();
  }
  bool isOpFuncConv2dInt4(const OpFuncs opFuncName) const;
  bool isOpFuncConv2dOs1(const OpFuncs opFuncName) const;
  bool isOpFuncConv2d(const OpFuncs opFuncName) const;
  bool isOpFuncBmmInt4(const OpFuncs opFuncName) const;
  bool isOpFuncBmmInt8(const OpFuncs opFuncName) const;
  bool isOpFuncBmmFp8NonXrf(const OpFuncs opFuncName) const;
  bool isOpFuncBmmFp8Xrf(const OpFuncs opFuncName) const;
  bool isOpFuncBmmFp16(const OpFuncs opFuncName) const;
  bool isOpFuncBmm(const OpFuncs opFuncName) const;
  bool isOpFuncScalarBroadcast(const OpFuncs opFuncName) const;
  bool isOpFuncReduction(const OpFuncs opFuncName) const;
  bool isOpFuncPooling(const OpFuncs opFuncName) const;
  bool isOpFuncDepthwiseConv(const OpFuncs opFuncName) const;
  bool isOpFuncQuantization(const OpFuncs opFuncName) const;
  bool isOpFuncConversionDl16AndFp32(const OpFuncs opFuncName) const;
  bool isOpFuncStridedWindow(const OpFuncs opFuncName) const;
  long getMinParamForDim(const SuperDsc &mySDsc, const DesignSpaceConfig &dsc,
                         PrimaryDimTypes dim) const;
  long getMinParamForDimFromOpFunc(const DesignSpaceConfig &dsc,
                                   PrimaryDimTypes dim) const;
  long computeMinParamForPaddedDim(const DesignSpaceConfig &dsc,
                                   const PrimaryDimTypes dim) const;
  long getMinParamConv2d(const DesignSpaceConfig &dsc,
                         const PrimaryDimTypes dim,
                         const OpFuncs opFuncName) const;
  long getMinParamBmm(const DesignSpaceConfig &dsc, const PrimaryDimTypes dim,
                      const OpFuncs opFuncName) const;
  long getMinParamScalarBroadcast(const DesignSpaceConfig &dsc,
                                  const PrimaryDimTypes dim) const;
  long getMinParamReduction(const DesignSpaceConfig &dsc,
                            const PrimaryDimTypes dim) const;
  long getMinParamPoolingAndDepthwiseConv(const DesignSpaceConfig &dsc,
                                          const PrimaryDimTypes dim,
                                          const OpFuncs opFuncName) const;
  long getMinParamQuantization(const DesignSpaceConfig &dsc,
                               const PrimaryDimTypes dim,
                               const OpFuncs opFuncName) const;
  long getMinParamConversionDl16AndFp32(const DesignSpaceConfig &dsc,
                                        const PrimaryDimTypes dim,
                                        const OpFuncs opFuncName) const;
  std::vector<std::unordered_map<PrimaryDimTypes, std::vector<long>>>
  generateDscParamCandidates(
      const SuperDsc &mySDsc, const std::vector<DataStructDims> &dscParams,
      const std::vector<PrimaryDimTypes> &primaryDims,
      const std::unordered_set<PrimaryDimTypes> &chunkDims,
      const std::unordered_set<PrimaryDimTypes> &coreSplitDims);
  void addChunkDataStageFromCandidates(
      DataStructDims &chunkParams, DesignSpaceConfig &dsc, const int dscIdx,
      const DscParamCandidateIndicesType &selectedIndices,
      const DscParamCandidatesType &dscCandidates,
      const std::vector<PrimaryDimTypes> &primaryDims);
  void getChunkParamsFromCandidates(
      DataStructDims &params,
      const DscParamCandidateIndicesType &selectedIndices,
      const DscParamCandidatesType &dscCandidates, const int dscIdx,
      const std::vector<PrimaryDimTypes> &primaryDims);
  double getBurstEfficiency(const unsigned burstNum,
                            const unsigned multicastDegree);
  unsigned long getLabeledDsChunkStickVolume(const DesignSpaceConfig &dsc,
                                             const int ldsIdx);
  unsigned long getLabeledDsNumOfStickVolumesInCore(
      const DesignSpaceConfig &dsc, const int ldsIdx,
      const unsigned long stickVolume,
      const std::vector<PrimaryDimTypes> &primaryDims);
  double calculateBurstEfficiency(
      const SuperDsc &mySDsc, const std::vector<PrimaryDimTypes> &primaryDims);
  void findBestParamsForMemoryBandwidth(
      DscParamCandidateIndicesType &dscCandidateIndices,
      const DscParamCandidatesType &dscCandidates, SuperDsc &mySDsc,
      const std::vector<PrimaryDimTypes> &primaryDims,
      const std::unordered_set<PrimaryDimTypes> &coreSplitDims,
      const bool isInputNeighborFetch);
  double calculateFlopPerByte(const SuperDsc &mySDsc,
                              const std::vector<PrimaryDimTypes> &primaryDims);
  void findBestParamsForArithmeticIntensity(
      DscParamCandidateIndicesType &dscCandidateIndices,
      const DscParamCandidatesType &dscCandidates, SuperDsc &mySDsc,
      const std::vector<PrimaryDimTypes> &primaryDims,
      const std::unordered_set<PrimaryDimTypes> &coreSplitDims);
  DataStructDims getInitialChunkParams(
      const SuperDsc &mySDsc, const DesignSpaceConfig &dsc,
      const std::unordered_set<PrimaryDimTypes> &chunkDims);
  void setChunkDataStageParams(SuperDsc &mySDsc);
  void createChunkLoops(SuperDsc &mySDsc);
  std::vector<PrimaryDimTypes> collectAllDimensionsForLoopOrder(
      const DesignSpaceConfig &dsc);
  ScheduleDimTableType buildScheduleDimensionsTable(
      SuperDsc &mySDsc, const int dscIdx, std::vector<PrimaryDimTypes> &dims,
      const bool isReuse);
  std::vector<PrimaryDimTypes> buildLoopOrder(
      SuperDsc &mySDsc, const int dscIdx,
      const std::vector<PrimaryDimTypes> &dims,
      const ScheduleDimTableType &schedDimTypesTable, const bool isReuse);
  void createChunkLoopNodes(SuperDsc &mySDsc,
                            const std::vector<PrimaryDimTypes> &loopOrder);
  std::set<PrimaryDimTypes> getOpReducedDimSet(
      const SuperDsc &mySDsc, const DesignSpaceConfig &dsc) const;
  bool isOpCrossCoreReduction(const SuperDsc &mySDsc,
                              const DesignSpaceConfig &dsc) const;
  std::vector<int> getLdsTransferCoreIds(const SuperDsc& mySDsc,
                                         const DesignSpaceConfig& dsc,
                                         const LabeledDsInfo& lds) const;
  void setSuperChunkDataStageParams(SuperDsc& mySDsc);
  void updateChunkDataStagesFromCandidates(
      DataStructDims &chunkParams, DesignSpaceConfig &dsc, const int dscIdx,
      const DscParamCandidateIndicesType &selectedIndices,
      const DscParamCandidatesType &dscCandidates,
      const std::vector<PrimaryDimTypes> &primaryDims);
  void exploreSuperChunkDataStageParams(SuperDsc& mySDsc, const int dscIdx);
  void addSuperChunkDataStage(DesignSpaceConfig& dsc);
  void createAllocationAndTransfer(SuperDsc& mySDsc);
  void createSynchronization(SuperDsc& mySDsc);
  void createSynchronizationDSC(SuperDsc& mySDsc, const int dscIdx);
  void addL3LUAndLXLUSyncNodeSequence(
      const dsc2::ScheduleNode* siblingRefNode) const;
  void addL3LUAndLXLUSoftSyncNodeSequence(
      const dsc2::ScheduleNode* siblingRefNode) const;
  void optimizeHbmLdsOutputInScheduleTree(SuperDsc& mySDsc);
  void optimizeHbmTransfers(SuperDsc& mySDsc);
  std::pair<size_t, size_t> getSharesAndGroupName(
      const SuperDsc &mySDsc, const DesignSpaceConfig &dsc,
      const LabeledDsInfo &lds,
      const std::map<PrimaryDimTypes, int> &currWkSlices,
      const std::vector<int> &processingCoreIds);
  void setCondGtr(
      SuperDsc &mySDsc, const int dscIdx, const int ldsIdx, const int coreId,
      dsc2::TransferNode &ldsL3LUTransNode,
      const std::vector<std::tuple<const dsc2::LoopNode *, PrimaryDimTypes, int,
                                   int>> &loopDimTripCounts);
  std::vector<int64_t> calculateCoreletOffsetInByte(
      const DesignSpaceConfig &dsc, dsc2::AllocateNode *allocNode) const;
  std::pair<int64_t, int64_t> getInitialStartAddressAndOffset(
      DesignSpaceConfig &dsc, const int ldsIdx,
      std::deque<int64_t> coord) const;
  void fillFinalStartAddressAndOffset(
      DesignSpaceConfig &dsc, const int ldsIdx,
      const std::vector<PrimaryDimTypes> &coreletSplitDims) const;
  void fillIBRStartAddressAndOffset(const SuperDsc &sdsc,
                                    DesignSpaceConfig &dsc,
                                    const int ldsIdx) const;
  void fillTransferMulticastInfo(SuperDsc &mySDsc);
  void fillAllocationStartAddrAndOffset(SuperDsc &mySDsc) const;
  void fillTransferZeroPaddingInfo(SuperDsc &mySDsc);
  // TODO: Clean up the helper functions below for support on HBM paged tensors.
  bool isIndexLds(const LabeledDsInfo &lds) const;
  bool isPagedLds(const LabeledDsInfo &lds) const;
  std::vector<PrimaryDimTypes> getPagedDimensions(
      const DesignSpaceConfig &dsc) const;
  std::vector<int> getAllPagedLdsIndices(const DesignSpaceConfig &dsc) const;
  void addIbrDataStage(SuperDsc& mySDsc, DesignSpaceConfig& dsc,
                       const std::vector<PrimaryDimTypes>& pagedDims);
  void addOnePageDataStage(SuperDsc& mySDsc, DesignSpaceConfig& dsc,
                           const std::vector<PrimaryDimTypes>& pagedDims);
  int getNewDataStageIndex(SuperDsc &mySDsc, DesignSpaceConfig &dsc) const;
  void processHbmPagedTensors(SuperDsc &mySDsc);
  void processDscHbmPagedTensors(SuperDsc &mySDsc, const int dscIdx);
  std::unordered_map<PrimaryDimTypes, dsc2::LoopNode *>
  createPagedDimChunkLoops(DesignSpaceConfig &dsc,
                           const std::vector<PrimaryDimTypes> &pagedDims,
                           std::set<dsc2::LoopNode *> &chunkLoopNodes);
  void processPagedTensorTransfers(
      SuperDsc &mySDsc, const int dscIdx,
      const std::vector<dsc2::TransferNode *> &l3TransferNodesAllTensors,
      const std::vector<dsc2::AllocateNode *> &pagedLdsHbmAllocNodes,
      const std::unordered_map<PrimaryDimTypes, dsc2::LoopNode *>
          &newPagedDimChunkLoopNodes,
      PrimaryDimTypes innerIndexDimInChunkLoops);
  void createStoreIndexTensorToLx(SuperDsc &mySDsc, const int dscIdx,
                                  const int pagedLdsIdx, const int indexLdsIdx,
                                  dsc2::AllocateNode &indexLdsHbmAllocNode,
                                  dsc2::LoopNode &newChunkLoopNode);
  void createStoreIndexTensorToIbr(DesignSpaceConfig &dsc, const int dscIdx,
                                   const int indexLdsIdx,
                                   dsc2::AllocateNode &indexLdsHbmAllocNode,
                                   dsc2::LoopNode &newChunkLoopNode,
                                   const bool isTransferIn);
  void convertTransferDirectToIndirect(dsc2::TransferNode &transNode,
                                       DesignSpaceConfig &dsc,
                                       const int indexLdsIdx,
                                       const PrimaryDimTypes indexStickDim,
                                       const bool isTransferIn);
  std::vector<dsc2::AllocateNode *> getHbmAllocations(
      const DesignSpaceConfig &dsc) const;
  void buildCoordinateForAllocation(
      SuperDsc &mySDsc, DesignSpaceConfig &dsc, dsc2::AllocateNode *allocNode,
      dsc2::CoordPropInfoType &coordPropInfo) const;
  void buildCoordinateFromAllocation(
      DesignSpaceConfig &dsc, dsc2::AllocateNode *refAllocNode,
      dsc2::ScheduleNode *node,
      dsc2::CoordinateType<CoordinateBaseType> &coordinate) const;
  void fillCoordinateCustomWkSliceId(
      SuperDsc &mySDsc, DesignSpaceConfig &dsc, const int ldsIdx,
      dsc2::CoordinateType<CoordinateBaseType> &coordinate) const;
  void propagateCoordinate(SuperDsc &mySDsc) const;
  void propagateCoordinateDSC(SuperDsc &mySDsc, DesignSpaceConfig &dsc) const;
  dsc2::ScheduleNode *getInsertionNode(
      const std::unordered_set<dsc2::ScheduleNode *> &nodes,
      const bool insertBefore = false) const;
  dsc2::BlockNode *getLxBelowBlockNode(dsc2::ScheduleTree &scheduleTree) const;
  int getTripCount(const DesignSpaceConfig &dsc, const PrimaryDimTypes dim,
                   const int dataStageNumId, const int dataStageDenId) const {
    auto num =
        dsc.dataStageParam_.at(dataStageNumId).ss_.primaryDimToVal_st(dim);
    auto den =
        dsc.dataStageParam_.at(dataStageDenId).ss_.primaryDimToVal_st(dim);
    int tripCount =
        std::ceil(static_cast<double>(num) / static_cast<double>(den));
    return tripCount;
  }
  std::vector<CrossCoreReductionGroup> getCrossCoreReductionGroupInfo(
      const SuperDsc &mySDsc, const DesignSpaceConfig &dsc) const;
  std::vector<const dsc2::TransferNode *> getLdsL3TransferNodes(
      const SuperDsc &mySDsc, const int dscIdx, const int ldsIdx,
      const std::vector<SenComponents> &srcStorages,
      const std::vector<SenComponents> &dstStorages) const;
  void fillExplicitTransferSize(SuperDsc &mySDsc) const;
  bool verifyLoopOrder(std::vector<PrimaryDimTypes> &loopOrder);
  bool verifyScheduleTree(const DesignSpaceConfig &dsc);
  void prepDsc(SuperDsc &mySDsc);
  void setLxBufferType(const SuperDsc &mySDsc);
  int computeMinHMICoreGroupSizeForSEN1P5(
      const SuperDsc &mySDsc, const std::vector<int> &hbmLdsIndices) const;
  int getHbmLdsTransferHMIRequestEstimate(const SuperDsc &mySDsc) const;

  // FIXME: The following function are copies (might be slightly modified) from
  // DDC. Make them into shared utility functions.
  std::string getLdsOrConstNameOfAllocNode(DesignSpaceConfig *currDsc,
                                           dsc2::AllocateNode *anode);
  bool allocAllMem(const SuperDsc &mySDsc, DesignSpaceConfig *currDsc,
                   const int dscIdx, bool commitIfValid);
  void fillLoopOffsetsAndAddresses(
      SuperDsc &mySDsc, const int dscIdx,
      const bool allowUnpaddedIndexingAtPaddedNoZeroPad = false);
  void gatherFoldParams(const FoldManager<CoordinateBaseType> &cfm,
                        std::vector<dsc2::FoldParamInfoType> &foldParams) const;
  void getEnclosingLoopsAndRelatedDims(
      dsc2::ScheduleNode *node, const DesignSpaceConfig *dsc,
      std::vector<dsc2::LoopNode *> &loopChain,
      std::unordered_set<PrimaryDimAndKind> &relatedDims) const;
  void findAndStoreLoopWithDim(
      DesignSpaceConfig *currDsc, const PrimaryDimAndKind dimToFind,
      dsc2::LoopNode *loop,
      const std::unordered_set<PrimaryDimAndKind> &relatedDims,
      dsc2::VectorOfLoopAndDim &relatedLoops, PadType accessPadType) const;
  void sliceCoordinateForCorelet(SuperDsc &mySDsc, DesignSpaceConfig *currDsc,
                                 dsc2::AllocateNode *allocNode) const;
  int constructDatastage(DesignSpaceConfig *currDsc,
                         dsc2::DataStage &refDataStage) const;
  dsc2::LoopNode *constructLoopNode(int numId, int denId,
                                    std::vector<PrimaryDimAndKind> dims) const;
  // END OF COPY
};

#endif  // L3_DLOPS_SCHEDULER_
