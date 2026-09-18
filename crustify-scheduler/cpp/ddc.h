/************************************************************
 * IBM Confidential
 * (C) Copyright IBM Corp. 2022, 2025
 ************************************************************/

/*
 * Description:
 *
 */

#ifndef DDC_H_
#define DDC_H_

#include <dsc/designSpaceConfig.h>
#include <util/foldManager/foldInfrastructure.h>
#include <dsc/superdsc.h>

#include <cstdlib>
#include <iostream>
#include <queue>
#include <stdexcept>
#include <unordered_set>
#include <vector>

#include "ddc_metadata.h"
#include "ddl/ddl_convert_interface.h"
#include "util/dtgetenv.hpp"

struct MemTrackBundle;
class DsTrackInMem;

// Deep Dataflow Constructor
namespace ddc {
class Ddc {
 public:
  static const std::unordered_set<SenComponents> registerComponents;
  const DesignSpaceConfigGlobal& dscGlobal;
  int verbose_;
  int latchDataIdCounter_ = 0;
  int transformationReportLevel_ = 0;
  int coordFoldReportLevel_ = 0;
  int coordPropReportLevel_ = 0;
  bool verifyCoordinateBasedLoopElemOff = false;
  bool datastageBasedElemOff = false;
  bool dscToDdl_ = false;
  bool trueLXTracker_ =
      false;  // this can be set to true when we pass a true LX memory tracker

  Ddc(const DesignSpaceConfigGlobal& dscGlobal, int verbose = 0,
      bool dscToDdl = false, int transformationReportLevel = 0,
      std::string coordOptionStr = "")
      : dscGlobal(dscGlobal),
        verbose_(verbose),
        dscToDdl_(dscToDdl),
        transformationReportLevel_(transformationReportLevel) {
    if (coordOptionStr.find("content_short") != std::string::npos) {
      coordFoldReportLevel_ = 1;
    } else if (coordOptionStr.find("content_med") != std::string::npos) {
      coordFoldReportLevel_ = 2;
    } else if (coordOptionStr.find("content_long") != std::string::npos) {
      coordFoldReportLevel_ = 3;
    }

    if (coordOptionStr.find("prop_short") != std::string::npos) {
      coordPropReportLevel_ = 1;
    } else if (coordOptionStr.find("prop_med") != std::string::npos) {
      coordPropReportLevel_ = 2;
    } else if (coordOptionStr.find("prop_long") != std::string::npos) {
      coordPropReportLevel_ = 3;
    }

    auto ddcCoordEnvOption = dtGetEnv<std::string>("DDCCOORD");
    if (ddcCoordEnvOption &&
        ddcCoordEnvOption->find("verify_loopelemoff") != std::string::npos) {
      verifyCoordinateBasedLoopElemOff = true;
    }
  }

  void run(SuperDsc& sdsc);

  bool run_v1(SuperDsc& sdsc);

  // Wraps a given transferNode with a new loop band. The denominator of each
  // loop in the band has strategy set to `minimize`.
  bool unrollTransfer(dsc2::TransferNode* transferNode);

  // Wraps a given transferNode with a new loop nest. Each loop in the nest
  // corresponds to a symbolic dimension. The denominator of each loop in the
  // nest takes the `granularity` of the corresponding symbolic dimension. The
  // order of the loops is determined by the transfer node's layout order.
  //
  // Returns
  //   - true, in case the caller can continue.
  //   - false, otherwise.
  bool unrollTransferForSymbolicDims(
      dsc2::TransferNode* transferNode,
      const std::map<PrimaryDimTypes, SymbolicDimInfo>& symbolicdims);

  // Memory trackers..
  // key: coreid, coreletid, (rowid)
  int exphase = -1;
  MemTrackBundle* memTrackers;

 private:
  Metadata metadata;
  SuperDsc* sdsc_ = nullptr;
  DesignSpaceConfig* currDsc = nullptr;
  std::unordered_set<const dsc2::LoopNode*> loopsBelowChunkBoundary;
  PrimaryDimTypes coreletSplitDim = PrimaryDimTypes::PrimaryDimTypesCount;

  std::string getLdsOrConstNameOfAllocNode(dsc2::AllocateNode*);
  bool dataStageExplorationDone_ = false;
  void calculateClStartAddress(dsc2::AllocateNode* allocNode);
  void prepDsc();
  void restoreDsc();
  bool allocAllMem(bool);
  void attachToPrefilledSchedule();
  void populateUnitTimeTransfers();
  void minimizeAllocations(bool has_auto_shuffling);
  void exploreAssignDataStages();
  void finalizeAllocateLayouts();
  void spreadDataInAllocate();
  void createDataConnectMetadata();
  void fillLoopOffsetsAndAddresses(
      const bool allowUnpaddedIndexingAtPaddedNoZeroPad = false);
  void adjustLoopOffsetsAndAddresses();
  void finalizeOps();
  void simplifyScheduleTree();
  void coordinateMasking();
  void updateLdsIdxMetadata(DesignSpaceConfig& dsc);
  void initGlobalData();
  void identifyBelowChunkBoundaryLoops();

  // Constructs a new AllocateNode and associates the node with a given
  // labeled datastage and storage. In case the allocated storage has
  // padding, the necessary padding information is also added to the
  // new AllocateNode.
  // Inputs:
  //   - Datainfo (refers to a labeled datastage)
  //   - DataLocation (refers to a storage)
  //   - Padding per dimension
  // Returns:
  //   - Pointer to the new AllocateNode, if successful.
  //   - Raises runtime error, otherwise.
  dsc2::AllocateNode* constructAllocation(const dsc2::DataInfo& di,
                                          const SenComponents storage,
                                          const PaddingFormType& paddingPerdDim,
                                          const dsc2::ScheduleNode* userNode);
  void reduceUsersOrDeleteAllocationAndMetadata(
      const dsc2::DataInfo& di, const SenComponents storage,
      const dsc2::ScheduleNode* userNode);
  PaddingFormType getPaddingPerDim(const dsc2::TransferNode* transferNode,
                                   bool forSrc) const;

  // Transformation related member functions

  // Constructs a new empty datastage.
  //
  // Returns the index of the new datastage in dataStageParam_ of the current
  // dsc.
  int constructDatastage();
  // Constructs a new datastage as a copy of a given reference datastage.
  // Inputs:
  //   - A reference datastage
  //
  // Returns the index of the new datastage in dataStageParam_ of the current
  // dsc.
  int constructDatastage(dsc2::DataStage& refDatastage);

  // Constructs a new LoopNode (non-parametric).
  // Inputs:
  //   - Id of the numerator datastage
  //   - Id of the denominator datastage
  //   - An ordered list of dimensions that are associated with the loop.
  //   - Parent ScheduleNode (optional)
  //
  // Returns a pointer to the newly constructed LoopNode. In case the parent is
  // not a nullptr, DDC metadata is updated using the parent as reference.
  //
  // Note: May invalidate iterators to parent's children.
  dsc2::LoopNode* constructLoopNode(int numId, int denId,
                                    std::vector<PrimaryDimAndKind> dims,
                                    dsc2::ScheduleNode* baseLoop = nullptr);

  // Splits a loop into two or more loops. After the splitting, all resulting
  // loops use the original datastages.
  //
  // Inputs:
  //   - The LoopNode to be split
  //   - One or more sets of dimension. Each set results in a separate loop.
  //     In case a set of dimensions is not specified, a new loop
  //     is construted with these remaining dimensions.
  //     The dimension sets must not have pairwise overlapping.
  //     The dimension sets must not include a dimension that is not associated
  //     with the base loop.
  //   - Indicates whether the loop corresponding to the unspecified dimensions
  //     become the innermost or the outermost loop.
  //
  // Returns a pointer to the innermost loop resulting from the loop-splitting.
  //
  // Note: All conditions inside the original loop are updated to reflect
  //     the distribution of the original dimensions to the new loops.
  dsc2::LoopNode* splitLoopBandOnDim(
      dsc2::LoopNode* baseLoop,
      const std::vector<std::vector<PrimaryDimAndKind>>&
          inputSplitDimSetsOuterToInner,
      bool unspecifiedDimsInnermost);

  // Splits a loop into two loops. Both loops are associated with the original
  // dimension list. A new datastage is used to rearrange the original
  // datastages as follows.
  // - The numerator DS of the original loop remains the same.
  // - The new DS becomes the denominator DS of the original loop.
  // - The new DS becomes the numerator DS of the new loop.
  // - The denominator DS of the original loop becomes the denominator DS of
  //   the new loop.
  // Inputs:
  //   - The LoopNode to be split
  //
  // Returns a pointer to the child loop resulting from the loop-splitting.
  //
  // Note: All conditions inside the original loop are updated to reflect
  //     the replication of the original dimensions to the new loops.
  dsc2::LoopNode* splitLoopBandOnDatastage(dsc2::LoopNode* baseLoop);

  // Moves a transfer node to a new parent loop.
  void moveTransferNode(dsc2::TransferNode* transferNode,
                        dsc2::LoopNode* newParentLoop);

  // Converts the result of a transfer from FIFO to register.
  // All consumers of the original FIFO result are updated to use the new
  // register result.
  // Inputs:
  //   - A transfer node possibly writing to a FIFO.
  //
  // Returns
  //   - true in case the transformation is performed,
  //   - false otherwise.
  bool convertResultFromFIFOtoReg(dsc2::TransferNode* transferNode);

  // Converts the result of a transfer from register to FIFO/latch. All
  // consumers of the original register-result are updated to use the FIFO/latch
  // instead. Inputs:
  //   - A transfer node writing to a register
  //   - Index of the transfer destination that needs to be converted to
  //   FIFO/latch
  //   - A flag to indicate whether the result should be converted to FIFO or
  //     latch
  //
  //  Returns
  //   - true in case the transformation is performed,
  //   - false otherwise (error condition).
  bool convertResultToSkipReg(dsc2::TransferNode* transferNode, int destIndex,
                              bool useLatch);
  void collectLoopReferences(const dsc2::ScheduleNode* node,
                             std::set<const dsc2::LoopNode*>& referenceLoops);

  // Inserts a compute node between the transfer and the register that the
  // transfer is writing to. This is achieved by changing the transfer from
  // writing from register to writing to fifo, while the compute will read from
  // fifo and write to register. Consumers of the original register-result do
  // not need to be updated.
  // Inputs:
  //   - A transfer node writing to a register
  //   - Index of the transfer destination that needs to be transformed
  //   - An owning pointer to a new compute node to be inserted. Fill compute
  //   type and other compute data, but do not insert in schedule tree, the
  //   function will take care of that
  //   - Index of the compute input that will read from the transfer
  //
  //  Returns
  //   - true in case the transformation is performed,
  //   - false otherwise (error condition).
  bool insertComputeBetweenTransferAndReg(dsc2::TransferNode* transferNode,
                                          int transferDestIndex,
                                          dsc2::ComputeNode* computeNode,
                                          int computeInputIndex);

  // Determines if a transfer to a register can be replaced by a direct usage
  // of the associated FIFO. The transformation allows skipping the allocation
  // of the correpsonding register(s).
  //
  // Returns
  //   - true in case the intended transformation is semantically correct,
  //   - false otherwise.
  bool canUseFifo(const dsc2::TransferNode* transferNode, size_t dstIndex,
                  const dsc2::ScheduleNode* consumer) const;
  // Determines if a transfer to a register can be latched. The transformation
  // allows skipping the allocation of the correpsonding register(s).
  //
  // Returns
  //   - true in case the intended transformation is semantically correct,
  //   - false otherwise.
  bool canUseLatch(const dsc2::TransferNode* transferNode, size_t dstIndex,
                   std::vector<dsc2::ScheduleNode*> consumer) const;

  std::string getNodeDescription(const dsc2::ScheduleNode* node) const;
  bool isExternalNode(const dsc2::ScheduleNode* node) const {
    return metadata.externalNodes_.count(node);
  }

  bool storageOrDatastreamIsExternal(const dsc2::DataInfo& dataInfo,
                                     SenComponents storage,
                                     bool isIncoming) const;

  bool srcRelatedToExternalNodes(const dsc2::TransferNode* node) const;
  bool destRelatedToExternalNodes(const dsc2::TransferNode* node) const;
  bool destRelatedToExternalNodes(const dsc2::TransferNode* node,
                                  int dstIndex) const;
  bool relatedToExternalNodes(const dsc2::TransferNode* node) const;

  bool inputRelatedToExternalNodes(const dsc2::ComputeNode* node) const;
  bool outputRelatedToExternalNodes(const dsc2::ComputeNode* node) const;
  bool relatedToExternalNodes(const dsc2::ComputeNode* node) const;
  bool relatedToExternalNodes(const dsc2::SyncNode* node) const;

  bool relatedToExternalNodes(const dsc2::BlockNode* root) const;

  // PE-SFP worksplitting
  dsc2::AllocateNode* cloneForPeSfpWorkSplit(dsc2::AllocateNode* node,
                                             bool skipMetadataUpdate = false);
  dsc2::ComputeNode* cloneForPeSfpWorkSplit(dsc2::ComputeNode* node);
  dsc2::TransferNode* cloneForPeSfpWorkSplit(dsc2::TransferNode* node);

  void cloneComputeForOffsetAdjustment(dsc2::ComputeNode* node);
  bool cloneForOffsetAdjustment();

  // High-level transformations:

  // Implements 4B splat read from LXLU, needed when reading scale=-2 input with
  // fp32 data format. The hardware only has 2Bsplat and 16Bsplat reads in LXLU,
  // so we achieve 4Bsplat with a 16Bsplat read from LXLU combined with a SPAT
  // instruction in the compute unit.
  bool transformFor4BsplatRead();

  // Moves a transfer above unrelated loops. An enclosing loop may be split
  // so that all loop-dimensions that are not related to the transfer are
  // associated with the innermost split-loop.
  bool hoistTransfersUpForReuse();

  // Detects and applies transformation for one of the following cases.
  // - Data is written to register and immediately read and never reused. The
  //   register result is converted to FIFO.
  // - Data is written to register and possibly reused but intervening
  //   operations do not use the relevant input-ports of the consumers. The
  //   register result is converted to latch.
  //
  // In both cases,
  // - the related producers and consumers are removed from the associated
  //   register allocation-tracker, and
  // - the consumers of the register result are updated to use the FIFO/latch.
  bool transformRegToFifoOrLatch();

  // Parallelize computation by splitting work across PE and SFP.
  // - Input SDSC datastages indicate which dimension (if there exists any) can
  //   be used for PE/SFP worksplit.
  // - Input DDL spcifies dataflow for one of the components (PE or SFP), DDC
  //   replicates the dataflow on the other component.
  // - Relevant data transfers are unrolled to achieve uniform granularity for
  //   all data transfers that are associated with the work-split.
  bool performPeSfpWorkSplit();

  // Identifies all "assign" compute nodes in the current DSC
  // and if their input and output layouts are incompatible,
  // inserts pack/merge operations to slice-wise data movement.
  bool performAutomaticShuffling();

  // Traverses the schedule tree and unrolls any transfer that has symbolic
  // size.
  bool unrollSymbolicTransfers();
  // bool transformForInterSliceTranspose();
  // bool transformAComputeNodeForInterSliceRestickify_old(
  //     dsc2::ComputeNode* compNode, int none_trivial_input_idx, int newLdsIdx,
  //     dsc2::TransferNode* releventTransferNode);
  // bool transformForInterSliceRestickify_old();

  bool unrollSpreadTransfers();

  // Set transferSize for transfers with fix size
  void setSizeForFixedSizeTransfers();

  bool transformAComputeNodeForInterSliceRestickify(dsc2::ComputeNode* compNode,
                                                    int none_trivial_input_idx,
                                                    PrimaryDimTypes dimForLoop);
  bool transformForInterSliceRestickify();

  // Determines if a scheduleNode is related to any of a given list of
  // components.
  //
  // Returns:
  // - True : The node is related. Parameter nodeComp is the node's component.
  // - False: The node is not related. Parameter nodeComp is not valid.
  bool isNodeRelatedToComps(const dsc2::ScheduleNode* node,
                            const std::vector<SenComponents>& comps,
                            SenComponents& nodeComp);

  // Methods related to coordinate capturing
  // ----------------------------------------------

  class CoordPropTracker {
   public:
    void addPropInfo(dsc2::ScheduleNode* refNode,
                     dsc2::ScheduleNode* nodeToFold,
                     const std::vector<PrimaryDimTypes> dims,
                     const std::string dataConnect = "",
                     const bool refIsProducer = true,
                     const bool scaleDown = false) {
      std::vector<PrimaryDimTypes> unseenDims;
      for (auto& dim : dims) {
        if (refsAdded_.count(nodeToFold) &&
            refsAdded_.at(nodeToFold).count(refNode) &&
            refsAdded_.at(nodeToFold).at(refNode).count(dim)) {
          // This propagation step has already been included.
          continue;
          ;
        }
        unseenDims.push_back(dim);
        refsAdded_[nodeToFold][refNode][dim] = 0;
      }
      if (unseenDims.empty()) {
        // No remaining dimension for propagation.
        return;
      }
      itemsToProcess_.push_back(dsc2::CoordPropInfoType{
          refNode, nodeToFold, dataConnect, refIsProducer,
          dsc2::CoordPropInfoType::PropStateType::NOT_PROCESSED, unseenDims,
          scaleDown});
    }

    void addPropInfo(const dsc2::CoordPropInfoType& rhs,
                     const std::vector<PrimaryDimTypes> dims) {
      addPropInfo(rhs.refNode, rhs.nodeToFold, dims, rhs.dataConnect,
                  rhs.refIsProducer, rhs.scaleDown);
    }

    void retry(const dsc2::CoordPropInfoType& propInfo,
               const std::vector<PrimaryDimTypes> dims) {
      itemsToProcess_.push_back(dsc2::CoordPropInfoType{
          propInfo.refNode, propInfo.nodeToFold, propInfo.dataConnect,
          propInfo.refIsProducer,
          dsc2::CoordPropInfoType::PropStateType::NOT_PROCESSED, dims});
      for (auto& dim : dims) {
        int retryCount = 0;
        if (refsAdded_[propInfo.nodeToFold].count(propInfo.refNode) &&
            refsAdded_[propInfo.nodeToFold].at(propInfo.refNode).count(dim)) {
          retryCount =
              refsAdded_[propInfo.nodeToFold].at(propInfo.refNode).at(dim);
          if (retryCount == 15) {
            DT_ERROR(
                "Retry threshold for propagation reached for " +
                propInfo.refNode->name_ + " -> " + propInfo.nodeToFold->name_ +
                ", dim= " + EnumsConversion::primaryDimToString.at(dim) + ".");
          }
          ++retryCount;
        }

        refsAdded_[propInfo.nodeToFold][propInfo.refNode][dim] = retryCount;
      }
    }

    // Fetches the next propagation entry. The processing of the entry is
    // assumed to be complete irrespective of the usage on the caller side.
    bool getCurrItem(dsc2::CoordPropInfoType& nextCoordPropInfo) {
      ++currItemToProcess_;
      if (currItemToProcess_ >= itemsToProcess_.size()) {
        return false;
      }
      itemsToProcess_.at(currItemToProcess_).propState =
          dsc2::CoordPropInfoType::PropStateType::COMPLETE;
      nextCoordPropInfo = itemsToProcess_.at(currItemToProcess_);
      return true;
    }

    void rollbackToPos(int newPos) {
      if (currItemToProcess_ < newPos || currItemToProcess_ < 0) {
        return;
      }
      if (newPos < 0 || newPos > currItemToProcess_) {
        DT_ERROR("Invalid rollback position " + std::to_string(newPos) +
                 ", acceptable range is [0 - " +
                 std::to_string(currItemToProcess_) + "].");
      }
      // Clear the computed coordinates up to the rollback position.
      for (int i = newPos; i <= currItemToProcess_; ++i) {
        itemsToProcess_.at(currItemToProcess_).propState =
            dsc2::CoordPropInfoType::PropStateType::ROLLED_BACK;
        if (itemsToProcess_[i].nodeToFold->nodeType_ ==
            dsc2::ScheduleNode::ALLOCATE) {
          auto allocNode =
              static_cast<dsc2::AllocateNode*>(itemsToProcess_[i].nodeToFold);
          allocNode->allocateCoordinates_.clear();
          allocNode->sliceViewCoordinates_.clear();
        } else if (itemsToProcess_[i].nodeToFold->nodeType_ ==
                   dsc2::ScheduleNode::TRANSFER) {
          auto transferNode =
              static_cast<dsc2::TransferNode*>(itemsToProcess_[i].nodeToFold);
          transferNode->transferCoordinates_.clear();
        } else if (itemsToProcess_[i].nodeToFold->nodeType_ ==
                   dsc2::ScheduleNode::COMPUTE) {
          auto computeNode =
              static_cast<dsc2::ComputeNode*>(itemsToProcess_[i].nodeToFold);
          computeNode->outputCoordinate_.clear();
          for (auto& inputCoord : computeNode->inputCoordinates_) {
            inputCoord.clear();
          }
        }
      }
      currItemToProcess_ = newPos - 1;
    }

    void rollBackNodesInBlock(DesignSpaceConfig* currDsc,
                              dsc2::BlockNode* blockRoot) {
      std::vector<dsc2::ScheduleNode*> blockNodes =
          currDsc->scheduleTree_.traverseTreeDFSMutable(
              blockRoot,
              {dsc2::ScheduleNode::ALLOCATE, dsc2::ScheduleNode::COMPUTE,
               dsc2::ScheduleNode::TRANSFER});

      for (int i = 0; i < currItemToProcess_; ++i) {
        if (is_any_of(itemsToProcess_.at(i).refNode, blockNodes) ||
            is_any_of(itemsToProcess_.at(i).nodeToFold, blockNodes)) {
          rollbackToPos(i);
          return;
        }
      }
    }

    void reset() {
      itemsToProcess_.clear();
      refsAdded_.clear();
      currItemToProcess_ = -1;
    }

    bool empty() { return itemsToProcess_.empty(); }

   private:
    std::deque<struct dsc2::CoordPropInfoType> itemsToProcess_;
    //   Outer level key: nodeForFold
    //   Inner level key: reference node
    //   Inner level val: per-dimension retry count due to row-bundling failure
    std::map<
        const dsc2::ScheduleNode*,
        std::map<const dsc2::ScheduleNode*, std::map<PrimaryDimTypes, int>>>
        refsAdded_;
    int currItemToProcess_ = -1;
  };

  CoordPropTracker coordPropTracker;

  std::map<const dsc2::ScheduleNode*,
           std::map<const dsc2::ScheduleNode*,
                    dsc2::LoopDistributionParamPerNodeType>>
      loopDistributionParamInfo;

  struct RowGroupInfo {
    enum Category {
      ROW_TO_SAME_ROW,
      NONROW_TO_ROW,
      ROW_TO_NONROW,
      ROW_NORTH_SOUTH,
      NO_BUNDLING
    } cat = NO_BUNDLING;
    // The innermost block node (loop, conditional, and so on) that contains all
    // nodes on the row-split group.
    dsc2::BlockNode* commonGroupAncestor = nullptr;
    struct RowGroupNodeInfo {
      dsc2::ScheduleNode* node = nullptr;
      int row = -1;
      // A scheduleNode may have multiple coordinates with different betas for
      // rows (computeNodes, for example). The following field keeps track of
      // the beta to be used for row-grouping.
      CoordinateBaseType beta = -1;
    };
    std::vector<RowGroupNodeInfo> nodeInfo;
    // In case the information relates to a single row (i.e. ROW_TO_SAME_ROW
    // category), activeRow identifies the specific row index.
    int activeRow = -1;
    bool ascendingOrder = true;

    void print(std::ostream& out) const {
      out << "\nRowgroup: "
          << "\n  Category= ";
      switch (cat) {
        case (Category::ROW_TO_NONROW):
          out << "Row-to-NonRow";
          break;
        case (Category::ROW_TO_SAME_ROW):
          out << "Row-to-SameRow";
          break;
        case (Category::NONROW_TO_ROW):
          out << "NonRow-to-Row";
          break;
        case (Category::ROW_NORTH_SOUTH):
          out << "Row-North-South";
          break;
        case (Category::NO_BUNDLING):
          out << "No-Bundling";
          break;
        default:
          break;
      }
      out << "\n  Group elements:";
      for (auto& [node, row, beta] : nodeInfo) {
        out << " (" << node->name_ << ", row= " << row << ", beta=" << beta
            << ")";
      }
      out.flush();
    }
  };

  void printFoldParams(std::vector<dsc2::FoldParamInfoType>& foldParams) {
    for (auto& fpInfo : foldParams) {
      std::cout << "(" << fpInfo.alpha << ", " << fpInfo.beta << ", "
                << fpInfo.cardinality << ", " << fpInfo.foldDimLabel << ") ";
    }
  }

  // Computes number of elements processed by an individual PT row along the
  // rowSplit dimension.
  int getNumElementsInPTSlice(int ldsIdx, PrimaryDimTypes dim);

  // Constructs spatial folds for a given schedule node and a given primary
  // dimension. The Spatial folds correspond to core-workslice and corelet
  // split.
  //
  // Inputs:
  //   - A schedule node (allocate, compute, or transfer, in practice).
  //   - Primary dimension
  //   - Reference to the coordinate object that is updated by this function.
  void buildSpatialFold(
      dsc2::ScheduleNode* node, const PrimaryDimTypes& currDim,
      const PadType& currPadType,
      dsc2::CoordinateType<CoordinateBaseType>& nodeCoordinates);
  void computeParamsForRowSplitFold(const PrimaryDimTypes& currDim,
                                    dsc2::FoldParamInfoType& resultFoldParams,
                                    RowGroupInfo& relatedRows);
  void buildFoldForExternalAllocation(dsc2::AllocateNode* allocNode);
  bool buildFoldForAllocation(
      dsc2::CoordPropInfoType& coordPropInfo,
      const dsc2::CoordinateType<CoordinateBaseType>& inputRefCoord,
      dsc2::AllocateNode* allocNode);
  void relateLoopsToAllocElemArr(
      dsc2::CoordPropInfoType& coordPropInfo, const PrimaryDimTypes dim,
      const dsc2::CoordinateType<CoordinateBaseType>& coordinate,
      const dsc2::VectorOfLoopAndDim& refLoopChain,
      dsc2::LoopDistributionParamPerNodeType& loopParamsAfterDistribution);
  void buildFoldFromAllocation(
      dsc2::CoordPropInfoType& coordPropInfo, dsc2::ScheduleNode* node,
      dsc2::CoordinateType<CoordinateBaseType>& coordinate,
      SenComponents sizeRefComp, SenComponents propRefComp,
      dsc2::LoopDistributionParamPerNodeType& loopParamsAfterDistribution,
      RowGroupInfo& refRowGroup);

  // A utility function to construct folds for cases where the reference is NOT
  // an allocateNode. The reference node and the working node can be in
  // different loop nests with a common loop ancestor.
  //
  // Inputs:
  //   - refNode       : Reference node which is not an allocateNode.
  //   - refCoordinate : Specifies which of the coordinates of the refNode
  //                     should be used as the reference coordinate. See Note
  //                     below.
  //   - nodeForFold   : Folds are constructed for this node.
  //   - coordinate    : Folds are constructed for this coordinate in
  //                     nodeForFold. See note below.
  //   - foldSingleDim :
  //     * When specified, fold is constructed only for the given dimension
  //       (needed while constructing folds for compute nodes).
  //     * Otherwise, fold is constructed for all dimensions present in the
  //       reference node's fold.
  //
  //  Note:
  //    ComputeNodes are associated with multiple coordinates. Depending on the
  //    padding type of the related data structures, the coordinates may vary
  //    among those coordinates. Therefore, it is necessary to specify which of
  //    those coordinates are relevant for the fold construction.

  bool buildFoldFromNonAllocRef(
      dsc2::CoordPropInfoType& coordPropInfo, const int refLdsIdx,
      const dsc2::CoordinateType<CoordinateBaseType>& refCoordinate,
      dsc2::CoordinateType<CoordinateBaseType>& coordinate,
      const SenComponents sizeRefComp, SenComponents propRefComp,
      dsc2::LoopDistributionParamPerNodeType& loopParamsAfterDistribution,
      RowGroupInfo& refRowGroup,
      PrimaryDimTypes foldSingleDim = PrimaryDimTypes::PrimaryDimTypesCount);

  // Constructs coordinates for computeNodes from a related reference
  // scheduleNode.
  //
  // Inputs:
  //   - computeNode   : Coordinates are constructed for this scheduleNode.
  //   - coordPropInfo : Provides information on reference coordinate.
  //   - constructedInputCoords  : Input indices for which coordinates are
  //   constructed in the current invocation.
  //   - constructedOutputCoords : Output indices for which coordinates are
  //   constructed in the current invocation.
  //
  // Returns:
  //   - true : in case new coordinates are constructed in the current
  //   invocation.
  //   - false: othewise.
  bool buildFoldForCompute(dsc2::ComputeNode* computeNode,
                           dsc2::CoordPropInfoType& coordPropInfo,
                           std::vector<int>& constructedInputCoords,
                           std::vector<int>& constructedOutputCoords);
  bool buildFoldForTransfer(dsc2::TransferNode* transferNode,
                            dsc2::CoordPropInfoType& coordPropInfo);
  void gatherFoldParams(const FoldManager<CoordinateBaseType>& fm,
                        std::vector<dsc2::FoldParamInfoType>& foldParams);
  bool sameCoordinateRange(const dsc2::ScheduleNode* lhsNode,
                           const dsc2::CoordinateType<CoordinateBaseType>& lhs,
                           const dsc2::ScheduleNode* rhsNode,
                           const dsc2::CoordinateType<CoordinateBaseType>& rhs,
                           SenComponents comp = SenComponents::ALL,
                           int ldsIdx = -1, bool commonDimsOnly = false);
  bool gatherRelatedPTRowsBase(const dsc2::CoordPropInfoType& coordPropInfo,
                               RowGroupInfo& relatedRows);
  bool gatherRelatedPTRows(
      RowGroupInfo& refRowGroup, const dsc2::CoordPropInfoType& coordPropInfo,
      const dsc2::CoordinateType<CoordinateBaseType>& refCoordinate);
  bool needNonRowBundling(std::string& dataConnect, bool checkProducers) const;
  dsc2::CoordinateType<CoordinateBaseType>& getRelatedComputeCoord(
      dsc2::ComputeNode* computeNode, dsc2::CoordPropInfoType& refFoldInfo,
      std::vector<int>& constructedInputCoords,
      std::vector<int>& constructedOutputCoords, int& selectedLdsIdx);
  void buildAndPropagateFold();
  void coordinateCapture();
  // Makes a copy of refLds and adds it to the new entry to labeledDs list right
  // before the last one. Also, it updates indexes of labeledDs for currDSC
  int addNewLds(LabeledDsInfo* refLds);

  // Updates the labeledDs indexes of the DSC2 nodes
  void updateNodesWithNewLds(int newLdsIdx, int oldLdsIdx,
                             dsc2::ScheduleNode* startNode);

  // Checks if there is a compute op in currDSC which benefits from PackStickDim
  // optimization. The first entry of returned pair indicates if there is such a
  // compute op, and the second one will return the number of inputs. The
  // compute node will be also assigned to releventComputeNode variable
  std::pair<bool, int> isReleventComputeToPackStickDim(
      dsc2::ComputeNode*& releventComputeNode);

  // Gets a compute op releventComputeNode. It will return the vector of
  // relevent input transfers (releventInputTransfers) and out put transfer
  // (releventOutputTransfer). Also it retuns the index of relevent destinations
  // (releventTransferDstIdx) to those input and output transfers.
  void findReleventStickPackingTransfers(
      dsc2::ComputeNode* releventComputeNode,
      std::vector<dsc2::TransferNode*>& releventInputTransfers,
      dsc2::TransferNode*& releventOutputTransfer,
      std::vector<int>& releventTransferDstIdx);

  // checks if the transfers has minimum eligibility for the optimization to
  // precede.
  bool isEligibleCompute(
      std::vector<dsc2::TransferNode*>& releventInputTransfers,
      int numberOfInputs, std::vector<int>& ldsInputIdx);

  // Do stick packing optimization, return true if it is success otherwise
  // false.
  // First finds the eligible compute op with input and output transfers. Then,
  // tries to find elements from other dimensions to pack more relevent elements
  // into one stick. It will updates the stick layout by creating new internal
  // tensors for inputs and output.
  // The original input tensor will be compressed to an internal input tensor
  // which is passed to compute op. The output of the compute op which is
  // compressed, will be expanded to the size of the original output tensor.
  //
  // for compression and expansion step, the following loops are
  // created.
  //
  // loop0 (ds_chunk/ds_packed) {
  //   innerLoop0 (ds_packed/newBotDs) {
  //     transfer LX -> SFP // original input
  //     dummy compute
  //     transfer SFP -> LX // compressed input
  //   }
  // }
  // sync LXLSU -> LXLU
  // parent_loop (ds_chunk/ds_I) {
  //   transfer LX -> SFP // compressed input
  //   Opaque Op
  //   transfer SFP -> LX // compressed output
  // }
  // sync LXLSU -> LXLU
  // loop1 (ds_chunk/ds_packed) {
  //   innerLoop1 (ds_packed/newBotDs) {
  //     transfer LX -> SFP // compressed output
  //     dummy compute
  //     transfer SFP -> LX // original output
  //   }
  // }
  //
  bool packStickDim();
  void scaleDownCoord(dsc2::CoordinateType<CoordinateBaseType>& lhsCoord,
                      dsc2::CoordinateType<CoordinateBaseType>& rhsCoord,
                      const int ldsIdx);
  void scaleUpCoord(dsc2::ScheduleNode* lhsNode,
                    dsc2::CoordinateType<CoordinateBaseType>& lhsCoord,
                    dsc2::CoordinateType<CoordinateBaseType>& rhsCoord,
                    const int ldsIdx, const SenComponents comp);
};

}  // namespace ddc
#endif
