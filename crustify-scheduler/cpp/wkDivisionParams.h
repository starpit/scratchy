/************************************************************
* IBM Confidential
* (C) Copyright IBM Corp. 2022, 2025
************************************************************/

/*
 * Description:
 *
 */

#ifndef FOLD_INFRASTRUCTURE_WK_PARAM
#define FOLD_INFRASTRUCTURE_WK_PARAM

/**
 * @brief Class captures work division params for each dimension
 *
 */

class WkSplitParam {
 public:
  struct StrWinPad {
    bool isStrWinPad_ = false;
    int32_t stride_ = -1;
    int32_t window_ = -1;
    int32_t extra_back_ = 0;
  };
  /**
   * @brief Construct a new Wk Split Param object
   *
   */
  WkSplitParam() {}

  /**
   * @brief Construct a new Wk Split Param object
   *
   * @param wk_ss
   * @param wk_epilogue
   * @param max_cores
   * @param start_cid_offset
   * @param num_ss_slices
   * @param num_epilogue_slices
   * @param gap_within_inner_repeat
   * @param repeat_factor_inner
   * @param gap_after_inner_repeat
   * @param gap_after_all_slices
   * @param outer_repeat_factor
   * @param real_coordinates
   * @param stride_window_padded_type
   */
  WkSplitParam(int32_t wk_ss, int32_t wk_epilogue, int32_t max_cores,
               int32_t start_cid_offset, int32_t num_ss_slices,
               int32_t num_epilogue_slices, int32_t gap_within_inner_repeat,
               int32_t repeat_factor_inner, int32_t gap_after_inner_repeat,
               int32_t gap_after_all_slices, int32_t outer_repeat_factor,
               const std::vector<std::pair<int32_t, int32_t>>& real_coordinates,
               const StrWinPad& swp_info_) {
    build(wk_ss, wk_epilogue, max_cores, start_cid_offset, num_ss_slices,
          num_epilogue_slices, gap_within_inner_repeat, repeat_factor_inner,
          gap_after_inner_repeat, gap_after_all_slices, outer_repeat_factor,
          real_coordinates, swp_info_);
  }

  /**
   * @brief Destroy the Wk Split Param object
   *
   */
  ~WkSplitParam() {}

  /**
   * @brief
   *
   * @param wk_ss
   * @param wk_epilogue
   * @param max_cores
   * @param start_cid_offset
   * @param num_ss_slices
   * @param num_epilogue_slices
   * @param repeat_factor_inner
   * @param gap_after_inner_repeat
   * @param gap_after_all_slices
   * @param outer_repeat_factor
   * @param real_coordinates
   * @param stride_window_padded_type
   */
  void build(int32_t wk_ss, int32_t wk_epilogue, int32_t max_cores,
             int32_t start_cid_offset, int32_t num_ss_slices,
             int32_t num_epilogue_slices, int32_t gap_within_inner_repeat,
             int32_t repeat_factor_inner, int32_t gap_after_inner_repeat,
             int32_t gap_after_all_slices, int32_t outer_repeat_factor,
             const std::vector<std::pair<int32_t, int32_t>>& real_coordinates,
             const StrWinPad& swp_info) {
    wk_ss_ = wk_ss;
    wk_epilogue_ = wk_epilogue;
    max_cores_ = max_cores;
    start_cid_offset_ = start_cid_offset;
    num_ss_slices_ = num_ss_slices;
    num_epilogue_slices_ = num_epilogue_slices;
    gap_within_inner_repeat_ = gap_within_inner_repeat;
    repeat_factor_inner_ = repeat_factor_inner;
    gap_after_inner_repeat_ = gap_after_inner_repeat;
    gap_after_all_slices_ = gap_after_all_slices;
    outer_repeat_factor_ = outer_repeat_factor;
    real_coordinates_ = real_coordinates;
    swp_info_ = swp_info;
    isBuilt_ = true;
    checkLegality();
  }

  void build(const WkSplitParam& ref) {
    wk_ss_ = ref.wk_ss_;
    wk_epilogue_ = ref.wk_epilogue_;
    max_cores_ = ref.max_cores_;
    start_cid_offset_ = ref.start_cid_offset_;
    num_ss_slices_ = ref.num_ss_slices_;
    num_epilogue_slices_ = ref.num_epilogue_slices_;
    gap_within_inner_repeat_ = ref.gap_within_inner_repeat_;
    repeat_factor_inner_ = ref.repeat_factor_inner_;
    gap_after_inner_repeat_ = ref.gap_after_inner_repeat_;
    gap_after_all_slices_ = ref.gap_after_all_slices_;
    outer_repeat_factor_ = ref.outer_repeat_factor_;
    real_coordinates_ = ref.real_coordinates_;
    swp_info_ = ref.swp_info_;
    isBuilt_ = true;
    checkLegality();
  }

  /**
   * @brief checks legality of wkSplit params
   *
   */
  void checkLegality() const {
    DT_CHECK(outer_repeat_factor_ > 0);
    DT_CHECK(repeat_factor_inner_ > 0);
    DT_CHECK((num_ss_slices_ + num_epilogue_slices_) * outer_repeat_factor_ <=
             max_cores_);
  }

  // utils..
  /**
   * @brief
   *
   * @param cid
   * @return int32_t
   */
  int32_t adjustCID(int32_t cid) const {
    return ((cid < 0) ? cid + max_cores_ : cid);
  }

  /**
   * @brief Get the single slice Inner Length object
   *
   * @param with_gap
   * @return int32_t
   */
  int32_t getSingleSliceInnerLength(bool with_gap = true) const {
    if (with_gap)
      return (gap_within_inner_repeat_ + 1) * repeat_factor_inner_ +
             gap_after_inner_repeat_;

    return (gap_within_inner_repeat_ + 1) * repeat_factor_inner_;
  }

  /**
   * @brief Get the Full Inner Length object
   *
   * @param with_gap
   * @return int32_t
   */
  int32_t getFullInnerLength(bool with_gap = true) const {
    if (with_gap)
      return getSingleSliceInnerLength() *
                 (num_ss_slices_ + num_epilogue_slices_) +
             gap_after_all_slices_;

    return getSingleSliceInnerLength() *
           (num_ss_slices_ + num_epilogue_slices_);
  }

  int32_t getOuterRepeatFactor() const { return outer_repeat_factor_; }

  int32_t getNumSSslices() const { return num_ss_slices_; }

  int32_t getNumElSlices() const { return num_epilogue_slices_; }

  int32_t getWkSs() { return wk_ss_; }
  void updateWkSs(int32_t new_wk_ss_) { wk_ss_ = new_wk_ss_; }
  int32_t getWkEl() { return wk_epilogue_; }
  void updateWkEl(int32_t new_wk_el) { wk_epilogue_ = new_wk_el; }

  /**
   * @brief Get the Slice Id object
   *
   * @param cid
   * @return int32_t
   */
  int32_t getSliceId(int32_t cid) const {
    int32_t offset_cid = adjustCID(cid - start_cid_offset_);
    auto full_wksl_size_with_after_gaps_ = getFullInnerLength();

    if (offset_cid >= full_wksl_size_with_after_gaps_ * outer_repeat_factor_)
      return -1;  // gap cores that come in the end

    // fold cids using outer repeat factor
    offset_cid = offset_cid % full_wksl_size_with_after_gaps_;

    // consider effect of gap_after_all_slices_
    auto full_wksl_size_no_after_gaps_ = getFullInnerLength(false);
    if (offset_cid >= full_wksl_size_no_after_gaps_)
      return -1;  // gap that comes after all wk slices are passed

    // Get the work slice (ignoring inner gaps)
    auto single_slice_span = getSingleSliceInnerLength();
    auto slid = offset_cid / single_slice_span;

    offset_cid = offset_cid % single_slice_span;

    // consider effect of gap_after_inner_repeat_
    auto single_slice_span_no_after_gaps_ = getSingleSliceInnerLength(false);
    if (offset_cid >= single_slice_span_no_after_gaps_)
      return -1;  // gap that comes each wk slice

    offset_cid = offset_cid % (gap_within_inner_repeat_ + 1);
    // consider effect of gap_within_inner_repeat_
    if (offset_cid >= 1) return -1;  // gap after each valid wk slice

    return slid;
  }

  /**
   * @brief Get the Size object
   *
   * @param cid
   * @return int32_t
   */
  int32_t getSize(int32_t cid) const {
    DT_CHECK(isBuilt_);
    auto slice_id = getSliceId(cid);
    if (slice_id == -1) {
      return 0;
    } else {
      auto vsize = (slice_id < num_ss_slices_) ? wk_ss_ : wk_epilogue_;
      if (swp_info_.isStrWinPad_) {
        vsize = (vsize - 1) * swp_info_.stride_ + swp_info_.window_ +
                swp_info_.extra_back_;
      }
      return vsize;
    }
  }

  std::vector<std::pair<int64_t, int64_t>> getCoordVec(int32_t cid) const {
    auto v_coord = getCoord(cid);
    std::vector<std::pair<int64_t, int64_t>> coord_vec;
    if (real_coordinates_.empty() || v_coord.first < 0) {
      coord_vec.push_back(v_coord);
    } else {
      int64_t cumRunningSize = 0;
      for (auto& pair : real_coordinates_) {
        std::pair<int64_t, int64_t> runningUnrealCoord(0, 0);
        runningUnrealCoord.first = cumRunningSize;
        runningUnrealCoord.second =
            runningUnrealCoord.first + (pair.second - pair.first);
        cumRunningSize += (pair.second - pair.first + 1);
        bool noOverlap = (runningUnrealCoord.first > v_coord.second ||
                          runningUnrealCoord.second < v_coord.first);
        if (!noOverlap) {
          std::pair<int64_t, int64_t> myRealCoord;
          if (v_coord.first <= runningUnrealCoord.first) {
            myRealCoord.first = pair.first;
          } else {
            myRealCoord.first =
                pair.first + (v_coord.first - runningUnrealCoord.first);
          }
          if (v_coord.second >= runningUnrealCoord.second) {
            myRealCoord.second = pair.second;
          } else {
            myRealCoord.second =
                pair.second - (runningUnrealCoord.second - v_coord.second);
          }
          coord_vec.push_back(myRealCoord);
        }
      }
    }
    return coord_vec;
  }

  bool isBuilt() const { return isBuilt_; }

  /**
   * @brief prints meta data
   *
   * @param out
   * @param ps
   */
  void printMetaData(std::ostream& out, std::string ps = "") const {
    auto QUOTE = [](std::string str) { return "\"" + str + "\""; };
    out << ps << QUOTE("isBuilt_") << " : " << isBuilt_ << ",";
    out << ps << QUOTE("wk_ss_") << " : " << wk_ss_ << ",";
    out << ps << QUOTE("wk_epilogue_") << " : " << wk_epilogue_ << ",";
    out << ps << QUOTE("max_cores_") << " : " << max_cores_ << ",";
    out << ps << QUOTE("start_cid_offset_") << " : " << start_cid_offset_
        << ",";
    out << ps << QUOTE("num_ss_slices_") << " : " << num_ss_slices_ << ",";
    out << ps << QUOTE("num_epilogue_slices_") << " : " << num_epilogue_slices_
        << ",";
    out << ps << QUOTE("gap_within_inner_repeat_") << " : "
        << gap_within_inner_repeat_ << ",";
    out << ps << QUOTE("repeat_factor_inner_") << " : " << repeat_factor_inner_
        << ",";
    out << ps << QUOTE("gap_after_inner_repeat_") << " : "
        << gap_after_inner_repeat_ << ",";
    out << ps << QUOTE("gap_after_all_slices_") << " : "
        << gap_after_all_slices_ << ",";
    out << ps << QUOTE("outer_repeat_factor_") << " : " << outer_repeat_factor_
        << ",";
    out << ps << QUOTE("swp_info_") << " : { ";
    out << QUOTE("isStrWinPad_") << " : " << swp_info_.isStrWinPad_ << ", ";
    out << QUOTE("stride_") << " : " << swp_info_.stride_ << ", ";
    out << QUOTE("window_") << " : " << swp_info_.window_ << ", ";
    out << QUOTE("extra_back_") << " : " << swp_info_.extra_back_;
    out << " },";
    out << ps << QUOTE("real_coordinates_") << " : [ ";
    for (int p = 0; p < real_coordinates_.size(); p++) {
      auto& pair = real_coordinates_.at(p);
      out << "[" << pair.first << ", " << pair.second << "]";
      if (p != real_coordinates_.size() - 1) {
        out << ", ";
      }
    }
    out << " ]";
  }

  /**
   * @brief Comparision operator
   *
   * @param rhs
   * @return true
   * @return false
   */
  bool operator==(const WkSplitParam& rhs) const {
    if (wk_ss_ != rhs.wk_ss_ || wk_epilogue_ != rhs.wk_epilogue_ ||
        max_cores_ != rhs.max_cores_ ||
        start_cid_offset_ != rhs.start_cid_offset_ ||
        num_ss_slices_ != rhs.num_ss_slices_ ||
        num_epilogue_slices_ != rhs.num_epilogue_slices_ ||
        gap_within_inner_repeat_ != rhs.gap_within_inner_repeat_ ||
        repeat_factor_inner_ != rhs.repeat_factor_inner_ ||
        gap_after_inner_repeat_ != rhs.gap_after_inner_repeat_ ||
        gap_after_all_slices_ != rhs.gap_after_all_slices_ ||
        outer_repeat_factor_ != rhs.outer_repeat_factor_ ||
        real_coordinates_ != rhs.real_coordinates_ ||
        swp_info_.isStrWinPad_ != rhs.swp_info_.isStrWinPad_ ||
        swp_info_.window_ != rhs.swp_info_.window_ ||
        swp_info_.stride_ != rhs.swp_info_.stride_ ||
        swp_info_.extra_back_ != rhs.swp_info_.extra_back_)
      return false;
    return true;
  }

  /**
   * @brief get real_coordinates_
   *
   */
  const std::vector<std::pair<int32_t, int32_t>>& get_real_coordinates_()
      const {
    return real_coordinates_;
  }

  /**
   * @brief set real_coordinates_
   *
   */
  void set_real_coordinates_(
      const std::vector<std::pair<int32_t, int32_t>>& real_coordinates) {
    DT_CHECK(isBuilt_);
    real_coordinates_ = real_coordinates;
  }

 private:
  bool isBuilt_ = false;

  // work size info..
  int32_t wk_ss_ = 0;        // work done by SteadyState cores
  int32_t wk_epilogue_ = 0;  // work done by Epilogue cores

  // core info..
  int32_t max_cores_ = 0;         // max. number of cores
  int32_t start_cid_offset_ = 0;  // start core idx

  // work slice info..
  // total unique slices = ss_slices_ + epilogue_slices_
  int32_t num_ss_slices_ = 0;        // number of cores with SteadyState work
  int32_t num_epilogue_slices_ = 0;  //  number of cores with Epilogue work

  // params to map sliceId to coreId
  int32_t gap_within_inner_repeat_ = 0;  // number of gap cores after each valid
                                         // core assignment

  int32_t repeat_factor_inner_ = 0;  //  number of cores (spead apart by
                                     //  gap_within_inner_repeat_) that use the
                                     //  same wk slice, aka. NumInnerSameWkSl
  int32_t gap_after_inner_repeat_ = 0;
  // gap cores after the same wk slice, i.e., gap
  // cores after every
  // gap_within_inner_repeat_*repeat_factor_inner_
  // cores, aka. GapAfterInnerSameWkSl

  int32_t gap_after_all_slices_ = 0;
  /* gap cores after all wk slices are passed, i.e.,
  gap after (num_ss_slices_+num_epilogue_slices_)
  * (gap_within_inner_repeat_* repeat_factor_inner_
  + gap_after_inner_repeat_), aka. GapAfterFullWkSl
*/

  int32_t outer_repeat_factor_ = 0;  // Number of time full work slice is
                                     // repeated, aka. RepeatFullWkSl

  // exclusive params for coordinate computations
  std::vector<std::pair<int32_t, int32_t>> real_coordinates_;

  StrWinPad swp_info_;

  /**
   * @brief Get the Coord object
   *
   * @param cid
   * @return std::pair<int32_t, int32_t>
   */
  std::pair<int32_t, int32_t> getCoord(int32_t cid) const {
    DT_CHECK(isBuilt_);
    auto slice_id = getSliceId(cid);

    if (slice_id == -1) return std::make_pair(-1, -1);

    int32_t start_vcoord = 0;
    int32_t end_vcoord = 0;

    if (slice_id < num_ss_slices_) {
      start_vcoord = slice_id * wk_ss_;
      end_vcoord = start_vcoord + wk_ss_ - 1;
    } else {
      start_vcoord =
          num_ss_slices_ * wk_ss_ + (slice_id - num_ss_slices_) * wk_epilogue_;
      end_vcoord = start_vcoord + wk_epilogue_ - 1;
    }

    if (swp_info_.isStrWinPad_) {
      auto vsize = end_vcoord - start_vcoord + 1;
      vsize = (vsize - 1) * swp_info_.stride_ + swp_info_.window_ +
              swp_info_.extra_back_;
      start_vcoord = start_vcoord * swp_info_.stride_;
      end_vcoord = start_vcoord + vsize - 1;
    }

    return std::make_pair(start_vcoord, end_vcoord);
  }
};

#endif