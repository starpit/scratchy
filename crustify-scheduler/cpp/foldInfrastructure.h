/************************************************************
 * IBM Confidential
 * (C) Copyright IBM Corp. 2022, 2025
 ************************************************************/

/*
 * Description:
 *
 */

#ifndef FOLD_INFRASTRUCTURE_
#define FOLD_INFRASTRUCTURE_

#include <util/import_utils.h>
#include <util/print_utils.h>
#include <util/utils.h>

#include <cmath>
#include <cstdint>
#include <deque>
#include <external/json11/json11.hpp>
#include <iostream>
#include <list>
#include <map>
#include <set>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "util/dt_exception.hpp"
#include "wkDivisionParams.h"

/**
 * @brief Supports fold function types
 *
 */

enum class BaseFuncType {
  Constant = 0,
  Map = 1,
  Affine = 2,
  WkSplit = 3,
  Unknown = 4
};
namespace FoldInfraUtils {
static const std::map<BaseFuncType, std::string> baseFuncTypeToString = {
    {BaseFuncType::Constant, "Const"},
    {BaseFuncType::Map, "Map"},
    {BaseFuncType::Affine, "Affine"},
    {BaseFuncType::WkSplit, "WkSplit"}};
static const std::map<std::string, BaseFuncType> stringToBaseFuncType =
    flipMap(baseFuncTypeToString);

/**
 * @brief Get the Flattened Coordinates by only expanding pos specified by
 * pos_to_expandsize
 *
 * @param coordinates
 * @param pos_to_expandsize
 * @param num_dims
 * @param scan_inner_outer
 */
static void getFlattenedCoordinatesWithConstraints(
    std::vector<std::deque<int64_t>>& coordinates,
    const std::map<int64_t, int64_t>& pos_to_expandsize, const int& num_dims,
    const bool scan_inner_outer = false) {
  std::vector<int64_t> unique_data_coords(num_dims, 1);
  int64_t total_count = 1;
  for (auto& kv : pos_to_expandsize) {
    total_count *= kv.second;
    DT_CHECK(unique_data_coords.size() > kv.first);
    unique_data_coords.at(kv.first) = kv.second;
  }

  coordinates.resize(total_count);
  for (int idx = 0; idx < total_count; idx++) {
    coordinates.at(idx).resize(unique_data_coords.size());
  }

  int repeat_factor = 1;
  for (int i = 0; i < unique_data_coords.size(); i++) {
    const int dim_idx =
        scan_inner_outer ? unique_data_coords.size() - 1 - i : i;
    for (int coord_idx = 0; coord_idx < total_count; coord_idx++) {
      coordinates.at(coord_idx).at(dim_idx) =
          (coord_idx / repeat_factor) % unique_data_coords.at(dim_idx);
    }
    repeat_factor *= unique_data_coords.at(dim_idx);
  }
}

/**
 * @brief Method overwrites coordinates for pos present in pos_to_fixCoord to
 * fixed value specified by pos_to_fixCoord.at(pos)
 *
 * @param coordinates
 * @param pos_to_fixCoord
 */
static void fixCoordinatesAtPos(
    std::vector<std::deque<int64_t>>& coordinates,
    const std::map<int64_t, int64_t>& pos_to_fixCoord) {
  for (auto& coord : coordinates) {
    for (int dim_idx = 0; dim_idx < coord.size(); dim_idx++) {
      if (pos_to_fixCoord.count(dim_idx)) {
        DT_CHECK(coord.at(dim_idx) == 0);
        coord.at(dim_idx) = pos_to_fixCoord.at(dim_idx);
      }
    }
  }
}

}  // namespace FoldInfraUtils

/**
 * @brief Captures properties of a folded dimension
 *
 */
class FoldDimProp {
 public:
  FoldDimProp() {};
  // methods..
  FoldDimProp(uint32_t factor, std::string label = "")
      : factor_(factor), label_(label) {}

  [[nodiscard]] const uint32_t getSize() const { return factor_; }
  [[nodiscard]] const std::string& Label() const { return label_; }
  void setLabel(std::string label) { label_ = label; }
  void setSize(uint32_t new_fator) { factor_ = new_fator; }

  void print(std::ostream& out) const {
    auto QUOTE = [](std::string str) { return "\"" + str + "\""; };
    out << QUOTE("factor_") << " : " << getSize() << ", ";
    out << QUOTE("label_") << " : " << QUOTE(label_);
  }
  void importFromJson(const json11::Json& json) {
    for (auto& [name, jsonVal] : json.object_items()) {
      if (name == "factor_") {
        factor_ = jsonVal.int_value();
      } else if (name == "label_") {
        label_ = jsonVal.string_value();
      } else {
        DT_ERROR("Unexpected import field");
      }
    }
  }

  bool operator==(const FoldDimProp& rhs) const {
    return factor_ == rhs.factor_ && label_ == rhs.label_;
  }

 private:
  uint32_t factor_;
  std::string label_;  // optinal..
};

/**
 * @brief Base class for managing fold functions
 *
 * @tparam Dtype
 */

template <typename Dtype>
class FoldFunction {
 public:
  enum FuncType {
    Constant_leaf = 0,
    Map_leaf = 1,
    Affine_leaf = 2,
    Constant_nonleaf = 3,
    Map_nonleaf = 4,
    Affine_nonleaf = 5,
    WkSplit_leaf = 6,
    Unknown = 7
  };

  FoldFunction(FuncType type) : type_(type) {}
  virtual ~FoldFunction<Dtype>() {}
  const FuncType Type() const { return type_; }
  FuncType type_ = FuncType::Unknown;

  bool isLeaf() const {
    return is_any_of(type_, FuncType::Constant_leaf, FuncType::Map_leaf,
                     FuncType::Affine_leaf);
  }

  bool isNonLeaf() const {
    return is_any_of(type_, FuncType::Constant_nonleaf, FuncType::Map_nonleaf,
                     FuncType::Affine_nonleaf);
  }

  // access function, takes N dimensional folding indices as variadic variable
  template <typename... Args, class Enable = std::enable_if_t<(
                                  ... && std::is_convertible_v<Args, int64_t>)>>
  Dtype getData(const Args&... list) const {
    std::deque<int64_t> dim_indices{list...};
    return getData(dim_indices, 0);
  }

  virtual Dtype getData(const std::deque<int64_t>& fold_dim_indices,
                        size_t idx) const = 0;

  virtual FoldFunction<Dtype>* getChild() {
    DT_ERROR("Accessing getChild in FoldFunction of invalid type\n");
  }

  FoldFunction<Dtype>* getFunc() { return this; }

  virtual FoldFunction<Dtype>* getFoldFunc(
      const std::deque<int64_t>& fold_dim_indices, size_t idx) {
    DT_ERROR("Accessing getFoldFunc in base class is illegal\n");
  }

  virtual std::vector<FoldFunction<Dtype>*>& getChildren() {
    DT_ERROR("Accessing getChildren in FoldFunction of invalid type\n");
  }

  virtual void insertFunc(FoldFunction<Dtype>* new_func) {
    DT_ERROR("Accessing insertFunc in FoldFunction of invalid type\n");
  }

  // access function, takes N dimensional folding indices as variadic variable
  template <typename... Args, class Enable = std::enable_if_t<(
                                  ... && std::is_convertible_v<Args, int64_t>)>>
  void insertData(const Dtype& new_data, const Args&... list) {
    std::deque<int64_t> dim_indices{list...};
    this->insertData(new_data, dim_indices, 0);
  }

  virtual void insertData(const Dtype& new_data,
                          const std::deque<int64_t>& fold_dim_indices,
                          size_t idx) = 0;

  // fold function related virtual functions
  virtual void insertAlpha(const Dtype& new_alpha) {
    DT_ERROR("Accessing insertAlpha in FoldFunction of invalid type\n");
  }

  virtual void insertBeta(const Dtype& new_alpha) {
    DT_ERROR("Accessing insertBeta in FoldFunction of invalid type\n");
  }

  virtual void insertWkSplitParam(WkSplitParam& new_params) {
    DT_ERROR("Accessing insertWkSplitParam in FoldFunction of invalid type\n");
  }

  virtual const WkSplitParam& getWkSplitParam() const {
    DT_ERROR("Accessing insertWkSplitParam in FoldFunction of invalid type\n");
  }
  virtual WkSplitParam& getWkSplitParamMutable() {
    DT_ERROR("Accessing insertWkSplitParam in FoldFunction of invalid type\n");
  }

  virtual void printMetaData(std::ostream& out, std::string ps = "") const {
    DT_ERROR("Accessing printMetaData in FoldFunction of invalid type\n");
  }
};

/**
 * @brief Non-leaf Constant fold function of templated Dtype.
 * Constant Fold Function findex) that returns child_ff_ (fold function)
 * independent of index
 * Get Method : getData(a2,.., aN) --> returns child_ff_->getData(a2,..,
 * aN-1)
 * @tparam Dtype
 */

template <typename Dtype>
class ConstFoldFunction_NonLeaf : public FoldFunction<Dtype> {
 public:
  ConstFoldFunction_NonLeaf(FoldFunction<Dtype>* new_child)
      : FoldFunction<Dtype>(FoldFunction<Dtype>::FuncType::Constant_nonleaf),
        child_ff_(new_child) {}
  virtual ~ConstFoldFunction_NonLeaf() {}
  Dtype getData(const std::deque<int64_t>& fold_dim_indices,
                size_t idx) const override {
    DT_CHECK(idx < fold_dim_indices.size() - 1);
    return child_ff_->getData(fold_dim_indices, idx + 1);
  }

  void insertData(const Dtype& new_data,
                  const std::deque<int64_t>& fold_dim_indices,
                  size_t idx) override {
    DT_CHECK(idx < fold_dim_indices.size() - 1);
    child_ff_->insertData(new_data, fold_dim_indices, idx + 1);
  }

  FoldFunction<Dtype>* getChild() override { return child_ff_; }
  void insertFunc(FoldFunction<Dtype>* new_func) override {
    child_ff_ = new_func;
  }

  FoldFunction<Dtype>* getFoldFunc(const std::deque<int64_t>& fold_dim_indices,
                                   size_t idx) override {
    DT_CHECK(idx < fold_dim_indices.size() - 1);
    return child_ff_->getFoldFunc(fold_dim_indices, idx + 1);
  }

 private:
  FoldFunction<Dtype>* child_ff_;
};

/**
 * @brief Constant Fold Function f(index) that return data independent of
 * index
 * Get Method : getData(a1) --> returns data_
 * @tparam Dtype
 */
template <typename Dtype>
class ConstFoldFunction_Leaf : public FoldFunction<Dtype> {
 public:
  ConstFoldFunction_Leaf(const Dtype& d)
      : FoldFunction<Dtype>(FoldFunction<Dtype>::FuncType::Constant_leaf),
        data_(d) {}
  ConstFoldFunction_Leaf()
      : FoldFunction<Dtype>(FoldFunction<Dtype>::FuncType::Constant_leaf) {}
  virtual ~ConstFoldFunction_Leaf() {}
  Dtype getData(const std::deque<int64_t>& fold_dim_indices,
                size_t idx) const override {
    return data_;
  }
  void insertData(const Dtype& new_data,
                  const std::deque<int64_t>& fold_dim_indices,
                  size_t idx) override {
    data_ = new_data;
  }

  FoldFunction<Dtype>* getFoldFunc(const std::deque<int64_t>& fold_dim_indices,
                                   size_t idx) override {
    return this;
  }

 private:
  Dtype data_{};
};

/**
 * @brief Implements affine fold function f(a2,.., aN)
 * Return  alpha_ * aN + beta + child_ff_.getData(a2,..,aN)
 *
 * @tparam Dtype
 */

template <typename Dtype>
class AffineFoldFunction_NonLeaf : public FoldFunction<Dtype> {
 public:
  AffineFoldFunction_NonLeaf(Dtype alpha, Dtype beta,
                             FoldFunction<Dtype>* new_child)
      : FoldFunction<Dtype>(FoldFunction<Dtype>::FuncType::Affine_nonleaf),
        alpha_(alpha),
        beta_(beta),
        child_ff_(new_child) {
    if (!(std::is_arithmetic<Dtype>::value ||
          std::is_same<Dtype, std::pair<int64_t, int64_t>>::value ||
          std::is_same<Dtype, std::vector<std::pair<int64_t, int64_t>>>::value))
      DT_ERROR(
          "Affine Fold can only be of type arithmetic, std::pair<int64_t, "
          "int64_t>, or std::vector<std::pair<int64_t, int64_t>>\n");
  }

  AffineFoldFunction_NonLeaf(Dtype alpha, Dtype beta)
      : FoldFunction<Dtype>(FoldFunction<Dtype>::FuncType::Affine_nonleaf),
        alpha_(alpha),
        beta_(beta) {
    if (!(std::is_arithmetic<Dtype>::value ||
          std::is_same<Dtype, std::pair<int64_t, int64_t>>::value ||
          std::is_same<Dtype, std::vector<std::pair<int64_t, int64_t>>>::value))
      DT_ERROR(
          "Affine Fold can only be of type arithmetic, std::pair<int64_t, "
          "int64_t>, or std::vector<std::pair<int64_t, int64_t>>\n");
  }

  AffineFoldFunction_NonLeaf(FoldFunction<Dtype>* new_child)
      : FoldFunction<Dtype>(FoldFunction<Dtype>::FuncType::Affine_nonleaf),
        child_ff_(new_child) {
    if (!(std::is_arithmetic<Dtype>::value ||
          std::is_same<Dtype, std::pair<int64_t, int64_t>>::value ||
          std::is_same<Dtype, std::vector<std::pair<int64_t, int64_t>>>::value))
      DT_ERROR(
          "Affine Fold can only be of type arithmetic, std::pair<int64_t, "
          "int64_t>, or std::vector<std::pair<int64_t, int64_t>>\n");
  }
  virtual ~AffineFoldFunction_NonLeaf() {}

  Dtype getData(const std::deque<int64_t>& fold_dim_indices,
                size_t idx) const override {
    return getDataAffine(fold_dim_indices, idx);
  }

  template <
      typename T = Dtype,
      std::enable_if_t<
          !(std::is_arithmetic<T>::value ||
            std::is_same<T, std::pair<int64_t, int64_t>>::value ||
            std::is_same<T, std::vector<std::pair<int64_t, int64_t>>>::value),
          bool> = true>
  Dtype getDataAffine(const std::deque<int64_t>& fold_dim_indices,
                      size_t idx) const {
    DT_CHECK(idx < fold_dim_indices.size() - 1);
    DT_ERROR("Affine Fold can only be of type arithmetic\n");
    Dtype val;
    return val;
  }

  template <typename T = Dtype,
            std::enable_if_t<std::is_arithmetic<T>::value, bool> = true>
  Dtype getDataAffine(const std::deque<int64_t>& fold_dim_indices,
                      size_t idx) const {
    DT_CHECK(idx < fold_dim_indices.size() - 1);
    auto dim_index = fold_dim_indices.at(idx);
    if (!std::is_arithmetic<decltype(child_ff_->getData(fold_dim_indices,
                                                        idx + 1))>::value)
      DT_ERROR("Affine Fold can only be of type arithmetic\n");
    return (alpha_ * dim_index + beta_) +
           child_ff_->getData(fold_dim_indices, idx + 1);
  }

  template <
      typename T = Dtype,
      std::enable_if_t<std::is_same<T, std::pair<int64_t, int64_t>>::value,
                       bool> = true>
  Dtype getDataAffine(const std::deque<int64_t>& fold_dim_indices,
                      size_t idx) const {
    DT_CHECK(idx < fold_dim_indices.size() - 1);
    auto dim_index = fold_dim_indices.at(idx);
    auto child_data = child_ff_->getData(fold_dim_indices, idx + 1);
    return {(alpha_.first * dim_index + beta_.first) + child_data.first,
            (alpha_.second * dim_index + beta_.second) + child_data.second};
  }

  template <
      typename T = Dtype,
      std::enable_if_t<
          std::is_same<T, std::vector<std::pair<int64_t, int64_t>>>::value,
          bool> = true>
  Dtype getDataAffine(const std::deque<int64_t>& fold_dim_indices,
                      size_t idx) const {
    DT_ERROR(
        "AffineFoldFunction_NonLeaf::getDataAffine: Not yet implemented for "
        "Dtype=std::vector<std::pair<int64_t, int64_t>>");
  }

  FoldFunction<Dtype>* getChild() override { return child_ff_; }
  Dtype getAlpha() const { return alpha_; }
  Dtype getBeta() const { return beta_; }
  void insertAlpha(const Dtype& new_alpha) override { alpha_ = new_alpha; }
  void insertBeta(const Dtype& new_beta) override { beta_ = new_beta; }
  void insertFunc(FoldFunction<Dtype>* new_func) override {
    child_ff_ = new_func;
  }

  void insertData(const Dtype& new_data,
                  const std::deque<int64_t>& fold_dim_indices,
                  size_t idx) override {
    DT_CHECK(idx < fold_dim_indices.size() - 1);
    child_ff_->insertData(new_data, fold_dim_indices, idx + 1);
  }

  FoldFunction<Dtype>* getFoldFunc(const std::deque<int64_t>& fold_dim_indices,
                                   size_t idx) override {
    DT_CHECK(idx < fold_dim_indices.size() - 1);
    return child_ff_->getFoldFunc(fold_dim_indices, idx + 1);
  }

  void printMetaData(std::ostream& out, std::string ps = "") const override {
    printMetaDataAffine(out, ps);
  }

  template <
      typename T = Dtype,
      std::enable_if_t<
          !(std::is_arithmetic<T>::value ||
            std::is_same<T, std::pair<int64_t, int64_t>>::value ||
            std::is_same<T, std::vector<std::pair<int64_t, int64_t>>>::value),
          bool> = true>
  void printMetaDataAffine(std::ostream& out, std::string ps = "") const {
    DT_ERROR("Unsupported");
  }

  /**
   * @brief prints meta data
   *
   * @tparam T
   * @param out
   * @param ps
   */
  template <typename T = Dtype,
            std::enable_if_t<std::is_arithmetic<T>::value, bool> = true>
  void printMetaDataAffine(std::ostream& out, std::string ps = "") const {
    auto QUOTE = [](std::string str) { return "\"" + str + "\""; };
    out << ps << QUOTE("alpha_") << " : " << alpha_ << ", ";
    out << ps << QUOTE("beta_") << " : " << beta_;
  }

  template <
      typename T = Dtype,
      std::enable_if_t<std::is_same<T, std::pair<int64_t, int64_t>>::value,
                       bool> = true>
  void printMetaDataAffine(std::ostream& out, std::string ps = "") const {
    auto QUOTE = [](std::string str) { return "\"" + str + "\""; };
    out << ps << QUOTE("alpha_") << " : { " << alpha_.first << ", "
        << alpha_.second << "}, ";
    out << ps << QUOTE("beta_") << " : {" << beta_.first << ", " << beta_.second
        << "}";
  }

  template <
      typename T = Dtype,
      std::enable_if_t<
          std::is_same<T, std::vector<std::pair<int64_t, int64_t>>>::value,
          bool> = true>
  void printMetaDataAffine(std::ostream& out, std::string ps = "") const {
    auto QUOTE = [](std::string str) { return "\"" + str + "\""; };
  }

 private:
  Dtype alpha_{};
  Dtype beta_{};
  FoldFunction<Dtype>* child_ff_;
};

/**
 * @brief Implements affine fold function f(a1)
 * Return  alpha_ * a1 + beta
 *
 * @tparam Dtype
 */

template <typename Dtype>
class AffineFoldFunction_Leaf : public FoldFunction<Dtype> {
 public:
  AffineFoldFunction_Leaf(Dtype alpha, Dtype beta)
      : FoldFunction<Dtype>(FoldFunction<Dtype>::FuncType::Affine_leaf),
        alpha_(alpha),
        beta_(beta) {
    if (!(std::is_arithmetic<Dtype>::value ||
          std::is_same<Dtype, std::pair<int64_t, int64_t>>::value ||
          std::is_same<Dtype, std::vector<std::pair<int64_t, int64_t>>>::value))
      DT_ERROR(
          "Affine Fold can only be of type arithmetic, std::pair<int64_t, "
          "int64_t>, or std::vector<std_pair<int64_t, int64_t>>\n");
  }

  AffineFoldFunction_Leaf()
      : FoldFunction<Dtype>(FoldFunction<Dtype>::FuncType::Affine_leaf) {
    if (!(std::is_arithmetic<Dtype>::value ||
          std::is_same<Dtype, std::pair<int64_t, int64_t>>::value ||
          std::is_same<Dtype, std::vector<std::pair<int64_t, int64_t>>>::value))
      DT_ERROR(
          "Affine Fold can only be of type arithmetic, std::pair<int64_t, "
          "int64_t>, or std::vector<std::pair<int64_t, int64_t>>\n");
  }

  virtual ~AffineFoldFunction_Leaf() {}

  Dtype getData(const std::deque<int64_t>& fold_dim_indices,
                size_t idx) const override {
    return getDataAffine(fold_dim_indices, idx);
  }

  template <
      typename T = Dtype,
      std::enable_if_t<
          !(std::is_arithmetic<T>::value ||
            std::is_same<T, std::pair<int64_t, int64_t>>::value ||
            std::is_same<T, std::vector<std::pair<int64_t, int64_t>>>::value),
          bool> = true>
  Dtype getDataAffine(const std::deque<int64_t>& fold_dim_indices,
                      size_t idx) const {
    DT_CHECK(idx == fold_dim_indices.size() - 1);
    auto dim_index = fold_dim_indices.at(idx);
    return alpha_;
  }

  template <typename T = Dtype,
            std::enable_if_t<std::is_arithmetic<T>::value, bool> = true>
  Dtype getDataAffine(const std::deque<int64_t>& fold_dim_indices,
                      size_t idx) const {
    DT_CHECK(idx == fold_dim_indices.size() - 1);
    auto dim_index = fold_dim_indices.at(idx);
    return alpha_ * dim_index + beta_;
  }

  template <
      typename T = Dtype,
      std::enable_if_t<std::is_same<T, std::pair<int64_t, int64_t>>::value,
                       bool> = true>
  Dtype getDataAffine(const std::deque<int64_t>& fold_dim_indices,
                      size_t idx) const {
    DT_CHECK(idx == fold_dim_indices.size() - 1);
    auto dim_index = fold_dim_indices.at(idx);
    return {alpha_.first * dim_index + beta_.first,
            alpha_.second * dim_index + beta_.second};
  }

  template <
      typename T = Dtype,
      std::enable_if_t<
          std::is_same<T, std::vector<std::pair<int64_t, int64_t>>>::value,
          bool> = true>
  Dtype getDataAffine(const std::deque<int64_t>& fold_dim_indices,
                      size_t idx) const {
    DT_ERROR(
        "AffineFoldFunction_Leaf::getDataAffine: Not yet implemented for "
        "Dtype=std::vector<std::pair<int64_t, int64_t>>");
  }

  Dtype getAlpha() const { return alpha_; }
  Dtype getBeta() const { return beta_; }
  void insertAlpha(const Dtype& new_alpha) override { alpha_ = new_alpha; }
  void insertBeta(const Dtype& new_beta) override { beta_ = new_beta; }

  void insertData(const Dtype& new_data,
                  const std::deque<int64_t>& fold_dim_indices,
                  size_t idx) override {
    // ignored
  }

  FoldFunction<Dtype>* getFoldFunc(const std::deque<int64_t>& fold_dim_indices,
                                   size_t idx) override {
    return this;
  }

  /**
   * @brief prints meta data
   *
   * @tparam T
   * @param out
   * @param ps
   */
  void printMetaData(std::ostream& out, std::string ps = "") const override {
    printMetaDataAffine(out, ps);
  }

  template <
      typename T = Dtype,
      std::enable_if_t<
          !(std::is_arithmetic<T>::value ||
            std::is_same<T, std::pair<int64_t, int64_t>>::value ||
            std::is_same<T, std::vector<std::pair<int64_t, int64_t>>>::value),
          bool> = true>
  void printMetaDataAffine(std::ostream& out, std::string ps = "") const {
    DT_ERROR("Unsupported");
  }

  template <typename T = Dtype,
            std::enable_if_t<std::is_arithmetic<T>::value, bool> = true>
  void printMetaDataAffine(std::ostream& out, std::string ps = "") const {
    auto QUOTE = [](std::string str) { return "\"" + str + "\""; };
    out << ps << QUOTE("alpha_") << " : " << alpha_ << ", ";
    out << ps << QUOTE("beta_") << " : " << beta_;
  }

  template <
      typename T = Dtype,
      std::enable_if_t<std::is_same<T, std::pair<int64_t, int64_t>>::value,
                       bool> = true>
  void printMetaDataAffine(std::ostream& out, std::string ps = "") const {
    auto QUOTE = [](std::string str) { return "\"" + str + "\""; };
    out << ps << QUOTE("alpha_") << " : { " << alpha_.first << ", "
        << alpha_.second << "}, ";
    out << ps << QUOTE("beta_") << " : {" << beta_.first << ", " << beta_.second
        << "}";
  }

  template <
      typename T = Dtype,
      std::enable_if_t<
          std::is_same<T, std::vector<std::pair<int64_t, int64_t>>>::value,
          bool> = true>
  void printMetaDataAffine(std::ostream& out, std::string ps = "") const {
    auto QUOTE = [](std::string str) { return "\"" + str + "\""; };
  }

 private:
  Dtype alpha_{};
  Dtype beta_{};
};

/**
 * @brief Implements 1D Map fold function f(a1, a2,.., aN)
 * Return  child_ffs_.at(a2,..,aN)
 *
 * @tparam Dtype
 */

template <typename Dtype>
class MapFoldFunction_NonLeaf : public FoldFunction<Dtype> {
 public:
  MapFoldFunction_NonLeaf(int dim_size)
      : FoldFunction<Dtype>(FoldFunction<Dtype>::FuncType::Map_nonleaf) {
    child_ff_vec_.resize(dim_size);
    for (int idx = 0; idx < dim_size; idx++) child_ff_vec_.at(idx) = nullptr;
  }
  MapFoldFunction_NonLeaf(std::vector<FoldFunction<Dtype>*> ff_ptrs)
      : FoldFunction<Dtype>(FoldFunction<Dtype>::FuncType::Map_nonleaf),
        child_ff_vec_(ff_ptrs) {}
  virtual ~MapFoldFunction_NonLeaf() {}
  Dtype getData(const std::deque<int64_t>& fold_dim_indices,
                size_t idx) const override {
    DT_CHECK(idx < fold_dim_indices.size() - 1);
    auto dim_index = fold_dim_indices.at(idx);
    DT_CHECK(child_ff_vec_.size() > dim_index);
    return child_ff_vec_.at(dim_index)->getData(fold_dim_indices, idx + 1);
  }
  std::vector<FoldFunction<Dtype>*>& getChildren() override {
    return child_ff_vec_;
  }

  void insertData(const Dtype& new_data,
                  const std::deque<int64_t>& fold_dim_indices,
                  size_t idx) override {
    DT_CHECK(idx < fold_dim_indices.size() - 1);
    auto dim_index = fold_dim_indices.at(idx);
    DT_CHECK(child_ff_vec_.size() > dim_index);
    child_ff_vec_.at(dim_index)->insertData(new_data, fold_dim_indices,
                                            idx + 1);
  }

  FoldFunction<Dtype>* getFoldFunc(const std::deque<int64_t>& fold_dim_indices,
                                   size_t idx) override {
    DT_CHECK(idx < fold_dim_indices.size() - 1);
    auto dim_index = fold_dim_indices.at(idx);
    DT_CHECK(child_ff_vec_.size() > dim_index);
    return child_ff_vec_.at(dim_index)->getFoldFunc(fold_dim_indices, idx + 1);
  }

 private:
  std::vector<FoldFunction<Dtype>*> child_ff_vec_;
};

/**
 * @brief Implements 1D Map fold function f(a1)
 * Return  data_vec_.at(a1,..,aN-1)
 *
 * @tparam Dtype
 */

template <typename Dtype>
class MapFoldFunction_Leaf : public FoldFunction<Dtype> {
 public:
  MapFoldFunction_Leaf(int dim_size)
      : FoldFunction<Dtype>(FoldFunction<Dtype>::FuncType::Map_leaf),
        data_vec_(dim_size) {}

  MapFoldFunction_Leaf(int dim_size, const Dtype& new_data)
      : FoldFunction<Dtype>(FoldFunction<Dtype>::FuncType::Map_leaf),
        data_vec_(dim_size, new_data) {}

  MapFoldFunction_Leaf(const std::vector<Dtype>& new_data_vec)
      : FoldFunction<Dtype>(FoldFunction<Dtype>::FuncType::Map_leaf),
        data_vec_(new_data_vec) {}

  virtual ~MapFoldFunction_Leaf() {}
  Dtype getData(const std::deque<int64_t>& fold_dim_indices,
                size_t idx) const override {
    DT_CHECK(idx == fold_dim_indices.size() - 1);
    auto dim_index = fold_dim_indices.at(idx);
    DT_CHECK(data_vec_.size() > dim_index);
    return data_vec_.at(dim_index);
  }

  void insertData(const Dtype& new_data,
                  const std::deque<int64_t>& fold_dim_indices,
                  size_t idx) override {
    DT_CHECK(idx == fold_dim_indices.size() - 1);
    auto dim_index = fold_dim_indices.at(idx);
    DT_CHECK(data_vec_.size() > dim_index);
    data_vec_.at(dim_index) = new_data;
  }

  int getSize() const { return data_vec_.size(); }
  std::vector<Dtype>& getDataVec() { return data_vec_; }
  const std::vector<Dtype>& getDataVec() const { return data_vec_; }

  FoldFunction<Dtype>* getFoldFunc(const std::deque<int64_t>& fold_dim_indices,
                                   size_t idx) override {
    return this;
  }

 private:
  std::vector<Dtype> data_vec_;
};

/**
 * @brief Implements work-split (wkSplit) fold function f(a1)
 * Return  wksplit_param_.getSize(a1) if Dtype is arithmetic else returns
 * wksplit_param_.getCoordVec(a1) if Dtype is std::vector<std::pair<int64_t,
 * int64_t>>
 *
 * @tparam Dtype
 */

template <typename Dtype>
class WkSplitFoldFunction_Leaf : public FoldFunction<Dtype> {
 public:
  WkSplitFoldFunction_Leaf(const WkSplitParam& wksplit_param)
      : FoldFunction<Dtype>(FoldFunction<Dtype>::FuncType::WkSplit_leaf),
        wksplit_param_(wksplit_param) {
    if (!(std::is_arithmetic<Dtype>::value ||
          std::is_same<Dtype, std::vector<std::pair<int64_t, int64_t>>>::value))
      DT_ERROR(
          "WkSplit Fold can only be of type arithmetic or "
          "std::vector<std::pair<int64_t, int64_t>>\n");
  }

  WkSplitFoldFunction_Leaf()
      : FoldFunction<Dtype>(FoldFunction<Dtype>::FuncType::WkSplit_leaf) {
    if (!(std::is_arithmetic<Dtype>::value ||
          std::is_same<Dtype, std::vector<std::pair<int64_t, int64_t>>>::value))
      DT_ERROR(
          "WkSplit Fold can only be of type arithmetic or "
          "std::vector<std::pair<int64_t, int64_t>>\n");
  }

  virtual ~WkSplitFoldFunction_Leaf() {}

  template <
      typename T = Dtype,
      std::enable_if_t<
          !(std::is_arithmetic<T>::value ||
            std::is_same<T, std::vector<std::pair<int64_t, int64_t>>>::value),
          bool> = true>
  Dtype getFoldedData(int64_t dim_index) const {
    DT_ERROR(
        "WkSplit Fold can only be of type arithmetic or "
        "std::vector<std::pair<int64_t, int64_t>>\n");
    Dtype temp_data;
    return temp_data;
  }

  template <
      typename T = Dtype,
      std::enable_if_t<
          (std::is_same<T, std::vector<std::pair<int64_t, int64_t>>>::value),
          bool> = true>
  Dtype getFoldedData(int64_t dim_index) const {
    return wksplit_param_.getCoordVec(dim_index);
  }

  template <typename T = Dtype,
            std::enable_if_t<std::is_arithmetic<T>::value, bool> = true>
  Dtype getFoldedData(int64_t dim_index) const {
    return (Dtype)wksplit_param_.getSize(dim_index);
  }

  Dtype getData(const std::deque<int64_t>& fold_dim_indices,
                size_t idx) const override {
    DT_CHECK(wksplit_param_.isBuilt());
    DT_CHECK(idx == fold_dim_indices.size() - 1);
    auto dim_index = fold_dim_indices.at(idx);
    return getFoldedData(dim_index);
  }

  const WkSplitParam& getWkSplitParam() const override {
    return wksplit_param_;
  }
  WkSplitParam& getWkSplitParamMutable() override { return wksplit_param_; }
  void insertWkSplitParam(WkSplitParam& new_params) override {
    wksplit_param_.build(new_params);
  }

  void insertData(const Dtype& new_data,
                  const std::deque<int64_t>& fold_dim_indices,
                  size_t idx) override {
    DT_ERROR("Illegal use of insertData on WkSplitFoldFunction_Leaf");
  }

  FoldFunction<Dtype>* getFoldFunc(const std::deque<int64_t>& fold_dim_indices,
                                   size_t idx) override {
    return this;
  }

  /**
   * @brief prints meta data
   *
   * @param out
   * @param ps
   */
  void printMetaData(std::ostream& out, std::string ps = "") const override {
    wksplit_param_.printMetaData(out, ps);
  }

 private:
  WkSplitParam wksplit_param_{};
};

using fm_dim_prop = std::vector<std::pair<const FoldDimProp*, BaseFuncType>>;

/**
 * @brief Manages storage and access N dimensional folded variable of generic
 * Dtype
 *
 * @tparam Dtype
 */

template <typename Dtype>
class FoldManager {
 public:
  // constructor : by default a constant zeroth fold is created
  FoldManager() {
    parent_func_ =
        static_cast<FoldFunction<Dtype>*>(new ConstFoldFunction_Leaf<Dtype>());
  }
  FoldManager(const Dtype& data) {
    parent_func_ = static_cast<FoldFunction<Dtype>*>(
        new ConstFoldFunction_Leaf<Dtype>(data));
  }

  FoldManager(const FoldManager<Dtype>& rhs) { clone(rhs); }

  FoldManager(FoldManager<Dtype>&&) noexcept = default;  // allow move ctor

  ~FoldManager() { clear(); }

  /**
   * @brief This operators copies the fold space of "rhs".
   *
   * @param rhs
   * @return FoldManager<Dtype>&
   */
  FoldManager<Dtype>& operator=(const FoldManager<Dtype>& rhs) {
    if (rhs.dim_prop_.size() != this->dim_prop_.size())
      DT_ERROR(
          "Fold space dimensionality of rhs and lhs variables should "
          "match\n");

    std::deque<std::pair<const FoldDimProp*, BaseFuncType>> new_dim_prop;
    for (int idx = 0; idx < rhs.dim_prop_.size(); idx++) {
      if (rhs.dim_prop_.at(idx).first->getSize() !=
          this->dim_prop_.at(idx).first->getSize())
        DT_ERROR("Cardinality mis-match at dimension " + std::to_string(idx) +
                 "\n");
      else {
        new_dim_prop.push_back(
            {this->dim_prop_.at(idx).first, rhs.dim_prop_.at(idx).second});
      }
    }

    if (this->parent_func_ != nullptr) {
      deleteSubTree(this->parent_func_);
      this->parent_func_ = nullptr;
    }
    this->dim_prop_.clear();

    // special case for zeroth order fold
    if (rhs.dim_prop_.size() == 0) {
      DT_CHECK(rhs.parent_func_->Type() ==
               FoldFunction<Dtype>::FuncType::Constant_leaf);
      this->parent_func_ = static_cast<FoldFunction<Dtype>*>(
          new ConstFoldFunction_Leaf<Dtype>(rhs.getData()));
      return *this;
    }

    // normal case
    // build bottom-up, i.e., leaf to non-leaf
    for (int idx = new_dim_prop.size() - 1; idx >= 0; idx--) {
      int pos = 0;
      this->buildDim(new_dim_prop.at(idx).first, new_dim_prop.at(idx).second,
                     pos);
    }

    // copy data
    // step 1 : get linear list for rhs and this
    std::vector<FoldFunction<Dtype>*> linear_ff_list_rhs;
    std::vector<FoldFunction<Dtype>*> linear_ff_list_this;

    rhs.getLinearFuncList(linear_ff_list_rhs);
    this->getLinearFuncList(linear_ff_list_this);

    // step 2 go through the list
    copy(linear_ff_list_this, linear_ff_list_rhs);

    return *this;
  }

  /**
   * @brief This method destroys "this" FM and creates a new one by cloning from
   * rhs. Dimensions at index specified by key of ignore_dim_idx_and_dim_value
   * is ignored and it's coordinate is fixed to value specified by
   * ignore_dim_idx_and_dim_val
   *
   * @param rhs
   * @param ignore_dim_idx_and_dim_val
   * @return None
   */
  void clone(const FoldManager<Dtype>& rhs,
             const std::map<int, int>& ignore_dim_idx_and_dim_val = {}) {
    this->clear();

    // special case for zeroth order fold
    if (rhs.dim_prop_.size() == 0) {
      DT_CHECK(rhs.parent_func_->Type() ==
               FoldFunction<Dtype>::FuncType::Constant_leaf);
      this->parent_func_ = static_cast<FoldFunction<Dtype>*>(
          new ConstFoldFunction_Leaf<Dtype>(rhs.getData()));
      return;
    }

    // normal case
    // build bottom-up, i.e., leaf to non-leaf
    for (int idx = rhs.dim_prop_.size() - 1; idx >= 0; idx--) {
      if (ignore_dim_idx_and_dim_val.count(idx)) continue;
      int pos = 0;
      this->buildDim(rhs.dim_prop_.at(idx).first, rhs.dim_prop_.at(idx).second,
                     pos);
    }

    // copy data
    // step 1 : get linear list for rhs and this
    std::vector<FoldFunction<Dtype>*> linear_ff_list_rhs;
    std::vector<FoldFunction<Dtype>*> linear_ff_list_this;

    rhs.getLinearFuncList(linear_ff_list_rhs, ignore_dim_idx_and_dim_val);
    this->getLinearFuncList(linear_ff_list_this);

    // step 2 go through the list
    copy(linear_ff_list_this, linear_ff_list_rhs);
  }

  /**
   * @brief This method destroys "this" FM and initializes it as new
   *
   * @return None
   */
  // void reset() {
  //   deleteSubTree(this->parent_func_);
  //   this->parent_func_ = nullptr;
  //   this->dim_prop_;
  //   this->FoldManager<Dtype>();
  // }
  template <typename... Args>
  void reset(Args&&... args) {
    static_assert(!std::has_virtual_destructor<FoldManager<Dtype>>::value,
                  "Unsafe");
    this->~FoldManager<Dtype>();
    new (this) FoldManager<Dtype>(std::forward<Args>(args)...);
  }

  /**
   * @brief This method copies linear_ff_list_rhs to linear_ff_list_lhs
   *
   * @param linear_ff_list_lhs
   * @param linear_ff_list_rhs
   */
  static void copy(std::vector<FoldFunction<Dtype>*>& linear_ff_list_lhs,
                   std::vector<FoldFunction<Dtype>*>& linear_ff_list_rhs) {
    if (linear_ff_list_rhs.size() != linear_ff_list_lhs.size())
      DT_ERROR("Unexpected");

    while (linear_ff_list_rhs.size()) {
      auto rhs_ff = linear_ff_list_rhs.back();
      auto this_ff = linear_ff_list_lhs.back();
      linear_ff_list_rhs.pop_back();
      linear_ff_list_lhs.pop_back();

      if (rhs_ff->Type() == FoldFunction<Dtype>::FuncType::Constant_leaf) {
        this_ff->insertData(rhs_ff->getData());
      } else if (rhs_ff->Type() == FoldFunction<Dtype>::FuncType::Affine_leaf) {
        auto this_leaf = dynamic_cast<AffineFoldFunction_Leaf<Dtype>*>(this_ff);
        auto rhs_leaf = dynamic_cast<AffineFoldFunction_Leaf<Dtype>*>(rhs_ff);
        auto alpha = rhs_leaf->getAlpha();
        auto beta = rhs_leaf->getBeta();
        this_leaf->insertAlpha(alpha);
        this_leaf->insertBeta(beta);
      } else if (rhs_ff->Type() ==
                 FoldFunction<Dtype>::FuncType::WkSplit_leaf) {
        auto this_leaf =
            dynamic_cast<WkSplitFoldFunction_Leaf<Dtype>*>(this_ff);
        auto rhs_leaf = dynamic_cast<WkSplitFoldFunction_Leaf<Dtype>*>(rhs_ff);
        this_leaf->insertWkSplitParam(rhs_leaf->getWkSplitParamMutable());
      } else if (rhs_ff->Type() ==
                 FoldFunction<Dtype>::FuncType::Affine_nonleaf) {
        auto this_nonleaf =
            dynamic_cast<AffineFoldFunction_NonLeaf<Dtype>*>(this_ff);
        auto rhs_nonleaf =
            dynamic_cast<AffineFoldFunction_NonLeaf<Dtype>*>(rhs_ff);
        auto alpha = rhs_nonleaf->getAlpha();
        auto beta = rhs_nonleaf->getBeta();
        this_nonleaf->insertAlpha(alpha);
        this_nonleaf->insertBeta(beta);
      } else if (rhs_ff->Type() == FoldFunction<Dtype>::FuncType::Map_leaf) {
        auto this_leaf = dynamic_cast<MapFoldFunction_Leaf<Dtype>*>(this_ff);
        auto rhs_leaf = dynamic_cast<MapFoldFunction_Leaf<Dtype>*>(rhs_ff);
        this_leaf->getDataVec() = rhs_leaf->getDataVec();
      }  // else : do nothing
    }
  }

  bool operator!=(const FoldManager<Dtype>& rhs) const {
    return !(*this == rhs);
  }

  /**
   * @brief Comparision operator : This operators compare with rhs fold tree
   *
   * @param rhs
   * @return true
   * @return false
   */
  bool operator==(const FoldManager<Dtype>& rhs) const {
    if (rhs.dim_prop_.size() != this->dim_prop_.size()) return false;

    for (int idx = 0; idx < rhs.dim_prop_.size(); idx++) {
      if (rhs.dim_prop_.at(idx).first->getSize() !=
              this->dim_prop_.at(idx).first->getSize() ||
          rhs.dim_prop_.at(idx).second != this->dim_prop_.at(idx).second)
        return false;
    }

    // special case for zeroth order fold
    if (rhs.dim_prop_.size() == 0) {
      DT_CHECK(rhs.parent_func_->Type() ==
               FoldFunction<Dtype>::FuncType::Constant_leaf);

      if (rhs.getData() != this->getData()) return false;
    }

    // step 1 : get linear list for rhs and this
    std::vector<FoldFunction<Dtype>*> linear_ff_list_rhs;
    std::vector<FoldFunction<Dtype>*> linear_ff_list_this;

    rhs.getLinearFuncList(linear_ff_list_rhs);
    this->getLinearFuncList(linear_ff_list_this);

    // step 2 go through the list
    if (linear_ff_list_rhs.size() != linear_ff_list_this.size())
      DT_ERROR("Unexpected");

    while (linear_ff_list_rhs.size()) {
      auto rhs_ff = linear_ff_list_rhs.back();
      auto this_ff = linear_ff_list_this.back();
      linear_ff_list_rhs.pop_back();
      linear_ff_list_this.pop_back();

      if (rhs_ff->Type() == FoldFunction<Dtype>::FuncType::Constant_leaf) {
        if (rhs_ff->getData() != this_ff->getData()) return false;
      } else if (rhs_ff->Type() == FoldFunction<Dtype>::FuncType::Affine_leaf) {
        auto this_leaf = dynamic_cast<AffineFoldFunction_Leaf<Dtype>*>(this_ff);
        auto rhs_leaf = dynamic_cast<AffineFoldFunction_Leaf<Dtype>*>(rhs_ff);
        if (rhs_leaf->getAlpha() != this_leaf->getAlpha()) return false;
        if (rhs_leaf->getBeta() != this_leaf->getBeta()) return false;
      } else if (rhs_ff->Type() ==
                 FoldFunction<Dtype>::FuncType::WkSplit_leaf) {
        auto this_leaf =
            dynamic_cast<WkSplitFoldFunction_Leaf<Dtype>*>(this_ff);
        auto rhs_leaf = dynamic_cast<WkSplitFoldFunction_Leaf<Dtype>*>(rhs_ff);

        if (!(this_leaf->getWkSplitParam() == rhs_leaf->getWkSplitParam()))
          return false;
      } else if (rhs_ff->Type() ==
                 FoldFunction<Dtype>::FuncType::Affine_nonleaf) {
        auto this_nonleaf =
            dynamic_cast<AffineFoldFunction_NonLeaf<Dtype>*>(this_ff);
        auto rhs_nonleaf =
            dynamic_cast<AffineFoldFunction_NonLeaf<Dtype>*>(rhs_ff);
        auto alpha = rhs_nonleaf->getAlpha();
        auto beta = rhs_nonleaf->getBeta();
        if (rhs_nonleaf->getAlpha() != this_nonleaf->getAlpha()) return false;
        if (rhs_nonleaf->getBeta() != this_nonleaf->getBeta()) return false;
      } else if (rhs_ff->Type() == FoldFunction<Dtype>::FuncType::Map_leaf) {
        auto this_leaf = dynamic_cast<MapFoldFunction_Leaf<Dtype>*>(this_ff);
        auto rhs_leaf = dynamic_cast<MapFoldFunction_Leaf<Dtype>*>(rhs_ff);
        if (this_leaf->getDataVec() != rhs_leaf->getDataVec()) return false;
      }  // else : do nothing
    }

    return true;
  }

  /**
   * @brief This operator copies folded sub space of "rhs". Folded Subspace is
   * determined by ignoring dimensions specified by key of
   * ignore_dim_idx_and_dim_value. The coordinate of each ignored dimensions is
   * specified by value in ignore_dim_idx_and_dim_val.
   *
   * @param rhs
   * @param ignore_dim_idx_and_dim_val
   */
  void copyFoldedSubSpace(
      const FoldManager<Dtype>& rhs,
      const std::map<int, int>& ignore_dim_idx_and_dim_val) {
    if (rhs.dim_prop_.size() - ignore_dim_idx_and_dim_val.size() !=
        this->dim_prop_.size())
      DT_ERROR(
          " Sub fold space dimensionality of rhs and lhs variables should "
          "match\n");

    std::deque<std::pair<const FoldDimProp*, BaseFuncType>> new_dim_prop;
    for (int rhs_idx = 0, lhs_idx = 0; rhs_idx < rhs.dim_prop_.size();
         rhs_idx++) {
      if (ignore_dim_idx_and_dim_val.count(rhs_idx)) {
        if (rhs.dim_prop_.at(rhs_idx).second == BaseFuncType::Affine ||
            rhs.dim_prop_.at(rhs_idx).second == BaseFuncType::WkSplit)
          DT_ERROR(
              "Cannot skip affine and wksplit fold functions so as to preserve "
              "underlying data\n");

        continue;
      }

      if (rhs.dim_prop_.at(rhs_idx).first->getSize() !=
          this->dim_prop_.at(lhs_idx).first->getSize())
        DT_ERROR("Cardinality mis-match at dimension " +
                 std::to_string(rhs_idx) + "\n");
      else {
        new_dim_prop.push_back({this->dim_prop_.at(lhs_idx).first,
                                rhs.dim_prop_.at(rhs_idx).second});
      }
      lhs_idx++;
    }

    this->clear();

    // special case for zeroth order fold
    if (rhs.dim_prop_.size() == 0) {
      DT_CHECK(rhs.parent_func_->Type() ==
               FoldFunction<Dtype>::FuncType::Constant_leaf);
      this->parent_func_ = static_cast<FoldFunction<Dtype>*>(
          new ConstFoldFunction_Leaf<Dtype>(rhs.getData()));
    }

    // normal case
    // build bottom-up, i.e., leaf to non-leaf
    for (int idx = new_dim_prop.size() - 1; idx >= 0; idx--) {
      int pos = 0;
      this->buildDim(new_dim_prop.at(idx).first, new_dim_prop.at(idx).second,
                     pos);
    }

    // copy data
    // step 1 : get linear list for rhs and this
    std::vector<FoldFunction<Dtype>*> linear_ff_list_rhs;
    std::vector<FoldFunction<Dtype>*> linear_ff_list_this;

    rhs.getLinearFuncList(linear_ff_list_rhs, ignore_dim_idx_and_dim_val);
    this->getLinearFuncList(linear_ff_list_this);

    // step 2 go through the list
    copy(linear_ff_list_this, linear_ff_list_rhs);
  }

  /**
   * @brief Applies a custom function to specific data elements in a FoldManager
   * and updates their values.
   *
   * @param posToFixCoord An optional parameter, which is a constant reference
   * to a std::map<int64_t, int64_t>. This map specifies the positions in the
   * FoldManager where the function func should be applied. If not provided, the
   * function will be applied to all data elements in the FoldManager.
   * @param func A reference to a function object that takes the FM Dtype as
   * first argument, an unlimited list of additional arguments, and returns a
   * Dtype.
   * @param funcArgs A variadic reference to additional arguments that will be
   * used in the function func.
   */
  template <class Func, typename... Args>
  void apply(const std::map<int64_t, int64_t>& posToFixCoord, Func&& func,
             Args&&... funcArgs) {
    if (hasZeroFoldDim()) {
      insertData(
          std::forward<Func>(func)(getData(), std::forward<Args>(funcArgs)...));
    } else {
      for (auto& [coord, data] : getDataAndFoldCoordinates(posToFixCoord)) {
        insertData(std::forward<Func>(func)(std::move(data),
                                            std::forward<Args>(funcArgs)...),
                   coord);
      }
    }
  }

  /**
   * @brief Applies a function between the data of the current fold manager and
   * another with the same type.
   *
   * @param rhs The fold manager with the function input data
   * @param posToFixCoord An optional parameter, which is a constant reference
   * to a std::map<int64_t, int64_t>. This map specifies the positions in the
   * FoldManager where the function func should be applied. If not provided, the
   * function will be applied to all data elements in both FoldManagers.
   * @param func A reference to a function object that takes this FM
   * Dtype as first argument, the Dtype of the rhs FM as second input, an
   * unlimited list of additional arguments, and returns a Dtype.
   * @param funcArgs A variadic reference to additional arguments that will be
   * used in the function func.
   */
  template <typename Dtype2, class Func, typename... Args>
  void apply(const FoldManager<Dtype2>& rhs,
             const std::map<int64_t, int64_t>& posToFixCoord, Func&& func,
             Args&&... funcArgs) {
    if (rhs.getFoldSpaceSize() != getFoldSpaceSize()) {
      DT_ERROR("Fold managers are not the same in apply function");
    }
    auto rhsFuncType = rhs.getFuncType();
    auto myFuncType = getFuncType();
    for (int f = 0; f < myFuncType.size(); f++) {
      if ((myFuncType.at(f) == BaseFuncType::Constant &&
           rhsFuncType.at(f) != BaseFuncType::Constant) ||
          (myFuncType.at(f) == BaseFuncType::Affine) ||
          (myFuncType.at(f) == BaseFuncType::WkSplit)) {
        DT_ERROR(
            "Fold managers do not have compatible functype to use apply "
            "function");
      }
    }

    if (hasZeroFoldDim()) {
      insertData(std::forward<Func>(func)(getData(), rhs.getData(),
                                          std::forward<Args>(funcArgs)...));
    } else {
      for (auto& [coord, data] : getDataAndFoldCoordinates(posToFixCoord)) {
        insertData(std::forward<Func>(func)(std::move(data), rhs.getData(coord),
                                            std::forward<Args>(funcArgs)...),
                   coord);
      }
    }
  }

  // build methods..
  void buildConstDim(const FoldDimProp* prop, int pos = 0) {
    buildDim(prop, BaseFuncType::Constant, pos);
  }

  void buildMapDim(const FoldDimProp* prop, int pos = 0) {
    buildDim(prop, BaseFuncType::Map, pos);
  }

  void buildAffineDim(const FoldDimProp* prop, int pos = 0) {
    buildDim(prop, BaseFuncType::Affine, pos);
  }

  void buildWkSplitDim(const FoldDimProp* prop, int pos = 0) {
    buildDim(prop, BaseFuncType::WkSplit, pos);
  }

  // build methods..
  void buildDim(const FoldDimProp* prop, BaseFuncType func_base_type,
                int pos = 0) {
    if (dim_prop_.size() == 0) {
      DT_CHECK(pos == 0);
      if (func_base_type == BaseFuncType::Constant && parent_func_ != nullptr)
        DT_CHECK(parent_func_->Type() ==
                 FoldFunction<Dtype>::FuncType::Constant_leaf);
      else {
        if (parent_func_ != nullptr) delete parent_func_;
        parent_func_ = createLeafFunc(prop, func_base_type);
      }
      dim_prop_.insert(dim_prop_.begin(), {prop, func_base_type});
    } else {
      // insert new node before pos
      if (pos == 0) {
        if (func_base_type == BaseFuncType::Map) {
          // rebuild the sub-tree
          fm_dim_prop dim_prop_sub_tree;
          getAllDimProFromPos(pos, dim_prop_sub_tree);

          // add new fold function in front
          auto old_parent = parent_func_;
          parent_func_ = createNonLeafFunc(prop, func_base_type);
          createSubTreeForEachMapChild(parent_func_, dim_prop_sub_tree);

          auto& children = parent_func_->getChildren();
          for (int idx = 0; idx < children.size(); idx++)
            copySubTree(children.at(idx), old_parent);

          deleteSubTree(old_parent);
        } else {
          // add new fold function in front
          parent_func_ = createNonLeafFunc(prop, func_base_type, parent_func_);
        }
        dim_prop_.insert(dim_prop_.begin(), {prop, func_base_type});
      } else if (pos == dim_prop_.size()) {
        // add new fold function in front
        auto curr_leaf_pos = pos - 1;
        if (curr_leaf_pos > 0) {
          auto last_non_leaf_pos = curr_leaf_pos - 1;
          std::vector<FoldFunction<Dtype>*> ffs_at_last_non_leaf_pos;
          collectFoldFunctionAtLevel(last_non_leaf_pos,
                                     ffs_at_last_non_leaf_pos);

          dim_prop_.push_back({prop, func_base_type});
          fm_dim_prop dim_prop_sub_tree;
          getAllDimProFromPos(curr_leaf_pos, dim_prop_sub_tree);

          for (auto& ff : ffs_at_last_non_leaf_pos) {
            if (ff->Type() == FoldFunction<Dtype>::FuncType::Map_nonleaf) {
              auto& children = ff->getChildren();
              for (int idx = 0; idx < children.size(); idx++) {
                auto curr_child = children.at(idx);
                auto new_child = createTree(dim_prop_sub_tree);
                children.at(idx) = new_child;
                deleteSubTree(curr_child);  // delete  curr child
              }
            } else {
              auto curr_child = ff->getChild();
              auto new_child = createTree(dim_prop_sub_tree);
              ff->insertFunc(new_child);
              deleteSubTree(curr_child);  // delete  curr child
            }
          }
        } else {
          auto old_parent = parent_func_;
          dim_prop_.push_back({prop, func_base_type});
          parent_func_ = createTree(dim_prop_);
          deleteSubTree(old_parent);  // delete  curr child
        }
      } else {
        DT_CHECK(pos < dim_prop_.size());
        // add new fold function in front
        auto pre_leaf_pos = pos - 1;
        std::vector<FoldFunction<Dtype>*> ffs_at_pre_leaf_pos;
        collectFoldFunctionAtLevel(pre_leaf_pos, ffs_at_pre_leaf_pos);
        DT_CHECK(ffs_at_pre_leaf_pos.size());
        dim_prop_.insert(dim_prop_.begin() + pos, {prop, func_base_type});
        fm_dim_prop dim_prop_sub_tree;
        getAllDimProFromPos(pos, dim_prop_sub_tree);

        for (auto& ff : ffs_at_pre_leaf_pos) {
          if (ff->Type() == FoldFunction<Dtype>::FuncType::Map_nonleaf) {
            auto& children = ff->getChildren();
            for (int idx = 0; idx < children.size(); idx++) {
              auto curr_child = children.at(idx);
              auto new_child = createTree(dim_prop_sub_tree);
              children.at(idx) = new_child;

              if (func_base_type == BaseFuncType::Map) {
                // copy data
                DT_CHECK(new_child->Type() ==
                         FoldFunction<Dtype>::FuncType::Map_nonleaf);
                auto& children_new = new_child->getChildren();
                for (int idx = 0; idx < children_new.size(); idx++)
                  copySubTree(children_new.at(idx), curr_child);
              } else {
                copySubTree(new_child->getChild(), curr_child);
              }
              deleteSubTree(curr_child);  // delete  curr child
            }
          } else {
            auto curr_child = ff->getChild();
            auto new_child = createTree(dim_prop_sub_tree);
            ff->insertFunc(new_child);

            if (func_base_type == BaseFuncType::Map) {
              // copy data
              DT_CHECK(new_child->Type() ==
                       FoldFunction<Dtype>::FuncType::Map_nonleaf);
              auto& children_new = new_child->getChildren();
              for (int idx = 0; idx < children_new.size(); idx++)
                copySubTree(children_new.at(idx), curr_child);
            } else {
              copySubTree(new_child->getChild(), curr_child);
            }
            deleteSubTree(curr_child);  // delete  curr child
          }
        }
      }
    }
  }

  /**
   * @brief The method picks the FoldDimProp (managed by the FoldManager) at
   * position pos in deque<dim_prop_> and rebuilds deque<dim_prop_> tree as
   * needed
   *
   * @param func_base_type
   * @param pos
   *  pos = positive number: index  deque<dim_prop_> forwards
   *  pos = negative number : index deque<dim_prop_> backwards
   *    i.e. -1 indicates last element, and -2 the one before..
   *
   * @return bool: true if update/rebuild is successful, false otherwise
   */
  bool rebuildDim(int pos, BaseFuncType func_base_type) {
    if (pos < 0) {
      pos = dim_prop_.size() + pos;
    }
    if (pos >= 0 && pos <= dim_prop_.size() - 1) {
      auto& myDimProp = dim_prop_.at(pos).first;
      rebuildDim(myDimProp, pos, func_base_type);
      return true;
    } else {
      return false;
    }
  }

  /**
   * @brief Methods rebuilds the func tree at pos
   *
   * @param prop
   * @param func_base_type
   * @param pos
   */
  // build methods..
  void rebuildDim(const FoldDimProp* prop, int pos,
                  BaseFuncType func_base_type = BaseFuncType::Unknown) {
    if (pos < 0) {
      pos = dim_prop_.size() + pos;
    }
    DT_CHECK(pos >= 0 && pos <= dim_prop_.size() - 1);
    DT_CHECK(dim_prop_.size() > 0);

    // update
    DT_CHECK(dim_prop_.at(pos).first == prop);
    bool need_to_rebuild = false;
    if (dim_prop_.at(pos).second == BaseFuncType::Map ||
        func_base_type == BaseFuncType::Map ||
        (dim_prop_.at(pos).second != func_base_type &&
         func_base_type != BaseFuncType::Unknown)) {
      need_to_rebuild = true;
    }

    bool can_copy = false;
    if (dim_prop_.at(pos).second != BaseFuncType::Map &&
        (func_base_type != BaseFuncType::Map ||
         (func_base_type == BaseFuncType::Map &&
          dim_prop_.at(pos).second == BaseFuncType::Constant)) &&
        pos != dim_prop_.size() - 1)
      can_copy = true;  // we can only copy if pos fold function was neither a
                        // map nor will be rebuild to a Map (e.g., copy does not
                        // make sense if we are going from Map to const)

    if (func_base_type != BaseFuncType::Unknown)
      dim_prop_.at(pos).second = func_base_type;

    if (need_to_rebuild) {
      if (pos == 0) {
        fm_dim_prop dim_prop_sub_tree;
        getAllDimProFromPos(pos, dim_prop_sub_tree);
        auto old_parent = parent_func_;
        parent_func_ = createTree(dim_prop_sub_tree);
        if (can_copy) {
          if (func_base_type != BaseFuncType::Map) {
            copySubTree(parent_func_->getChild(), old_parent->getChild());
          } else {
            auto& children = parent_func_->getChildren();
            for (int idx = 0; idx < children.size(); idx++)
              copySubTree(children.at(idx), old_parent->getChild());
          }
        }
        deleteSubTree(old_parent);  // delete  old_parent
      } else {
        auto pre_leaf_pos = pos - 1;
        // get all children at pre_pos
        std::vector<FoldFunction<Dtype>*> ffs_at_pre_leaf_pos;
        collectFoldFunctionAtLevel(pre_leaf_pos, ffs_at_pre_leaf_pos);
        DT_CHECK(ffs_at_pre_leaf_pos.size());

        fm_dim_prop dim_prop_sub_tree;
        getAllDimProFromPos(pos, dim_prop_sub_tree);

        for (auto& ff : ffs_at_pre_leaf_pos) {
          if (ff->Type() == FoldFunction<Dtype>::FuncType::Map_nonleaf) {
            auto& pre_node_children = ff->getChildren();
            for (int idx = 0; idx < pre_node_children.size(); idx++) {
              auto old_child = pre_node_children.at(idx);
              auto new_child = createTree(dim_prop_sub_tree);
              pre_node_children.at(idx) = new_child;
              if (can_copy) {
                if (func_base_type != BaseFuncType::Map) {
                  copySubTree(new_child->getChild(), old_child->getChild());
                } else {
                  auto& childrenof_new_child = new_child->getChildren();
                  for (int idx2 = 0; idx2 < childrenof_new_child.size(); idx2++)
                    copySubTree(childrenof_new_child.at(idx2),
                                old_child->getChild());
                }
              }

              deleteSubTree(old_child);  // delete  old_child
            }
          } else {
            auto old_child = ff->getChild();
            auto new_child = createTree(dim_prop_sub_tree);
            ff->insertFunc(new_child);
            if (can_copy) {
              if (func_base_type != BaseFuncType::Map) {
                copySubTree(new_child->getChild(), old_child->getChild());
              } else {
                auto& childrenof_newchild = new_child->getChildren();
                for (int idx = 0; idx < childrenof_newchild.size(); idx++)
                  copySubTree(childrenof_newchild.at(idx),
                              old_child->getChild());
              }
            }

            deleteSubTree(old_child);  // delete  curr child
          }
        }
      }
    }  // else need not rebuild
  }

  /**
   * @brief Method to (re)build the whole fold space with each folddim of type
   * constant
   *
   * @param std::deque<const FoldDimProp*> props
   */
  void buildAllConstantFoldSpace(const std::deque<const FoldDimProp*>& props) {
    this->clear();
    fm_dim_prop dim_prop_sub_tree;

    for (int idx = 0; idx < props.size(); idx++) {
      dim_prop_.push_back({props.at(idx), BaseFuncType::Constant});
      dim_prop_sub_tree.push_back({props.at(idx), BaseFuncType::Constant});
    }

    this->parent_func_ = createTree(dim_prop_sub_tree);
  }

  /**
   * @brief Method to (re)build the whole fold space with each folddim of type
   * Map
   *
   * @param props
   */
  void buildAllMapFoldSpace(const std::deque<const FoldDimProp*>& props) {
    this->clear();

    fm_dim_prop dim_prop_sub_tree;
    for (int idx = 0; idx < props.size(); idx++) {
      dim_prop_.push_back({props.at(idx), BaseFuncType::Map});
      dim_prop_sub_tree.push_back({props.at(idx), BaseFuncType::Map});
    }
    this->parent_func_ = createTree(dim_prop_sub_tree);
  }

  /**
   * @brief Method to (re)build the whole fold space with each folddim of type
   * specified by func_base_types
   *
   * @param props
   * @param func_base_types
   */
  void buildFoldSpace(const std::deque<const FoldDimProp*>& props,
                      const std::deque<BaseFuncType>& func_base_types) {
    if (props.size() != func_base_types.size())
      DT_ERROR("Size of props and func_base_types should be same");

    fm_dim_prop dim_prop_sub_tree;
    for (int idx = 0; idx < props.size(); idx++) {
      dim_prop_sub_tree.emplace_back(props.at(idx), func_base_types.at(idx));
    }
    buildFoldSpace(dim_prop_sub_tree);
  }

  /**
   * @brief Method to (re)build the whole fold space with each folddim of type
   * specified by func_base_types
   *
   * @param props
   */
  void buildFoldSpace(const fm_dim_prop& dim_prop_sub_tree) {
    this->clear();
    dim_prop_.assign(dim_prop_sub_tree.begin(), dim_prop_sub_tree.end());
    this->parent_func_ = createTree(dim_prop_sub_tree);
  }

  // legality check
  template <typename... Args, class Enable = std::enable_if_t<(
                                  ... && std::is_convertible_v<Args, int64_t>)>>
  bool isLegal(const Args&... list) const {
    std::deque<int64_t> dim_indices{list...};
    return isLegal(dim_indices);
  }
  bool isLegal(const std::deque<int64_t>& fold_dim_indices) const {
    if (hasZeroFoldDim()) return true;  // always legal
    if (fold_dim_indices.size() != dim_prop_.size())
      DT_ERROR("number of dimensions in query and fold space are different\n");

    if (fold_dim_indices.size() == 0)
      if (parent_func_->Type() != FoldFunction<Dtype>::FuncType::Constant_leaf)
        DT_ERROR(
            "Fold space with zero (no) dimension should of type Constant\n");

    for (int idx = 0; idx < fold_dim_indices.size(); idx++) {
      if (dim_prop_.at(idx).first->getSize() <= fold_dim_indices.at(idx))
        DT_ERROR("query fold dimension with higher fold factor\n");
    }
    return true;
  }

  // get methods..
  template <typename... Args>
  Dtype getData(const Args&... list) const {
    // check access legality
    isLegal(list...);
    return parent_func_->getData(list...);
  }

  Dtype getData(const std::deque<int64_t>& fold_dim_indices) const {
    isLegal(fold_dim_indices);
    if (auto* affine_non_leaf_ptr =
            dynamic_cast<AffineFoldFunction_NonLeaf<Dtype>*>(parent_func_)) {
      return affine_non_leaf_ptr->getData(fold_dim_indices, 0);
    } else if (auto* affine_leaf_ptr =
                   dynamic_cast<AffineFoldFunction_Leaf<Dtype>*>(
                       parent_func_)) {
      return affine_leaf_ptr->getData(fold_dim_indices, 0);
    }
    DT_CHECK(parent_func_ != nullptr);
    return parent_func_->getData(fold_dim_indices, 0);
  }

  // access function, takes N dimensional folding indices as variadic variable
  template <typename... Args>
  void insertData(const Dtype& new_data, const Args&... list) {
    isLegal(list...);
    parent_func_->insertData(new_data, list...);
  }

  void insertData(const Dtype& new_data,
                  const std::deque<int64_t>& fold_dim_indices) {
    isLegal(fold_dim_indices);
    parent_func_->insertData(new_data, fold_dim_indices, 0);
  }

  // helper methods..
  /**
   * @brief the method constructs a horizontal group of FoldFunctions at
   * level=pos.
   *
   * @param pos
   * @param ffs_at_pos
   */
  void collectFoldFunctionAtLevel(
      int pos, std::vector<const FoldFunction<Dtype>*>& ffs_at_pos) const {
    DT_CHECK(pos < dim_prop_.size());  // leafs FFs do not store FF pointers
    std::deque<std::pair<FoldFunction<Dtype>*, uint32_t>> fifo;

    if (pos == 0) {
      ffs_at_pos.push_back(parent_func_);
    } else {
      fifo.push_back({parent_func_, 0});  // inital push
      int time = 0;
      while (fifo.size()) {
        auto next_entry = fifo.front();
        fifo.pop_front();
        if (next_entry.second == pos - 1) {
          if (next_entry.first->Type() !=
              FoldFunction<Dtype>::FuncType::Map_nonleaf)
            ffs_at_pos.push_back(next_entry.first->getChild());
          else {
            MapFoldFunction_NonLeaf<Dtype>* map_func =
                static_cast<MapFoldFunction_NonLeaf<Dtype>*>(next_entry.first);
            for (auto& ff : map_func->getChildren()) ffs_at_pos.push_back(ff);
          }
        } else if (next_entry.second < pos - 1) {
          if (next_entry.first->Type() !=
              FoldFunction<Dtype>::FuncType::Map_nonleaf)
            fifo.push_back(
                {next_entry.first->getChild(), next_entry.second + 1});
          else {
            MapFoldFunction_NonLeaf<Dtype>* map_func =
                static_cast<MapFoldFunction_NonLeaf<Dtype>*>(next_entry.first);
            for (auto& ff : map_func->getChildren())
              fifo.push_back({ff, next_entry.second + 1});
          }
        } else {
          DT_CHECK(0);  // cannot happen
        }
        time++;
        if (time >= std::pow(2, 20)) DT_ERROR("timeout\n");
      }
    }
  }

  /**
   * @brief the method constructs a horizontal group of FoldFunctions at
   * level=pos.
   *
   * @param pos
   * @param ffs_at_pos
   */
  void collectFoldFunctionAtLevel(
      int pos, std::vector<FoldFunction<Dtype>*>& ffs_at_pos) const {
    std::vector<const FoldFunction<Dtype>*> const_ffs_at_pos;
    collectFoldFunctionAtLevel(pos, const_ffs_at_pos);
    for (auto* ff : const_ffs_at_pos) {
      ffs_at_pos.push_back(const_cast<FoldFunction<Dtype>*>(ff));
    }
  }

  /**
   * @brief Get the Linear Func List object from FoldFunc Tree. Dimensions at
   * index specified by key of ignore_dim_idx_and_dim_value is ignored and it's
   * coordinate is fixed to value specified by ignore_dim_idx_and_dim_val
   * @brief Get the Linear Func List object
   *
   * @param linear_ff_list
   * @param ignore_dim_idx_and_dim_val
   */
  void getLinearFuncList(
      std::vector<FoldFunction<Dtype>*>& linear_ff_list,
      const std::map<int, int>& ignore_dim_idx_and_dim_val = {}) const {
    std::deque<std::pair<FoldFunction<Dtype>*, uint32_t>> fifo;
    if (dim_prop_.size()) {
      fifo.push_back({parent_func_, 0});  // inital push
    }

    int time = 0;
    while (fifo.size()) {
      auto next_entry = fifo.front();
      fifo.pop_front();
      int curr_pos = next_entry.second;
      int next_pos = curr_pos + 1;
      bool is_non_leaf = next_entry.first->isNonLeaf();

      if (ignore_dim_idx_and_dim_val.count(curr_pos) == 0)
        linear_ff_list.push_back(next_entry.first);

      if (is_non_leaf) {
        if (next_entry.first->Type() !=
            FoldFunction<Dtype>::FuncType::Map_nonleaf) {
          fifo.push_back({next_entry.first->getChild(), next_pos});
        } else {
          MapFoldFunction_NonLeaf<Dtype>* map_func =
              static_cast<MapFoldFunction_NonLeaf<Dtype>*>(next_entry.first);

          if (ignore_dim_idx_and_dim_val.count(curr_pos) == 0) {
            for (auto& ff : map_func->getChildren())
              fifo.push_back({ff, next_pos});
          } else {
            // just insert one function..
            fifo.push_back({map_func->getChildren().at(
                                ignore_dim_idx_and_dim_val.at(curr_pos)),
                            next_pos});
          }
        }
      }
      time++;
      if (time >= std::pow(2, 20)) DT_ERROR("timeout\n");
    }
  }

  /**
   * @brief Get the Linear Func List object from FoldFunc Tree starting at
   * ref_func
   *
   * @param linear_ff_list
   */
  void getLinearFuncList(FoldFunction<Dtype>* ref_func,
                         std::vector<FoldFunction<Dtype>*>& linear_ff_list) {
    std::deque<FoldFunction<Dtype>*> fifo;
    if (dim_prop_.size()) {
      fifo.push_back(ref_func);  // inital push
    }

    int time = 0;
    while (fifo.size()) {
      auto next_entry = fifo.front();
      fifo.pop_front();
      linear_ff_list.push_back(next_entry);
      bool is_non_leaf = next_entry->isNonLeaf();

      if (is_non_leaf) {
        if (next_entry->Type() != FoldFunction<Dtype>::FuncType::Map_nonleaf) {
          fifo.push_back(next_entry->getChild());
        } else {
          MapFoldFunction_NonLeaf<Dtype>* map_func =
              static_cast<MapFoldFunction_NonLeaf<Dtype>*>(next_entry);
          for (auto& ff : map_func->getChildren()) fifo.push_back(ff);
        }
      }
      time++;
      if (time >= std::pow(2, 20)) DT_ERROR("timeout\n");
    }
  }

  inline FoldFunction<Dtype>* createLeafFunc(const FoldDimProp* prop,
                                             BaseFuncType func_base_type) {
    if (func_base_type == BaseFuncType::Constant)
      return static_cast<FoldFunction<Dtype>*>(
          new ConstFoldFunction_Leaf<Dtype>());
    else if (func_base_type == BaseFuncType::Map)
      return static_cast<FoldFunction<Dtype>*>(
          new MapFoldFunction_Leaf<Dtype>(prop->getSize()));
    else if (func_base_type == BaseFuncType::Affine)
      return static_cast<FoldFunction<Dtype>*>(
          new AffineFoldFunction_Leaf<Dtype>());
    else if (func_base_type == BaseFuncType::WkSplit)
      return static_cast<FoldFunction<Dtype>*>(
          new WkSplitFoldFunction_Leaf<Dtype>());
    else
      DT_ERROR("Unknown base fold function\n");
    return static_cast<FoldFunction<Dtype>*>(
        new ConstFoldFunction_Leaf<Dtype>());
  }

  inline FoldFunction<Dtype>* createLeafFunc(const FoldDimProp* prop,
                                             BaseFuncType func_base_type,
                                             const Dtype& new_data) {
    if (func_base_type == BaseFuncType::Constant)
      return static_cast<FoldFunction<Dtype>*>(
          new ConstFoldFunction_Leaf<Dtype>(new_data));
    else if (func_base_type == BaseFuncType::Map)
      return static_cast<FoldFunction<Dtype>*>(
          new MapFoldFunction_Leaf<Dtype>(prop->getSize(), new_data));
    else if (func_base_type == BaseFuncType::Affine)
      return static_cast<FoldFunction<Dtype>*>(
          new AffineFoldFunction_Leaf<Dtype>());
    else if (func_base_type == BaseFuncType::WkSplit)
      return static_cast<FoldFunction<Dtype>*>(
          new WkSplitFoldFunction_Leaf<Dtype>());
    else
      DT_ERROR("Unknown base fold function\n");
    return static_cast<FoldFunction<Dtype>*>(
        new ConstFoldFunction_Leaf<Dtype>());
  }

  inline FoldFunction<Dtype>* createNonLeafFunc(
      const FoldDimProp* prop, BaseFuncType func_base_type,
      FoldFunction<Dtype>* next_child = nullptr) {
    if (func_base_type == BaseFuncType::Constant)
      return static_cast<FoldFunction<Dtype>*>(
          new ConstFoldFunction_NonLeaf<Dtype>(next_child));
    else if (func_base_type == BaseFuncType::Map) {
      return static_cast<FoldFunction<Dtype>*>(
          new MapFoldFunction_NonLeaf<Dtype>(prop->getSize()));
    } else if (func_base_type == BaseFuncType::Affine)
      return static_cast<FoldFunction<Dtype>*>(
          new AffineFoldFunction_NonLeaf<Dtype>(next_child));
    else
      DT_ERROR("Unknown base fold function\n");
    return static_cast<FoldFunction<Dtype>*>(
        new ConstFoldFunction_NonLeaf<Dtype>(next_child));
  }

  /**
   * @brief Get the Any One Data object
   *
   * @return Dtype
   */
  Dtype getSingleData(
      const std::map<int64_t, int64_t>& pos_to_fixCoord = {}) const {
    std::deque<int64_t> fold_dim_indices(dim_prop_.size(), 0);
    for (auto& [dim, coord] : pos_to_fixCoord) fold_dim_indices.at(dim) = coord;
    return getData(fold_dim_indices);
  }

  /**
   * @brief Get the Number Of Unique Data Coords object
   *
   * @param unique_data_coords
   * @return int
   */
  int64_t getNumUniqueCoordsInEachFold(
      std::vector<int64_t>& unique_data_coords) const {
    if (dim_prop_.size() == 0) return 0;

    int total_count = 1;
    for (auto& dp : dim_prop_) {
      if (dp.second == BaseFuncType::Constant) {
        unique_data_coords.push_back(1);
      } else {
        unique_data_coords.push_back(dp.first->getSize());
        total_count *= dp.first->getSize();
      }
    }
    return total_count;
  }

  int64_t getAllCoordsInEachFold(std::vector<int64_t>& all_fold_coords) const {
    if (dim_prop_.size() == 0) return 0;
    all_fold_coords.clear();
    int total_count = 1;
    for (auto& dp : dim_prop_) {
      all_fold_coords.push_back(dp.first->getSize());
      total_count *= dp.first->getSize();
    }
    return total_count;
  }

  /**
   * @brief Get the Flattened Coordinates object. The pos present in
   * pos_to_fixCoord are not flattened, i.e., their coordinates do not vary and
   * it is fix to the value specified in pos_to_fixCoord
   *
   * @param coordinates
   * @param pos_to_fixCoord
   * @param scan_inner_outer
   */
  void getFlattenedCoordinates(
      std::vector<std::deque<int64_t>>& coordinates,
      const std::map<int64_t, int64_t>& pos_to_fixCoord = {},
      const bool scan_inner_outer = false,
      bool expand_if_any_map = false) const {
    if (dim_prop_.size() > 0) {
      bool do_expand = false;
      if (expand_if_any_map) {
        for (int i = 0; i < dim_prop_.size(); i++) {
          const auto& [fp_dim, func] = dim_prop_.at(i);
          if (!pos_to_fixCoord.count(i) && func == BaseFuncType::Map &&
              fp_dim->getSize() > 1) {
            do_expand = true;
            break;
          }
        }
      }
      std::vector<int64_t> unique_data_coords;
      auto total_count = do_expand
                             ? getAllCoordsInEachFold(unique_data_coords)
                             : getNumUniqueCoordsInEachFold(unique_data_coords);

      // make adjustments for pos_to_fixCoord
      if (!pos_to_fixCoord.empty()) {
        for (auto& kv : pos_to_fixCoord) {
          DT_CHECK(total_count % unique_data_coords.at(kv.first) == 0);
          total_count /= unique_data_coords.at(kv.first);
          unique_data_coords.at(kv.first) = 1;
        }
      }

      coordinates.resize(total_count);
      for (int idx = 0; idx < total_count; idx++) {
        coordinates.at(idx).resize(unique_data_coords.size());
      }

      int repeat_factor = 1;
      for (int i = 0; i < unique_data_coords.size(); i++) {
        const int dim_idx =
            scan_inner_outer ? unique_data_coords.size() - 1 - i : i;
        for (int coord_idx = 0; coord_idx < total_count; coord_idx++) {
          coordinates.at(coord_idx).at(dim_idx) =
              (coord_idx / repeat_factor) % unique_data_coords.at(dim_idx);
        }
        repeat_factor *= unique_data_coords.at(dim_idx);
      }

      FoldInfraUtils::fixCoordinatesAtPos(coordinates, pos_to_fixCoord);
    }
  }

  /**
   * @brief Get the Flattened Coordinates object. The pos present in
   * pos_to_fixCoord are not flattened, i.e., their coordinates do not vary and
   * it is fix to the value specified in pos_to_fixCoord
   *
   * @param pos_to_fixCoord
   * @param scan_inner_outer
   * @return std::vector<std::deque<int64_t>>
   */
  std::vector<std::deque<int64_t>> getFlattenedCoordinates(
      const std::map<int64_t, int64_t>& pos_to_fixCoord = {},
      const bool scan_inner_outer = false,
      bool expand_if_any_map = false) const {
    std::vector<std::deque<int64_t>> coordinates;
    getFlattenedCoordinates(coordinates, pos_to_fixCoord, scan_inner_outer,
                            expand_if_any_map);
    return coordinates;
  }

  /**
   * @brief Get the Data And Fold Coordinates object. The pos present in
   * pos_to_fixCoord are not flattened, i.e., their coordinates do not vary and
   * it is fix to the value specified in pos_to_fixCoord
   *
   * @param data_and_coord
   * @param pos_to_fixCoord
   * @param scan_inner_outer
   */
  void getDataAndFoldCoordinates(
      std::vector<std::pair<std::deque<int64_t>, Dtype>>& data_and_coord,
      const std::map<int64_t, int64_t>& pos_to_fixCoord = {},
      const bool scan_inner_outer = false) const {
    if (dim_prop_.size() == 0) {
      std::deque<int64_t> zero_fold_coord;
      zero_fold_coord.push_back(-1);
      data_and_coord.push_back(std::make_pair(zero_fold_coord, getData()));
    } else {
      std::vector<std::deque<int64_t>> coordinates;
      getFlattenedCoordinates(coordinates, pos_to_fixCoord, scan_inner_outer);
      data_and_coord.resize(coordinates.size());
      for (int idx = 0; idx < coordinates.size(); idx++) {
        data_and_coord.at(idx) =
            std::make_pair(coordinates.at(idx), getData(coordinates.at(idx)));
      }
    }
  }

  /**
   * @brief Get the Data And Fold Coordinates object. The pos present in
   * pos_to_fixCoord are not flattened, i.e., their coordinates do not vary and
   * it is fix to the value specified in pos_to_fixCoord
   *
   * @param pos_to_fixCoord
   * @param scan_inner_outer
   * @return std::vector<std::pair<std::deque<int64_t>, Dtype>>
   */
  std::vector<std::pair<std::deque<int64_t>, Dtype>> getDataAndFoldCoordinates(
      const std::map<int64_t, int64_t>& pos_to_fixCoord = {},
      const bool scan_inner_outer = false) const {
    std::vector<std::pair<std::deque<int64_t>, Dtype>> data_and_coord;
    getDataAndFoldCoordinates(data_and_coord, pos_to_fixCoord,
                              scan_inner_outer);
    return data_and_coord;
  }

  /**
   * @brief Get the Ordered Data For object. The pos present in
   * pos_to_fixCoord are not flattened, i.e., their coordinates do not vary and
   * it is fix to the value specified in pos_to_fixCoord
   *
   * @param pos_to_fixCoord
   * @param scan_inner_outer
   * @return std::vector<Dtype>
   */
  std::vector<Dtype> getAllData(
      const std::map<int64_t, int64_t>& pos_to_fixCoord = {},
      const bool scan_inner_outer = false) const {
    std::vector<Dtype> all_data;
    if (dim_prop_.size() == 0) {
      std::deque<int64_t> zero_fold_coord;
      zero_fold_coord.push_back(-1);
      all_data.push_back(getData());
    } else {
      std::vector<std::deque<int64_t>> coordinates;
      getFlattenedCoordinates(coordinates, pos_to_fixCoord, scan_inner_outer);
      all_data.resize(coordinates.size());
      for (int idx = 0; idx < coordinates.size(); idx++) {
        all_data.at(idx) = getData(coordinates.at(idx));
      }
    }
    return all_data;
  }

  std::vector<Dtype> getAllDataWithMapUnrolled(
      const std::map<int64_t, int64_t>& pos_to_fixCoord = {},
      const bool scan_inner_outer = false) const {
    std::vector<Dtype> all_data;
    if (dim_prop_.size() == 0) {
      std::deque<int64_t> zero_fold_coord;
      zero_fold_coord.push_back(-1);
      all_data.push_back(getData());
    } else {
      std::vector<std::deque<int64_t>> coordinates;
      getFlattenedCoordinates(coordinates, pos_to_fixCoord, scan_inner_outer,
                              true);
      all_data.resize(coordinates.size());
      for (int idx = 0; idx < coordinates.size(); idx++) {
        all_data.at(idx) = getData(coordinates.at(idx));
      }
    }
    return all_data;
  }

  /**
   * @brief Get the Fold Coordinates to Data Map object. The pos present in
   * pos_to_fixCoord are not flattened, i.e., their coordinates do not vary and
   * it is fix to the value specified in pos_to_fixCoord
   *
   * @param pos_to_fixCoord
   * @param data_and_coord
   */
  void getFoldCoordinatesAndDataMap(
      std::map<std::deque<int64_t>, Dtype>& data_and_coord,
      const std::map<int64_t, int64_t>& pos_to_fixCoord = {}) const {
    if (dim_prop_.size() == 0) {
      std::deque<int64_t> zero_fold_coord;
      zero_fold_coord.push_back(-1);
      data_and_coord.emplace(zero_fold_coord, getData());
    } else {
      std::vector<std::deque<int64_t>> coordinates;
      getFlattenedCoordinates(coordinates, pos_to_fixCoord);
      for (int idx = 0; idx < coordinates.size(); idx++) {
        data_and_coord.emplace(coordinates.at(idx),
                               getData(coordinates.at(idx)));
      }
    }
  }

  /**
   * @brief Get the Fold Coordinates to Data Map object. The pos present in
   * pos_to_fixCoord are not flattened, i.e., their coordinates do not vary and
   * it is fix to the value specified in pos_to_fixCoord
   *
   * @return std::map<std::deque<int64_t>, Dtype>
   */
  std::map<std::deque<int64_t>, Dtype> getFoldCoordinatesAndDataMap(
      const std::map<int64_t, int64_t>& pos_to_fixCoord = {}) const {
    std::map<std::deque<int64_t>, Dtype> data_and_coord;
    getFoldCoordinatesAndDataMap(data_and_coord, pos_to_fixCoord);
    return data_and_coord;
  }

  void printMetaData(std::ostream& out, std::string ps = "",
                     bool add_comma = true, bool compressed = false) const {
    auto QUOTE = [](std::string str) { return "\"" + str + "\""; };
    out << ps << QUOTE("dim_prop_func") << " : [\n";
    auto ps2 = ps + "  ";
    for (int idx = 0; idx < dim_prop_.size(); idx++) {
      out << ps2 << "{ "
          << QUOTE(FoldInfraUtils::baseFuncTypeToString.at(
                 dim_prop_.at(idx).second))
          << " : {";
      if ((dim_prop_.at(idx).second == BaseFuncType::Map ||
           dim_prop_.at(idx).second == BaseFuncType::Constant))  // no meta data
        out << "} }";
      else {
        // print meta-data
        // get all children at pos
        std::vector<const FoldFunction<Dtype>*> ffs_at_pos;
        collectFoldFunctionAtLevel(idx /*pos*/, ffs_at_pos);
        DT_CHECK(ffs_at_pos.size());
        ffs_at_pos.front()->printMetaData(out, "");
        out << "} }";
      }
      if (idx != dim_prop_.size() - 1) out << ",";
      out << "\n";
    }
    out << ps << "]";
    if (!compressed) {
      out << "," << std::endl;
      out << ps << QUOTE("dim_prop_attr") << " : [";
      for (int idx = 0; idx < dim_prop_.size(); idx++) {
        out << "\n" << ps2 << "{ ";
        dim_prop_.at(idx).first->print(out);
        out << " }";
        if (idx != dim_prop_.size() - 1) out << ",";
      }
      out << "\n" << ps << "]";
    }
    if (add_comma) out << ",";
    out << std::endl;
  }

  FoldFunction<Dtype>* createTree(fm_dim_prop dim_prop_sub_tree) {
    if (dim_prop_sub_tree.empty())
      DT_ERROR("Need at least one fold dim to build a tree");

    if (dim_prop_sub_tree.size() == 1) {
      auto& curr_dim_prop = dim_prop_sub_tree.front();
      return createLeafFunc(curr_dim_prop.first, curr_dim_prop.second);
    } else {
      auto& curr_dim_prop = dim_prop_sub_tree.front();
      FoldFunction<Dtype>* tree_top =
          createNonLeafFunc(curr_dim_prop.first, curr_dim_prop.second);
      bool is_map = curr_dim_prop.second == BaseFuncType::Map;
      // pop last entry
      dim_prop_sub_tree.erase(dim_prop_sub_tree.begin());
      if (is_map) {
        createSubTreeForEachMapChild(tree_top, dim_prop_sub_tree);
      } else {
        tree_top->insertFunc(createTree(dim_prop_sub_tree));
      }
      return tree_top;
    }
  }

  void createSubTreeForEachMapChild(FoldFunction<Dtype>* map_non_leaf,
                                    const fm_dim_prop& dim_prop_sub_tree) {
    DT_CHECK(map_non_leaf->Type() ==
             FoldFunction<Dtype>::FuncType::Map_nonleaf);
    auto& children = map_non_leaf->getChildren();
    for (int idx = 0; idx < children.size(); idx++)
      children.at(idx) = createTree(dim_prop_sub_tree);
  }

  void getAllDimProFromPos(int pos, fm_dim_prop& dim_prop_sub_tree) {
    DT_CHECK(pos >= 0);
    for (int idx = pos; idx < dim_prop_.size(); idx++)
      dim_prop_sub_tree.push_back(dim_prop_.at(idx));
  }

  /**
   * @brief Get the Num Dims object
   *
   * @return int
   */
  int getNumDims() const { return dim_prop_.size(); }

  /**
   * @brief method insert alpha in the func tree
   *
   * @param new_alpha
   * @param pos
   */
  void insertAlpha(const Dtype& new_alpha, int pos) {
    if (pos < 0) {
      pos = dim_prop_.size() + pos;
    }
    DT_CHECK(pos >= 0 && pos <= dim_prop_.size() - 1);
    DT_CHECK(dim_prop_.size() > 0);

    // update
    if (dim_prop_.at(pos).second != BaseFuncType::Affine)
      DT_ERROR(" Cannot insert alpha in non affine fold func\n");

    // get all children at pos
    std::vector<FoldFunction<Dtype>*> ffs_at_pos;
    collectFoldFunctionAtLevel(pos, ffs_at_pos);

    for (auto& ff : ffs_at_pos) ff->insertAlpha(new_alpha);
  }

  /**
   * @brief method insert beta in the func tree
   *
   * @param new_beta
   * @param pos
   */
  void insertBeta(const Dtype& new_beta, int pos) {
    if (pos < 0) {
      pos = dim_prop_.size() + pos;
    }
    DT_CHECK(pos >= 0 && pos <= dim_prop_.size() - 1);
    DT_CHECK(dim_prop_.size() > 0);

    // update
    if (dim_prop_.at(pos).second != BaseFuncType::Affine)
      DT_ERROR(" Cannot insert beta in non affine fold func\n");

    // get all children at pos
    std::vector<FoldFunction<Dtype>*> ffs_at_pos;
    collectFoldFunctionAtLevel(pos, ffs_at_pos);

    for (auto& ff : ffs_at_pos) ff->insertBeta(new_beta);
  }

  /**
   * @brief method get alpha in the func tree
   *
   * @param pos
   */
  Dtype getAlpha(int pos) const {
    if (pos < 0) {
      pos = dim_prop_.size() + pos;
    }
    DT_CHECK(pos >= 0 && pos <= dim_prop_.size() - 1);
    DT_CHECK(dim_prop_.size() > 0);

    // update
    if (dim_prop_.at(pos).second != BaseFuncType::Affine)
      DT_ERROR(" Cannot get alpha in non affine fold func\n");

    // get all children at pos
    std::vector<FoldFunction<Dtype>*> ffs_at_pos;
    collectFoldFunctionAtLevel(pos, ffs_at_pos);
    DT_CHECK_MSG(ffs_at_pos.size() == 1, "Expect only one fold function.");

    if (dynamic_cast<AffineFoldFunction_NonLeaf<Dtype>*>(ffs_at_pos.front())) {
      auto ff =
          dynamic_cast<AffineFoldFunction_NonLeaf<Dtype>*>(ffs_at_pos.front());
      return ff->getAlpha();
    } else {
      auto ff =
          dynamic_cast<AffineFoldFunction_Leaf<Dtype>*>(ffs_at_pos.front());
      if (ff == nullptr)
        DT_ERROR(" Cannot get alpha in non affine fold func\n");
      return ff->getAlpha();
    }
  }

  /**
   * @brief method get beta in the func tree
   *
   * @param pos
   */
  Dtype getBeta(int pos) const {
    if (pos < 0) {
      pos = dim_prop_.size() + pos;
    }
    DT_CHECK(pos >= 0 && pos <= dim_prop_.size() - 1);
    DT_CHECK(dim_prop_.size() > 0);

    // update
    if (dim_prop_.at(pos).second != BaseFuncType::Affine)
      DT_ERROR(" Cannot get beta in non affine fold func\n");

    // get all children at pos
    std::vector<FoldFunction<Dtype>*> ffs_at_pos;
    collectFoldFunctionAtLevel(pos, ffs_at_pos);
    DT_CHECK_MSG(ffs_at_pos.size() == 1, "Expect only one fold function.");

    if (dynamic_cast<AffineFoldFunction_NonLeaf<Dtype>*>(ffs_at_pos.front())) {
      auto ff =
          dynamic_cast<AffineFoldFunction_NonLeaf<Dtype>*>(ffs_at_pos.front());
      return ff->getBeta();
    } else {
      auto ff =
          dynamic_cast<AffineFoldFunction_Leaf<Dtype>*>(ffs_at_pos.front());
      if (ff == nullptr) DT_ERROR(" Cannot get beta in non affine fold func\n");
      return ff->getBeta();
    }
  }

  /**
   * @brief method insert alpha and beta in the func tree
   *
   * @param new_alpha
   * @param new_beta
   * @param pos
   */
  void insertAlphaBeta(const Dtype& new_alpha, const Dtype& new_beta, int pos) {
    if (pos < 0) {
      pos = dim_prop_.size() + pos;
    }
    DT_CHECK(pos >= 0 && pos <= dim_prop_.size() - 1);
    DT_CHECK(dim_prop_.size() > 0);

    // update
    if (dim_prop_.at(pos).second != BaseFuncType::Affine)
      DT_ERROR(" Cannot insert beta in non affine fold func\n");

    // get all children at pos
    std::vector<FoldFunction<Dtype>*> ffs_at_pos;
    collectFoldFunctionAtLevel(pos, ffs_at_pos);

    for (auto& ff : ffs_at_pos) {
      ff->insertAlpha(new_alpha);
      ff->insertBeta(new_beta);
    }
  }

  /**
   * @brief Method to get alpha and beta at pos
   *
   * @param alpha
   * @param beta
   * @param pos
   */
  void getAlphaBeta(Dtype& alpha, Dtype& beta, int pos) const {
    if (pos < 0) {
      pos = dim_prop_.size() + pos;
    }
    DT_CHECK(pos >= 0 && pos <= dim_prop_.size() - 1);
    DT_CHECK(dim_prop_.size() > 0);

    // update
    if (dim_prop_.at(pos).second != BaseFuncType::Affine)
      DT_ERROR(" Cannot query beta in non affine fold func\n");

    // get all children at pos
    std::vector<FoldFunction<Dtype>*> ffs_at_pos;
    collectFoldFunctionAtLevel(pos, ffs_at_pos);

    DT_CHECK_MSG(!ffs_at_pos.empty(), "should be non empty");
    auto& ff = *ffs_at_pos.begin();
    if (ff->Type() == FoldFunction<Dtype>::FuncType::Affine_leaf) {
      AffineFoldFunction_Leaf<Dtype>* aff =
          static_cast<AffineFoldFunction_Leaf<Dtype>*>(ff);
      alpha = aff->getAlpha();
      beta = aff->getBeta();
    } else if (ff->Type() == FoldFunction<Dtype>::FuncType::Affine_nonleaf) {
      AffineFoldFunction_NonLeaf<Dtype>* aff =
          static_cast<AffineFoldFunction_NonLeaf<Dtype>*>(ff);
      alpha = aff->getAlpha();
      beta = aff->getBeta();
    }
  }

  /**
   * @brief Get the Fold Space Size object
   *
   * @return std::vector<int>
   */
  std::vector<int64_t> getFoldSpaceSize() const {
    std::vector<int64_t> foldSpace;  // [0] is outer most dim
    for (int idx = 0; idx < dim_prop_.size(); idx++)
      foldSpace.push_back(dim_prop_.at(idx).first->getSize());
    return foldSpace;
  }

  /**
   * @brief method to copy data and meta-data of ref_sub_tree to sub_tree
   *
   * @param sub_tree
   * @param ref_sub_tree
   */
  void copySubTree(FoldFunction<Dtype>* sub_tree,
                   FoldFunction<Dtype>* ref_sub_tree) {
    // copy data
    // step 1 : get linear list for rhs and this
    std::vector<FoldFunction<Dtype>*> linear_ff_list_rhs;
    std::vector<FoldFunction<Dtype>*> linear_ff_list_this;

    getLinearFuncList(sub_tree, linear_ff_list_this);
    getLinearFuncList(ref_sub_tree, linear_ff_list_rhs);

    // step 2 go through the list
    if (linear_ff_list_rhs.size() != linear_ff_list_this.size())
      DT_ERROR("Unexpected");

    while (linear_ff_list_rhs.size()) {
      auto rhs_ff = linear_ff_list_rhs.back();
      auto this_ff = linear_ff_list_this.back();
      linear_ff_list_rhs.pop_back();
      linear_ff_list_this.pop_back();
      DT_CHECK(rhs_ff->Type() == this_ff->Type());

      if (rhs_ff->Type() == FoldFunction<Dtype>::FuncType::Constant_leaf) {
        this_ff->insertData(rhs_ff->getData());
      } else if (rhs_ff->Type() == FoldFunction<Dtype>::FuncType::Affine_leaf) {
        auto this_leaf = dynamic_cast<AffineFoldFunction_Leaf<Dtype>*>(this_ff);
        auto rhs_leaf = dynamic_cast<AffineFoldFunction_Leaf<Dtype>*>(rhs_ff);
        auto alpha = rhs_leaf->getAlpha();
        auto beta = rhs_leaf->getBeta();
        this_leaf->insertAlpha(alpha);
        this_leaf->insertBeta(beta);
      } else if (rhs_ff->Type() ==
                 FoldFunction<Dtype>::FuncType::Affine_nonleaf) {
        auto this_nonleaf =
            dynamic_cast<AffineFoldFunction_NonLeaf<Dtype>*>(this_ff);
        auto rhs_nonleaf =
            dynamic_cast<AffineFoldFunction_NonLeaf<Dtype>*>(rhs_ff);
        auto alpha = rhs_nonleaf->getAlpha();
        auto beta = rhs_nonleaf->getBeta();
        this_nonleaf->insertAlpha(alpha);
        this_nonleaf->insertBeta(beta);
      } else if (rhs_ff->Type() ==
                 FoldFunction<Dtype>::FuncType::WkSplit_leaf) {
        auto this_leaf =
            dynamic_cast<WkSplitFoldFunction_Leaf<Dtype>*>(this_ff);
        auto rhs_leaf = dynamic_cast<WkSplitFoldFunction_Leaf<Dtype>*>(rhs_ff);
        this_leaf->insertWkSplitParam(rhs_leaf->getWkSplitParamMutable());
      } else if (rhs_ff->Type() == FoldFunction<Dtype>::FuncType::Map_leaf) {
        auto this_leaf = dynamic_cast<MapFoldFunction_Leaf<Dtype>*>(this_ff);
        auto rhs_leaf = dynamic_cast<MapFoldFunction_Leaf<Dtype>*>(rhs_ff);
        this_leaf->getDataVec() = rhs_leaf->getDataVec();
      }  // else : do nothing
    }
  }

  /**
   * @brief method to delete a sub tree with sub_tree as source
   *
   * @param sub_tree
   */
  void deleteSubTree(FoldFunction<Dtype>* sub_tree) {
    // step 1 : get linear list
    std::vector<FoldFunction<Dtype>*> linear_ff_list;
    getLinearFuncList(sub_tree, linear_ff_list);

    while (linear_ff_list.size()) {
      auto ff = linear_ff_list.back();
      linear_ff_list.pop_back();
      delete ff;
    }
  }

  /**
   * @brief method insert alpha in the func tree
   *
   * @param new_alpha
   * @param pos
   */

  /**
   * @brief method to insert WkSplitParam in the func tree at pos
   *
   * @param wksplit_param
   * @param pos
   */
  void insertWkSplitParam(WkSplitParam& wksplit_param, int pos) {
    if (pos < 0) {
      pos = dim_prop_.size() + pos;
    }
    DT_CHECK(pos >= 0 && pos <= dim_prop_.size() - 1);
    DT_CHECK(dim_prop_.size() > 0);

    // update
    if (dim_prop_.at(pos).second != BaseFuncType::WkSplit)
      DT_ERROR(" Cannot insert wkSplitParam in non WkSplit fold func\n");

    // get all children at pos
    std::vector<FoldFunction<Dtype>*> ffs_at_pos;
    collectFoldFunctionAtLevel(pos, ffs_at_pos);

    for (auto& ff : ffs_at_pos) ff->insertWkSplitParam(wksplit_param);
  }

  /**
   * @brief Get the Func Type object
   *
   * @param pos
   * @return BaseFuncType
   */
  BaseFuncType getFuncType(int pos) const {
    if (pos >= dim_prop_.size()) DT_ERROR(" Illegal access\n");
    return dim_prop_.at(pos).second;
  }

  /**
   * @brief Get base function type for all fold levels
   *
   * @return std::deque<BaseFuncType>
   */
  std::deque<BaseFuncType> getFuncType() const {
    std::deque<BaseFuncType> base_func_types;
    for (int i = 0; i < dim_prop_.size(); i++) {
      base_func_types.push_back(dim_prop_.at(i).second);
    }
    return base_func_types;
  }

  /**
   * @brief Method checks if the managed object has no fold dimensions, meaning
   * the object is mannaged as a zeroth order constant fold
   *
   * @return true
   * @return false
   */
  bool hasZeroFoldDim() const { return (dim_prop_.size() == 0); }

  /**
   * @brief Get the Fold Dim Prop obj
   *
   * @return std::deque<const FoldDimProp*>
   */
  std::deque<const FoldDimProp*> getFoldDimProp() const {
    std::deque<const FoldDimProp*> fold_dim_props;
    for (int i = 0; i < dim_prop_.size(); i++) {
      fold_dim_props.push_back(dim_prop_.at(i).first);
    }
    return fold_dim_props;
  }

  /**
   * @brief Get the Fold Dim Prop object at particular pos
   *
   * @param pos
   * @return const FoldDimProp*
   */
  const FoldDimProp* getFoldDimProp(int pos) { return dim_prop_.at(pos).first; }

  /**
   * @brief Get the dimension size at pos
   *
   * @return int
   */
  int getFoldDimSize(int pos) const {
    return dim_prop_.at(pos).first->getSize();
  }

  /**
   * @brief Print dim_prop_ pointers
   *
   * @return int
   */
  void printDimPropPtr() const {
    int pos = 0;
    for (auto& pair : dim_prop_) {
      std::cout << "pos: " << pos << " -- ptr: " << pair.first << std::endl;
      pos++;
    }
  }

  /**
   * @brief Method inserts wksplit param in a particular wksplit function
   * specified by "fold_dim_indices"
   *
   * @param wksplit_param
   * @param fold_dim_indices
   */
  void insertWkSplitParam(WkSplitParam& wksplit_param,
                          const std::deque<int64_t>& fold_dim_indices) {
    auto fold_func = getFoldFunc(fold_dim_indices);
    fold_func->insertWkSplitParam(wksplit_param);
  }

  /**
   * @brief Get the const reference of Wk Split Param object
   *
   * @param fold_dim_indices
   * @return const WkSplitParam&
   */
  const WkSplitParam& getWkSplitParam(
      const std::deque<int64_t>& fold_dim_indices) const {
    auto fold_func = getFoldFunc(fold_dim_indices);
    return fold_func->getWkSplitParam();
  }

  /**
   * @brief Get the mutable reference of Wk Split Param object
   *
   * @param fold_dim_indices
   * @return WkSplitParam&
   */
  WkSplitParam& getWkSplitParam(const std::deque<int64_t>& fold_dim_indices) {
    auto fold_func = getFoldFunc(fold_dim_indices);
    return fold_func->getWkSplitParamMutable();
  }

  template <typename T = Dtype,
            std::enable_if_t<std::is_arithmetic<T>::value, bool> = true>
  void printData(std::ostream& out, const Dtype& data) const {
    auto QUOTE = [](std::string str) { return "\"" + str + "\""; };
    out << QUOTE(std::to_string(data));
  }

  template <
      typename T = Dtype,
      std::enable_if_t<std::is_same<T, std::set<typename T::value_type>>::value,
                       bool> = true>
  void printData(std::ostream& out, const Dtype& data) const {
    auto QUOTE = [](std::string str) { return "\"" + str + "\""; };
    out << "[" << PrintUtil::printSetwithQuotes(data) << "]";
  }

  template <typename T = Dtype,
            std::enable_if_t<
                std::is_same<T, std::vector<typename T::value_type>>::value,
                bool> = true>
  void printData(std::ostream& out, const Dtype& data) const {
    auto QUOTE = [](std::string str) { return "\"" + str + "\""; };
    out << "[" << PrintUtil::printVecwithQuotes(data) << "]";
  }

  template <typename T = Dtype,
            std::enable_if_t<
                !(std::is_arithmetic<T>::value ||
                  std::is_same<T, std::set<typename T::value_type>>::value ||
                  std::is_same<T, std::vector<typename T::value_type>>::value),
                bool> = true>
  void printData(std::ostream& out, const Dtype& data) const {
    auto QUOTE = [](std::string str) { return "\"" + str + "\""; };
    DT_ERROR("Print util is not available for this Dtype\n");
  }

  /**
   * @brief This method is a generic print for data managed by fold-manager
   *
   * @param out
   * @param ps
   */
  void print(std::ostream& out, std::string ps = "",
             bool printContent = true) const {
    auto QUOTE = [](std::string str) { return "\"" + str + "\""; };

    if (hasZeroFoldDim()) {
      printData(out, getData());
    } else {
      auto l_so2 = ps + "  ";
      out << " {\n";
      printMetaData(out, l_so2, printContent);
      if (printContent) {
        auto data_and_coord = getDataAndFoldCoordinates({}, true);
        out << l_so2 << QUOTE("data_") << " : {\n";
        int c1 = data_and_coord.size();
        for (auto& [coord, data] : data_and_coord) {
          out << l_so2 << "  " << "\"[" << PrintUtil::printVec(coord)
              << "]\" :";
          printData(out, data);
          out << (--c1 > 0 ? ",\n" : "\n");
        }
        out << l_so2 << "}\n";
      }
      out << ps << "}";
    }
  }

  /**
   * @brief This method is a generic import for fold-manager exported with above
   * print method. FoldDimProps must be imported before and be available for
   * construction of this FM. Their number should match what comes from the json
   *
   * @param json
   * @param props
   */
  void importFromJson(const json11::Json& json,
              const std::deque<const FoldDimProp*>& props) {
    DT_CHECK_MSG(hasZeroFoldDim(),
                 "Expected empty FoldManager when importing from json");
    if (!json.is_object() || props.empty()) {  // zero fold
      insertData(ImportUtil::importData<Dtype>(json));
      return;
    }
    auto& jsonMap = json.object_items();
    // DT_CHECK_MSG(jsonMap.size() == 3, "Unexpected number of entries in
    // json");
    auto& dimSizes = jsonMap.at("dim_prop_attr").array_items();
    DT_CHECK_MSG(props.size() == dimSizes.size(),
                 "Different number of dims between json and caller");
    for (int i = 0; i < props.size(); i++) {
      DT_CHECK_MSG(props[i]->getSize() ==
                       dimSizes[i].object_items().at("factor_").int_value(),
                   "Different cardinality between json and caller");
    }
    auto& dimProps = jsonMap.at("dim_prop_func").array_items();
    std::vector<int> posOfAffineFolds;
    for (int i = dimProps.size() - 1; i >= 0; i--) {
      auto& funcTypeStruct = dimProps[i].object_items();
      DT_CHECK(funcTypeStruct.size() == 1);
      auto& [funcTypeStr, funcProps] = *funcTypeStruct.begin();
      auto funcType = FoldInfraUtils::stringToBaseFuncType.at(funcTypeStr);
      DT_CHECK_MSG(is_any_of(funcType, BaseFuncType::Constant,
                             BaseFuncType::Map, BaseFuncType::Affine),
                   "Func type not yet supported in import function");
      buildDim(props[i], funcType, 0);
      auto& metadataJson = funcProps.object_items();
      if (funcType == BaseFuncType::Affine) {
        posOfAffineFolds.push_back(i);
        auto alpha = ImportUtil::importData<Dtype>(metadataJson.at("alpha_")),
             beta = ImportUtil::importData<Dtype>(metadataJson.at("beta_"));
        insertAlphaBeta(alpha, beta, 0);
      }
      // TODO:
      // For affine, if we have affine mixed with other types (map or const),
      // how do we adjust data values to subtract beta during import?
      // For WkSplit, import metadata
    }
    DT_CHECK_MSG(
        posOfAffineFolds.empty() || posOfAffineFolds.size() == props.size(),
        "Affine folds together with other types of folds require adjustment of "
        "the imported data to compensate for beta. Not currently handled");

    if (jsonMap.count("data_")) {
      for (auto& [coordStr, value] : jsonMap.at("data_").object_items()) {
        auto coord = ImportUtil::importData<std::deque<int64_t>>(coordStr);
        DT_CHECK_MSG(coord.size() == props.size(),
                     "Num of dimensions in coordinate not matching num folds");
        for (const auto& pos : posOfAffineFolds) {
          if (coord[pos] != 0) continue;
        }
        insertData(ImportUtil::importData<Dtype>(value), coord);
      }
    }
  }

  const fm_dim_prop& getDimProp() const { return dim_prop_; }

  /**
   * @brief Method compresses managed variable to use const fold function from
   * pos to the lowest dim if possible. If compression was performed it returns
   * true. Compresses Maps to Const
   *
   * @param pos
   * @return true
   * @return false
   */
  bool compressMapToConst(int pos) {
    DT_CHECK(pos < dim_prop_.size());
    if (dim_prop_.at(pos).second != BaseFuncType::Map) {
      return false;
    }

    bool areAllConstant = true;
    std::vector<std::pair<std::deque<int64_t>, Dtype>> coord_to_data;
    getDataAndFoldCoordinates(coord_to_data);

    // To make a fold func at pos constant,
    // itself and all subtree data must be identical.
    std::map<std::deque<int64_t>, Dtype> reduced_coord_to_data;
    for (const auto& [coord, data] : coord_to_data) {
      // Ignore itself and all subtree coords
      std::deque<int64_t> reduced_coord;
      if (pos == 0) {
        reduced_coord = {0};
      } else {
        for (int i = 0; i < coord.size(); i++) {
          if (i < pos) {
            reduced_coord.push_back(coord.at(i));
          }
        }
      }
      // The reduced coord must have only one data to be constant
      auto coord_found = reduced_coord_to_data.count(reduced_coord);
      if (coord_found == 0) {
        reduced_coord_to_data[reduced_coord] = data;
      } else {
        if (reduced_coord_to_data.at(reduced_coord) != data) {
          areAllConstant = false;
          break;
        }
      }
    }

    if (areAllConstant) {
      // Make fold func at pos and subtree constant
      for (int i = pos; i < dim_prop_.size(); i++) {
        if (dim_prop_.at(i).second != BaseFuncType::Constant) {
          rebuildDim(pos, BaseFuncType::Constant);
          for (auto& [coord, data] : coord_to_data) {
            insertData(data, coord);
          }
        }
      }
      return true;
    }
    return false;
  }

  /**
   * @brief Method compresses managed variable to use const fold function for
   * all dims (starting from the lowest dim) if possible. If at least one
   * compression was performed it returns true. Compresses Maps to Const
   *
   * @return true
   * @return false
   */
  bool compressMapToConstForAllDims() {
    bool wasCompressed = false;
    auto num_fold_dims = getNumDims();
    for (int pos = num_fold_dims - 1; pos >= 0; --pos) {
      auto did_compress = compressMapToConst(pos);
      wasCompressed = wasCompressed || did_compress;
      // Since trying to compress map to constant from the lowest dim, if the
      // subtree fold func is not a constant, the upper one cannot be a
      // constant.
      if (dim_prop_.at(pos).second != BaseFuncType::Constant) {
        break;
      }
    }
    return wasCompressed;
  }

 private:
  FoldFunction<Dtype>* parent_func_ = nullptr;
  fm_dim_prop dim_prop_;  // [0] --> outer most, pos=0 is index[0]

  FoldFunction<Dtype>* getFoldFunc(
      const std::deque<int64_t>& fold_dim_indices) const {
    isLegal(fold_dim_indices);
    return parent_func_->getFoldFunc(fold_dim_indices, 0);
  }

  void clear() {
    deleteSubTree(this->parent_func_);
    this->parent_func_ = nullptr;
    this->dim_prop_.clear();
  }
};

namespace FoldInfraUtils {

/**
 * @brief Method retruns a single value given a partial set of indices. This is
 * only allowed if all folds apart from those specified in pos_to_fixCoord are
 * constant, making it an easy way to get a single value while verifying that
 * there is no folding on unspecified dimensions.
 *
 * @param pos_to_fixCoord
 * @return DType
 */
template <typename Dtype>
static Dtype getSingleDataStrict(
    const FoldManager<Dtype>& fm,
    const std::map<int64_t, int64_t>& pos_to_fixCoord = {}) {
  const auto allFuncTypes = fm.getFuncType();
  for (int i = 0; i < allFuncTypes.size(); i++) {
    DT_CHECK_MSG(pos_to_fixCoord.count(i) ||
                     allFuncTypes[i] == BaseFuncType::Constant ||
                     fm.getFoldDimSize(i) == 1,
                 "Function can only be used if all dimensions without a fixed "
                 "coordinate are constant folded");
  }
  std::deque<int64_t> coordinates(allFuncTypes.size(), 0);
  for (auto& [dim, coord] : pos_to_fixCoord) coordinates.at(dim) = coord;
  return fm.getData(coordinates);
}

/**
 * @brief For an affine FM, this method returns the first (lexicographically
 * smallest) coordinate that produces the target value.
 *
 * @param fm
 * @param target
 * @param pos_to_fixCoord
 * @return std::vector<int64_t>
 */

template <typename Dtype,
          std::enable_if_t<std::is_arithmetic<Dtype>::value, bool> = true>
static std::vector<int64_t> lexiAffineSolve(
    const FoldManager<Dtype>& fm, const Dtype& target,
    const std::map<int64_t, int64_t>& pos_to_fixCoord = {}) {
  const auto nDims = fm.getNumDims();
  Dtype beta = fm.getSingleData(pos_to_fixCoord);
  if (beta == target) {  // shortcut
    std::vector<int64_t> sol(nDims, 0);
    for (auto& [dim, coord] : pos_to_fixCoord) sol.at(dim) = coord;
    return sol;
  }
  std::vector<Dtype> alphas(nDims, 0);
  std::vector<int64_t> factors(nDims, 1);
  for (int i = 0; i < nDims; i++) {
    if (pos_to_fixCoord.count(i)) continue;
    alphas.at(i) = fm.getAlpha(i);
    factors.at(i) = fm.getFoldDimSize(i);
  }
  std::optional<std::vector<int64_t>> sol =
      LexiAffineSolver(alphas, factors, beta).solve(target);
  DT_CHECK_MSG(sol, "Solution cannot be found");
  for (auto& [dim, coord] : pos_to_fixCoord) sol->at(dim) = coord;
  return sol.value();
}

/**
 * @brief This method returns the distance in steps from one coordinate to
 * another
 *
 * @param fm
 * @param startCoord
 * @param endCoord
 * @return int64_t
 */

template <typename Dtype>
static int64_t coordDistanceInSteps(const FoldManager<Dtype>& fm,
                                    const std::vector<int64_t>& startCoord,
                                    const std::vector<int64_t>& endCoord) {
  int64_t distance = 0, cumFactor = 1;
  for (int i = fm.getNumDims() - 1; i >= 0; i--) {
    distance += (endCoord.at(i) - startCoord.at(i)) * cumFactor;
    cumFactor *= fm.getFoldDimSize(i);
  }
  return distance;
}

/**
 * @brief For an affine FM, this method returns the distance in steps from the
 * fixed coordinate to the first (lexicographically smallest) coordinate that
 * produces the target value.
 *
 * @param fm
 * @param target
 * @param pos_to_fixCoord
 * @return int64_t
 */

template <typename Dtype,
          std::enable_if_t<std::is_arithmetic<Dtype>::value, bool> = true>
static int64_t lexiAffineSolveDistanceInSteps(
    const FoldManager<Dtype>& fm, const Dtype& target,
    const std::map<int64_t, int64_t>& pos_to_fixCoord = {}) {
  const std::vector<int64_t> targetCoord =
      lexiAffineSolve(fm, target, pos_to_fixCoord);
  std::vector<int64_t> startCoord(fm.getNumDims(), 0);
  for (auto& [dim, coord] : pos_to_fixCoord) startCoord.at(dim) = coord;
  return coordDistanceInSteps(fm, startCoord, targetCoord);
}

}  // namespace FoldInfraUtils

#endif
