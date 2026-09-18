/************************************************************
* IBM Confidential
* (C) Copyright IBM Corp. 2022, 2025
************************************************************/

/*
 * Description:
 *
 */

#ifndef MAPWITH_FOLDMANAGER_HELPER_
#define MAPWITH_FOLDMANAGER_HELPER_

#include <util/print_utils.h>

#include "util/dt_exception.hpp"
#include "foldInfrastructure.h"

/**
 * @brief The class provides methods to manipulate a folded value inside a map
 * of type key
 *
 * @tparam Dkey
 * @tparam Dval
 */
template <typename Dkey, typename Dval>
class MapWithFMHelper {
 public:
  MapWithFMHelper(std::map<Dkey, FoldManager<Dval>>& managed_map)
      : key_val_(managed_map) {};

  // helper methods..

  // query about keys
  /**
   * @brief method to find number of keys
   *
   * @return int
   */
  int numKeys() const { return key_val_.size(); }

  MapWithFMHelper& operator=(const MapWithFMHelper&) = delete;
  MapWithFMHelper(const MapWithFMHelper&) = delete;
  MapWithFMHelper(MapWithFMHelper&&) noexcept = default;  // allow move ctor

  /**
   * @brief Get all Keys of the managed map
   *
   * @return std::set<Dkey>
   */
  std::set<Dkey> getAllKeys() const {
    std::set<Dkey> key_set;
    for (auto& kv : key_val_) key_set.insert(kv.first);
    return key_set;
  }

  /**
   * @brief Method to build a fold space for a new key with only constant fold
   * func
   *
   * @param new_key
   * @param fold_dim_prop
   */
  void addKeyBuildConstFoldSpace(
      Dkey new_key, const std::deque<const FoldDimProp*>& fold_dim_prop) {
    if (fold_dim_prop.size() == 0) {
      // case 1 : fold_dim_prop is empty
      if (key_val_.count(new_key) == 0) {
        // create zero order constant fold
        key_val_[new_key];
      } else {
        DT_CHECK(key_val_.at(new_key).getNumDims() == 0);
      }
    } else {
      if (key_val_.count(new_key) == 0) {
        // create all constant fold
        (key_val_)[new_key].buildAllConstantFoldSpace(fold_dim_prop);

      } else {
        DT_CHECK(key_val_.at(new_key).getNumDims() ==
                 fold_dim_prop.size());  // do nothing
      }
    }
  }

  void addKeyBuildAnyFoldSpace(
      Dkey new_key, const std::deque<const FoldDimProp*>& fold_dim_prop,
      std::deque<BaseFuncType> fold_func) {
    if (fold_dim_prop.size() == 0) {
      // case 1 : fold_dim_prop is empty
      if (key_val_.count(new_key) == 0) {
        // create zero order constant fold
        key_val_[new_key];
      } else {
        DT_CHECK(key_val_.at(new_key).getNumDims() == 0);
      }
    } else {
      if (key_val_.count(new_key) == 0) {
        // create all constant fold
        (key_val_)[new_key].buildFoldSpace(fold_dim_prop, fold_func);

      } else {
        DT_CHECK(key_val_.at(new_key).getNumDims() ==
                 fold_dim_prop.size());  // do nothing
      }
    }
  }

  /**
   * @brief Generic method to build fold func tree for a new key
   *
   * @param new_key
   * @param fold_dim_prop
   */
  void addKeyBuildFoldSpace(Dkey new_key, const fm_dim_prop& fold_dim_prop) {
    if (fold_dim_prop.size() == 0) {
      // case 1 : fold_dim_prop is empty
      if (key_val_.count(new_key) == 0) {
        // create zero order constant fold
        key_val_[new_key];
      } else {
        DT_CHECK(key_val_.at(new_key).getNumDims() == 0);
      }
    } else {
      if (key_val_.count(new_key) == 0) {
        // create all constant fold
        (key_val_)[new_key].buildFoldSpace(fold_dim_prop);
      } else {
        DT_CHECK(key_val_.at(new_key).getNumDims() ==
                 fold_dim_prop.size());  // do nothing
      }
    }
  }

  /**
   * @brief Generic method to build fold func tree for a new key
   *
   * @param new_key
   * @param props
   * @param func_base_types
   */
  void addKeyBuildFoldSpace(Dkey new_key, std::deque<const FoldDimProp*>& props,
                            const std::deque<BaseFuncType>& func_base_types) {
    DT_CHECK(props.size() == func_base_types.size());
    if (props.size() == 0) {
      // case 1 : props is empty
      if (key_val_.count(new_key) == 0) {
        // create zero order constant fold
        key_val_[new_key];
      } else {
        DT_CHECK(key_val_.at(new_key).getNumDims() == 0);
      }
    } else {
      if (key_val_.count(new_key) == 0) {
        // create all constant fold
        (key_val_)[new_key].buildFoldSpace(props, func_base_types);
      } else {
        DT_CHECK(key_val_.at(new_key).getNumDims() ==
                 func_base_types.size());  // do nothing
      }
    }
  }

  /**
   * @brief adds a new fold dimension for each key
   *
   * @param prop
   * @param func_base_type
   * @param pos
   */
  void buildDim(const FoldDimProp* prop, BaseFuncType func_base_type,
                int pos = 0) {
    for (auto& kv : key_val_) {
      kv.second.buildDim(prop, func_base_type, pos);
    }
  }

  /**
   * @brief Rebuild that fold dimension at pos
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
    bool retVal = true;
    for (auto& kv : key_val_) {
      auto rebuildVal = kv.second.rebuildDim(pos, func_base_type);
      retVal = retVal & rebuildVal;
    }
    return retVal;
  }

  /**
   * @brief Get Data For a given Key in the folded space specified by
   * fold_dim_indices
   *
   * @param key
   * @param fold_dim_indices
   * @return Dval
   */
  Dval getDataForKey(Dkey key, std::deque<int64_t> fold_dim_indices) const {
    DT_CHECK(key_val_.count(key));
    return key_val_.at(key).getData(fold_dim_indices);
  }

  /**
   * @brief Get the All Data For Key object
   *
   * @param key
   * @return std::set<Dval>
   */
  std::set<Dval> getAllDataForKey(Dkey key) const {
    DT_CHECK(key_val_.count(key));

    std::vector<std::pair<std::deque<int64_t>, Dval>> data_and_coord;
    key_val_.at(key).getDataAndFoldCoordinates(data_and_coord);

    std::set<Dval> retSet;
    for (auto pair : data_and_coord) {
      auto data = pair.second;
      retSet.insert(data);
    }

    return retSet;
  }

  /**
   * @brief  Insert data in the folded space for a given key
   *
   * @param key
   * @param new_data
   * @param fold_dim_indices
   */
  void insertDataForKey(Dkey key, const Dval& new_data,
                        std::deque<int64_t> fold_dim_indices) {
    DT_CHECK(key_val_.count(key));
    key_val_.at(key).insertData(new_data, fold_dim_indices);
  }

  /**
   * @brief  Get the Data For Key object in the folded space
   *
   * @tparam Args
   * @param key
   * @param list
   * @return Dval
   */
  template <typename... Args, class Enable = std::enable_if_t<(
                                  ... && std::is_convertible_v<Args, int64_t>)>>
  Dval getDataForKey(Dkey key, const Args&... list) const {
    std::deque<int64_t> fold_dim_indices{list...};
    return getDataForKey(key, fold_dim_indices);
  }

  /**
   * @brief Insert data in the folded space for a given key
   *
   * @tparam Args
   * @param key
   * @param new_data
   * @param list
   */
  template <typename... Args, class Enable = std::enable_if_t<(
                                  ... && std::is_convertible_v<Args, int64_t>)>>
  void insertDataForKey(Dkey key, const Dval& new_data, const Args&... list) {
    std::deque<int64_t> fold_dim_indices{list...};
    insertDataForKey(key, new_data, fold_dim_indices);
  }

  /**
   * @brief  Insert alpha in the affine folded space for a given key
   *
   * @param key
   * @param new_data
   * @param fold_dim_index
   */
  void insertAlphaForKey(Dkey key, Dval& new_data, int64_t fold_dim_index) {
    DT_CHECK(key_val_.count(key));
    key_val_.at(key).insertAlpha(new_data, fold_dim_index);
  }

  /**
   * @brief Get alpha For a given Key in the affine folded space specified by
   * fold_dim_index
   *
   * @param key
   * @param fold_dim_index
   * @return Dval
   */
  Dval getAlphaForKey(Dkey key, int64_t fold_dim_index) const {
    DT_CHECK(key_val_.count(key));
    return key_val_.at(key).getAlpha(fold_dim_index);
  }

  /**
   * @brief  Insert beta in the affine folded space for a given key
   *
   * @param key
   * @param new_data
   * @param fold_dim_index
   */
  void insertBetaForKey(Dkey key, Dval& new_data, int64_t fold_dim_index) {
    DT_CHECK(key_val_.count(key));
    key_val_.at(key).insertBeta(new_data, fold_dim_index);
  }

  /**
   * @brief Get beta For a given Key in the affine folded space specified by
   * fold_dim_index
   *
   * @param key
   * @param fold_dim_index
   * @return Dval
   */
  Dval getBetaForKey(Dkey key, int64_t fold_dim_index) const {
    DT_CHECK(key_val_.count(key));
    return key_val_.at(key).getBeta(fold_dim_index);
  }

  /**
   * @brief This method copies sub folded space. We allow skipping of dimensions
   * in "rhs_data". Sub fold space is determined by ignoring dimensions
   * specified by key of ignore_dim_idx_and_dim_value. The coordinate of each
   * ignored dimensions is specified by value in ignore_dim_idx_and_dim_val.
   *
   * @param rhs_data
   * @param fold_dim_prop
   * @param ignore_dim_idx_and_dim_val
   */
  void copy(const std::map<Dkey, FoldManager<Dval>>& rhs_data,
            const std::deque<const FoldDimProp*>& fold_dim_prop,
            std::map<int, int> ignore_dim_idx_and_dim_val = {}) {
    key_val_.clear();

    for (auto& kv : rhs_data) {
      // build initial fold space for each new key using constant fold function
      addKeyBuildConstFoldSpace(kv.first, fold_dim_prop);

      if (ignore_dim_idx_and_dim_val.empty())
        key_val_[kv.first] = kv.second;
      else
        key_val_[kv.first].copyFoldedSubSpace(kv.second,
                                              ignore_dim_idx_and_dim_val);
    }
  }

  /**
   * @brief Zeroth fold copy method which copies managed object with no folded
   * dimensions, i.e., all FoldManager<Dval> are managed as zeroth order
   * constant fold
   *
   * @param rhs_data
   */
  void copy(const std::map<Dkey, FoldManager<Dval>>& rhs_data) {
    key_val_.clear();
    for (auto& kv : rhs_data) {
      DT_CHECK_MSG(kv.second.hasZeroFoldDim(),
                   "should be zeroth order fold type");
      key_val_[kv.first] = kv.second;
    }
  }

  /**
   * @brief Copy only a subset of key values from the reference map with
   * FMHelper. This method has similar functionality as copy but the user has
   * additional control over the keys to be included from the reference object.
   *
   * @param rhs_data : Reference object
   * @param fold_dim_prop
   * @param keys : Set of keys to copy from the reference object
   * @param ignore_dim_idx_and_dim_val
   */
  void copyForSelectedKeys(const std::map<Dkey, FoldManager<Dval>>& rhs_data,
                           const std::deque<const FoldDimProp*>& fold_dim_prop,
                           std::set<Dkey> keys,
                           std::map<int, int> ignore_dim_idx_and_dim_val = {}) {
    key_val_.clear();

    for (auto& kv : rhs_data) {
      auto eligibleKey = keys.find(kv.first) != keys.end();
      if (eligibleKey) {
        // * build initial fold space for each new key using constant fold
        // * function
        addKeyBuildConstFoldSpace(kv.first, fold_dim_prop);

        if (ignore_dim_idx_and_dim_val.empty())
          key_val_[kv.first] = kv.second;
        else
          key_val_[kv.first].copyFoldedSubSpace(kv.second,
                                                ignore_dim_idx_and_dim_val);
      }
    }
  }

  /**
   * @brief This method creates a map entry for a new key by cloning the fold
   * space and values from an existing key.
   *
   * @param newKey : The new key for which the fold space and values will be
   * cloned
   * @param existingKey : The existing key of which the clone will be created.
   */
  void cloneEntryForNewKey(Dkey newKey, Dkey existingKey) {
    DT_CHECK(key_val_.find(existingKey) != key_val_.end());

    // * Build the dimensions of the new FM entry before copy
    fm_dim_prop allDimProp;
    key_val_.at(existingKey).getAllDimProFromPos(0, allDimProp);

    std::deque<const FoldDimProp*> foldPropQ;
    std::deque<BaseFuncType> foldFuncQ;
    for (auto pair : allDimProp) {
      foldPropQ.push_back(pair.first);
      foldFuncQ.push_back(pair.second);
    }

    key_val_[newKey].buildFoldSpace(foldPropQ, foldFuncQ);

    // * Call copy to replicate the data from the existing entry
    key_val_.at(newKey).copyFoldedSubSpace(key_val_.at(existingKey), {});
  }

  /**
   * @brief Get the Ref object
   *
   * @return const std::map<Dkey, FoldManager<Dval>>&
   */
  const std::map<Dkey, FoldManager<Dval>>& getRef() const { return key_val_; }

  /**
   * @brief Function to insert a map at particular point in fold space given by
   * list...
   *
   * @tparam Args
   * @param new_map
   * @param list
   */
  template <typename... Args, class Enable = std::enable_if_t<(
                                  ... && std::is_convertible_v<Args, int64_t>)>>
  void insertMapData(const std::map<Dkey, Dval>& new_map, const Args&... list) {
    for (auto& kv : new_map) {
      key_val_[kv.first].insertData(kv.second, list...);
    }
  }

  /**
   * @brief Function to insert a map at particular point in fold space given by
   * coord
   *
   * @tparam Args
   * @param new_map
   * @param coord
   */
  void insertMapData(const std::map<Dkey, Dval>& new_map,
                     std::deque<int64_t> coord) {
    for (auto& kv : new_map) {
      key_val_[kv.first].insertData(kv.second, coord);
    }
  }

  /**
   * @brief Get the Map Data object
   *
   * @tparam Args
   * @param list
   * @return std::map<Dkey, Dval>
   */
  template <typename... Args>
  std::map<Dkey, Dval> getMapData(const Args&... list) const {
    std::map<Dkey, Dval> my_map;
    for (auto& kv : key_val_) {
      my_map[kv.first] = kv.second.getData(list...);
    }
    return my_map;
  }

  /**
   * @brief Get the Map Data object
   *
   * @param coord
   * @return std::map<Dkey, Dval>
   */
  std::map<Dkey, Dval> getMapData(std::deque<int64_t> coord) const {
    std::map<Dkey, Dval> my_map;
    for (auto& kv : key_val_) {
      my_map[kv.first] = kv.second.getData(coord);
    }
    return my_map;
  }

  /**
   * @brief Check legality
   *
   * @return true
   * @return false
   */
  bool isLegal() {
    std::vector<int> foldDimSize;
    for (auto& kv : key_val_) {
      if (foldDimSize.empty()) {
        foldDimSize = kv.second.getFoldSpaceSize();
      } else {
        if (foldDimSize == kv.second.getFoldSpaceSize()) return false;
        // DT_ERROR(
        //     " Illegal folding of values : all values should have same fold "
        //     "dimensionality\n");
      }
    }
    return true;
  }

  void getNumUniqueCoordsInEachFold(
      std::vector<int64_t>& unique_data_coords) const {
    for (auto& kv : key_val_) {
      auto myFoldSpaceSize = kv.second.getFoldSpaceSize();
      // if const fold, make unique_data_coords to 1 i.e, use only coord 0..
      for (int idx = 0; idx < myFoldSpaceSize.size(); idx++) {
        if (kv.second.getFuncType(idx) == BaseFuncType::Constant) {
          myFoldSpaceSize.at(idx) = 1;
        }
      }
      if (unique_data_coords.empty()) {
        unique_data_coords = myFoldSpaceSize;
      } else {
        if (unique_data_coords != myFoldSpaceSize) {
          bool is_super_set_curr = false;
          bool is_super_set_old = false;
          DT_CHECK(unique_data_coords.size() == myFoldSpaceSize.size());
          for (int idx = 0; idx < myFoldSpaceSize.size(); idx++) {
            if (myFoldSpaceSize.at(idx) == unique_data_coords.at(idx)) continue;

            if (myFoldSpaceSize.at(idx) > unique_data_coords.at(idx))
              is_super_set_curr = true;
            else
              is_super_set_old = true;
          }

          if (is_super_set_curr && is_super_set_old)
            DT_ERROR(
                " Illegal folding of values : all values should have same fold "
                "dimensionality\n");

          if (is_super_set_curr) unique_data_coords = myFoldSpaceSize;
        }
      }
    }
  }

  /**
   * @brief Get the Common Flattened Coordinates object
   *
   * @param coordinates
   */
  void getCommonFlattenedCoordinates(
      std::vector<std::deque<int64_t>>& coordinates) const {
    // check if fold dim size is the same
    std::vector<int64_t> foldDimSize;
    getNumUniqueCoordsInEachFold(foldDimSize);

    if (foldDimSize.size() > 0) {
      auto total_count = 1;
      for (auto& size : foldDimSize) total_count *= size;

      coordinates.resize(total_count);
      for (int idx = 0; idx < total_count; idx++) {
        coordinates.at(idx).resize(foldDimSize.size());
      }

      int repeat_factor = 1;
      for (int dim_idx = 0; dim_idx < foldDimSize.size(); dim_idx++) {
        for (int coord_idx = 0; coord_idx < total_count; coord_idx++) {
          coordinates.at(coord_idx).at(dim_idx) =
              (coord_idx / repeat_factor) % foldDimSize.at(dim_idx);
        }
        repeat_factor *= foldDimSize.at(dim_idx);
      }
    }
  }

  /**
   * @brief Method prints the underlying managed object
   *
   * @param var_name
   * @param so
   * @param use_comma
   */
  void print(std::ostream& json, std::string var_name, std::string so,
             bool use_comma = true, bool compressed = false) {
    auto QUOTE = [](std::string str) { return "\"" + str + "\""; };
    json << so << QUOTE(var_name) << " : {\n";
    auto so2 = so + "  ";
    std::vector<std::deque<int64_t>> folddim_coord;
    getCommonFlattenedCoordinates(folddim_coord);

    std::vector<int64_t> unique_data_coords;
    getNumUniqueCoordsInEachFold(unique_data_coords);
    if (folddim_coord.size()) {
      for (int idx = 0; idx < folddim_coord.size(); idx++) {
        json << so2 << "\"[" << PrintUtil::printDeque(folddim_coord.at(idx))
             << "]\" : ";
        if (compressed) {
          json << "["
               << PrintUtil::printMap(getMapData(folddim_coord.at(idx)), true)
               << "]";
        } else {
          json << "{"
               << PrintUtil::printMap(getMapData(folddim_coord.at(idx)))
               << "}";
        }
        if (idx != folddim_coord.size() - 1)
          json << "," << std::endl;
        else
          json << std::endl;
      }
      json << so2 << "},\n";
      if (compressed) {
        json << so << QUOTE(var_name + "Key_") << " : [";
        int idx = 0;
        for (auto& kv : key_val_) {
          json << kv.first;
          if (idx < key_val_.size() - 1) json << ", ";
          idx++;
        }
        json << "],\n";
      }

      json << so << QUOTE(var_name + "Prop_") << " : {\n";
      printFoldProp(json, so + "  ", true, compressed);
    } else {
      json << so2 << QUOTE("[-1]") << ": {" << PrintUtil::printMap(getMapData())
           << "}\n";
    }
    if (use_comma)
      json << so << "},\n";
    else
      json << so << "}\n";
  }

  /**
   * @brief Get the Fold Space Size object
   *
   * @return std::vector<int>
   */
  std::vector<int64_t> getFoldSpaceSize() {
    std::vector<int64_t> foldDimSize;
    for (auto& kv : key_val_) {
      if (foldDimSize.empty()) {
        foldDimSize = kv.second.getFoldSpaceSize();
      } else {
        if (foldDimSize != kv.second.getFoldSpaceSize())
          DT_ERROR(
              "Unexpected : fold dimenionality cannot differ across keys in "
              "key_val_\n");
      }
    }
    return foldDimSize;
  }

  /**
   * @brief Get the Num Dims object
   *
   * @return int
   */
  int getNumDims() {
    int num_dims = -1;
    for (auto& kv : key_val_) {
      if (num_dims == -1) {
        num_dims = kv.second.getNumDims();
      } else {
        if (num_dims != kv.second.getNumDims())
          DT_ERROR(
              "Unexpected : fold dimenionality cannot differ across keys in "
              "key_val_\n");
      }
    }
    return num_dims;
  }

  void printFoldProp(std::ostream& out, std::string ps = "", bool use_comma = true, bool compressed = false) {
    auto QUOTE = [](std::string str) { return "\"" + str + "\""; };
    int count = 0;
    for (auto& kv : key_val_) {
      out << ps << QUOTE(std::to_string(kv.first)) << " : {\n";
      kv.second.printMetaData(out, ps + "  ", false, compressed);
      out << ps << "}";
      if (count < key_val_.size() - 1) out << ",";
      out << "\n";
      count++;
    }
  }

  /**
   * @brief Removes an fold manager entry from MapWithFM with the given key
   *
   * @param existingKey : Key to the existing entry
   */
  void removeKey(Dkey existingKey) {
    DT_CHECK(key_val_.find(existingKey) != key_val_.end());
    key_val_.erase(existingKey);
  }

  /**
   * @brief Method checks presence of a key
   *
   * @param key
   * @return true
   * @return false
   */
  bool isKeyPresent(Dkey key) const { return (key_val_.count(key) > 0); }

  /**
   * @brief Get the Value object for a given key
   *
   * @param key
   * @return FoldManager<Dval>&
   */
  FoldManager<Dval>& getVal(Dkey key) const {
    if (!key_val_.count(key)) DT_ERROR("Illegal access : key is missing");
    return key_val_.at(key);
  }

  /**
   * @brief Method checks if the managed object of all keys has no fold
   * dimensions, meaning the object is mannaged as a zeroth order constant fold
   *
   * @return true
   * @return false
   */
  bool hasZeroFoldDim() const {
    bool is_zero_fold = true;
    for (auto& kv : key_val_)
      if (!kv.second.hasZeroFoldDim()) {
        is_zero_fold = false;
        break;
      }
    return is_zero_fold;
  }

  /**
   * @brief Get the folded Func Type For the particular key object at given
   * "pos"
   *
   * @param key
   * @param pos
   * @return BaseFuncType
   */
  BaseFuncType getFuncTypeForkey(Dkey key, int pos) const {
    if (!key_val_.count(key)) DT_ERROR("Illegal access : key is missing");
    return key_val_.at(key).getFuncType(pos);
  }

  /**
   * @brief Get all folded Func Type For the particular key object
   *
   * @param key
   * @param pos
   * @return   std::deque<BaseFuncType>
   */
  std::deque<BaseFuncType> getFuncTypeForkey(Dkey key) const {
    if (!key_val_.count(key)) DT_ERROR("Illegal access : key is missing");
    return key_val_.at(key).getFuncType();
  }

  bool isAllFoldsConstant(Dkey key) {
    auto base_fold_func_vec = getFuncTypeForkey(key);
    bool all_const = true;

    for (auto base_func : base_fold_func_vec) {
      if (base_func != BaseFuncType::Constant) {
        all_const = false;
        break;
      }
    }
    return all_const;
  }

  [[nodiscard]] bool isALLKeyConstFolded(int pos) const {
    bool all_const = true;
    for (auto& [key, fm_dval] : key_val_) {
      if (fm_dval.getFuncType(pos) != BaseFuncType::Constant) {
        all_const = false;
        break;
      }
    }
    return all_const;
  }

  /**
   * @brief Method compresses fm variables for all keys to use const fold
   * function from pos to the lowest dim if possible. If compression was
   * performed it returns true. Compresses Maps to Const
   *
   * @param pos
   * @return true
   * @return false
   */
  bool compressMapToConst(int pos) {
    bool wasCompressed = false;
    for (auto& [key, fm_dval] : key_val_) {
      auto did_compress = fm_dval.compressMapToConst(pos);
      wasCompressed = wasCompressed || did_compress;
    }
    return wasCompressed;
  }

  /**
   * @brief Method compresses fm variables for all keys to use const fold
   * function for all dims (starting from the lowest dim) if possible. If
   * at least one compression was performed it returns true. Compresses Maps to
   * Const
   *
   * @return true
   * @return false
   */
  bool compressMapToConstForAllDims() {
    bool wasCompressed = false;
    for (auto& [key, fm_dval] : key_val_) {
      auto did_compress = fm_dval.compressMapToConstForAllDims();
      wasCompressed = wasCompressed || did_compress;
    }
    return wasCompressed;
  }

 private:
  std::map<Dkey, FoldManager<Dval>>&
      key_val_;  // a reference of variable managed
};

#endif
