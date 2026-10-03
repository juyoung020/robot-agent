/* -----------------------------------------------------------------------------
 * Copyright 2022 Massachusetts Institute of Technology.
 * All Rights Reserved
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  1. Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *
 *  2. Redistributions in binary form must reproduce the above copyright notice,
 *     this list of conditions and the following disclaimer in the documentation
 *     and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * Research was sponsored by the United States Air Force Research Laboratory and
 * the United States Air Force Artificial Intelligence Accelerator and was
 * accomplished under Cooperative Agreement Number FA8750-19-2-1000. The views
 * and conclusions contained in this document are those of the authors and should
 * not be interpreted as representing the official policies, either expressed or
 * implied, of the United States Air Force or the U.S. Government. The U.S.
 * Government is authorized to reproduce and distribute reprints for Government
 * purposes notwithstanding any copyright notation herein.
 * -------------------------------------------------------------------------- */
#pragma once

#include <chrono>
#include <list>
#include <map>
#include <memory>
#include <optional>
#include <ostream>
#include <string>

#include "spark_dsg/bounding_box.h"
#include "spark_dsg/color.h"
#include "spark_dsg/metadata.h"
#include "spark_dsg/scene_graph_types.h"
#include "spark_dsg/serialization/attribute_registry.h"

namespace spark_dsg {
namespace serialization {
class Visitor;
}

struct NodeAttributes;

template <typename T>
using NodeAttributeRegistration =
    serialization::AttributeRegistration<NodeAttributes, T>;

#define REGISTER_NODE_ATTRIBUTES(attr_type)                                  \
  inline static const auto registration_ =                                   \
      NodeAttributeRegistration<attr_type>(#attr_type);                      \
  const serialization::RegistrationInfo& registrationImpl() const override { \
    return registration_.info;                                               \
  }                                                                          \
  static_assert(true, "")

// TODO(nathan) handle this better
/**
 * @brief Typedef representing the semantic class of an object or other node
 */
using SemanticLabel = uint32_t;

/**
 * @brief Base node attributes.
 *
 * All nodes have a pointer to node attributes (that contain most of the useful
 * information about the node). As every node has to be spatially consistent
 * with the quantity it represents, every node must have a position.
 */
struct NodeAttributes {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  friend class serialization::Visitor;

  //! desired node pointer type
  using Ptr = std::unique_ptr<NodeAttributes>;

  //! Make a default set of attributes
  NodeAttributes();
  //! Set the node position
  explicit NodeAttributes(const Eigen::Vector3d& position);
  virtual ~NodeAttributes() = default;
  virtual NodeAttributes::Ptr clone() const;

  virtual void transform(const Eigen::Isometry3d& transform);

  //! Position of the node
  Eigen::Vector3d position;
  //! last time the place was updated (while active)
  uint64_t last_update_time_ns;
  //! whether or not the node is in the active window
  bool is_active;
  //! whether the node was observed by Hydra, or added as a prediction
  bool is_predicted;
  //! Arbitrary node metadata
  Metadata metadata;

  /**
   * @brief output attribute information
   * @param out output stream
   * @param attrs attributes to print
   * @returns original output stream
   */
  friend std::ostream& operator<<(std::ostream& out, const NodeAttributes& attrs);

  bool operator==(const NodeAttributes& other) const;

  const serialization::RegistrationInfo& registration() const {
    return registrationImpl();
  }

 protected:
  //! actually output information to the std::ostream
  virtual std::ostream& fill_ostream(std::ostream& out) const;
  //! dispatch function for serialization
  virtual void serialization_info();
  //! dispatch function for serialization
  void serialization_info() const;
  //! compute equality
  virtual bool is_equal(const NodeAttributes& other) const;

  inline static const auto registration_ =
      NodeAttributeRegistration<NodeAttributes>("NodeAttributes");

  //! get registration
  virtual const serialization::RegistrationInfo& registrationImpl() const {
    return registration_.info;
  }
};

/**
 * @brief Base class for any node with additional semantic meaning.
 */
struct SemanticNodeAttributes : public NodeAttributes {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  //! pointer type for node
  using Ptr = std::unique_ptr<SemanticNodeAttributes>;

  //! alias for semantic label
  using Label = SemanticLabel;
  // !flag for whether or not semantic label should be considered valid
  inline static constexpr Label NO_SEMANTIC_LABEL = std::numeric_limits<Label>::max();

  SemanticNodeAttributes();
  virtual ~SemanticNodeAttributes() = default;
  NodeAttributes::Ptr clone() const override;
  void transform(const Eigen::Isometry3d& transform) override;

  bool hasLabel() const;
  bool hasFeature() const;

  //! Name of the node
  std::string name;
  //! Color of the node (if it exists)
  Color color;
  //! Extents of the node (if they exists)
  BoundingBox bounding_box;
  //! semantic label of object
  SemanticLabel semantic_label;
  //! semantic feature of object
  Eigen::MatrixXf semantic_feature;

 protected:
  std::ostream& fill_ostream(std::ostream& out) const override;
  void serialization_info() override;
  bool is_equal(const NodeAttributes& other) const override;
  // registers derived attributes
  REGISTER_NODE_ATTRIBUTES(SemanticNodeAttributes);
};

/**
 * @brief Additional node attributes for an object
 *
 * In addition to the normal semantic properties, an object also potentially has
 * a pose, a collection of vertices that it is comprised of in the mesh, and a
 * bounding box.
 */
struct ObjectNodeAttributes : public SemanticNodeAttributes {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  //! desired pointer type of node
  using Ptr = std::unique_ptr<ObjectNodeAttributes>;

  //! Make a default set of attributes
  ObjectNodeAttributes();
  virtual ~ObjectNodeAttributes() = default;
  NodeAttributes::Ptr clone() const override;
  void transform(const Eigen::Isometry3d& transform) override;

  //! Whether or not the object is known (and registered)
  bool registered;
  //! rotation of object w.r.t. world (only valid when registerd)
  Eigen::Quaterniond world_R_object;

 protected:
  std::ostream& fill_ostream(std::ostream& out) const override;
  void serialization_info() override;
  bool is_equal(const NodeAttributes& other) const override;
  // registers derived attributes
  REGISTER_NODE_ATTRIBUTES(ObjectNodeAttributes);
};

/**
 * @brief Additional node attributes for a room
 * For now, a room has identical attributes to any semantic node,
 * but that may change
 */
struct RoomNodeAttributes : public SemanticNodeAttributes {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  //! desired pointer type of node
  using Ptr = std::unique_ptr<RoomNodeAttributes>;

  //!  Make a default set of attributes
  RoomNodeAttributes();
  virtual ~RoomNodeAttributes() = default;
  NodeAttributes::Ptr clone() const override;

  std::map<std::string, double> semantic_class_probabilities;

 protected:
  std::ostream& fill_ostream(std::ostream& out) const override;
  void serialization_info() override;
  bool is_equal(const NodeAttributes& other) const override;
  // registers derived attributes
  REGISTER_NODE_ATTRIBUTES(RoomNodeAttributes);
};

/**
 * @brief Additional node attributes for a place (free-space skeleton node)
 * In addition to the normal semantic properties, a place has the distance to the
 * nearest obstacle (its clearance). Map_Vla: the GVD basis points, mesh connections
 * and frontier fields were removed.
 */
struct PlaceNodeAttributes : public SemanticNodeAttributes {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  //! desired pointer type of node
  using Ptr = std::unique_ptr<PlaceNodeAttributes>;

  PlaceNodeAttributes();

  /**
   * @brief make places node attributes
   * @param distance distance to nearest obstalce
   */
  explicit PlaceNodeAttributes(double distance);
  virtual ~PlaceNodeAttributes() = default;
  NodeAttributes::Ptr clone() const override;

  //! distance to nearest obstacle
  double distance;

 protected:
  std::ostream& fill_ostream(std::ostream& out) const override;
  void serialization_info() override;
  bool is_equal(const NodeAttributes& other) const override;
  // registers derived attributes
  REGISTER_NODE_ATTRIBUTES(PlaceNodeAttributes);
};

struct AgentNodeAttributes : public NodeAttributes {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using Ptr = std::unique_ptr<AgentNodeAttributes>;
  using BowIdVector = Eigen::Matrix<uint32_t, Eigen::Dynamic, 1>;

  AgentNodeAttributes();
  AgentNodeAttributes(std::chrono::nanoseconds timestamp,
                      const Eigen::Quaterniond& world_R_body,
                      const Eigen::Vector3d& world_P_body,
                      NodeId external_key);
  virtual ~AgentNodeAttributes() = default;
  NodeAttributes::Ptr clone() const override;
  void transform(const Eigen::Isometry3d& transform) override;

  std::chrono::nanoseconds timestamp;
  Eigen::Quaterniond world_R_body;
  NodeId external_key;
  BowIdVector dbow_ids;
  Eigen::VectorXf dbow_values;

 protected:
  std::ostream& fill_ostream(std::ostream& out) const override;
  void serialization_info() override;
  bool is_equal(const NodeAttributes& other) const override;
  // registers derived attributes
  REGISTER_NODE_ATTRIBUTES(AgentNodeAttributes);
};

}  // namespace spark_dsg
