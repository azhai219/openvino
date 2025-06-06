// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>

#include "openvino/core/attribute_visitor.hpp"
#include "openvino/core/model.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/node_output.hpp"
#include "openvino/core/node_vector.hpp"
#include "openvino/op/op.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/util/sub_graph_base.hpp"
#include "openvino/op/util/variable.hpp"
#include "openvino/op/util/variable_extension.hpp"

namespace ov::intel_cpu {

class ReadValueWithSubgraph : public ov::op::util::SubGraphOp, public ov::op::util::VariableExtension {
public:
    OPENVINO_OP("ReadValueWithSubgraph", "cpu_plugin_opset", ov::op::util::SubGraphOp);

    ReadValueWithSubgraph() = default;
    ReadValueWithSubgraph(const std::shared_ptr<ov::op::util::Variable>& variable,
                          const std::shared_ptr<ov::Model>& body);
    ReadValueWithSubgraph(const std::shared_ptr<ov::op::util::Variable>& variable,
                          const std::shared_ptr<ov::Model>& body,
                          const OutputVector& args);

    std::string get_variable_id() const override;

    void set_input(const Output<Node>& value, const std::shared_ptr<op::v0::Parameter>& body_parameter);

    Output<Node> set_output(const std::shared_ptr<op::v0::Result>& body_result);

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
    bool visit_attributes(AttributeVisitor& visitor) override;
    void validate_and_infer_types() override;
};

}  // namespace ov::intel_cpu
