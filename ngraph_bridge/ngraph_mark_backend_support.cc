/*******************************************************************************
 * Copyright 2019 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *******************************************************************************/

#include "ngraph_bridge/ngraph_mark_backend_support.h"

namespace ng = ngraph;
using namespace std;

namespace tensorflow {

namespace ngraph_bridge {

Status GetBackendSupportInfoForTFSubgraph(
    const ng::runtime::Backend* op_backend, GraphDef* g,
    std::map<std::string, bool>& result_map) {
  result_map.clear();
  // TODO: fill this function
  // Call translate graph. Then call GetBackendSupportInfoForNgfunction

  // TODO, populate possible_to_translate correctly later
  bool possible_to_translate = false;
  if (possible_to_translate) {
    // call translategraph etc and GetBackendSupportInfoForNgfunction
    // TODO
    return errors::Internal("Unimplemented: Call TranslateGraph");
  } else {
    unique_ptr<Graph> graph_ptr(new Graph(OpRegistry::Global()));
    GraphConstructorOptions opts;
    opts.allow_internal_ops = true;
    TF_RETURN_IF_ERROR(ConvertGraphDefToGraph(opts, *g, graph_ptr.get()));
    bool supported_op;
    for (auto node : graph_ptr->nodes()) {
        if (NodeIsMarkedForClustering(node)){
      TF_RETURN_IF_ERROR(IsSupportedByBackend(node, op_backend, supported_op));
      result_map.insert({node->name(), supported_op});
        }
    }
  }
  return Status::OK();
}

Status GetBackendSupportInfoForNgfunction(
    const ng::runtime::Backend* op_backend,
    const shared_ptr<ng::Function>& ng_function,
    std::map<std::string, bool>& result_map) {
  result_map.clear();
  // TODO: fill this function

  return Status::OK();
}

Status IsSupportedByBackend(
    const Node* node, const ng::runtime::Backend* op_backend,
    std::map<std::string, std::set<std::shared_ptr<ngraph::Node>>>&
        TFtoNgraphOpMap,
    bool& is_supported) {
  is_supported = true;

  auto ng_op = TFtoNgraphOpMap.find(node->type_string());
  if (ng_op == TFtoNgraphOpMap.end()) {
    return errors::Internal("TF Op is not found in the map: ",
                            node->type_string());
  }

  // Loop through the ngraph op list to query
  for (auto it = ng_op->second.begin(); it != ng_op->second.end(); it++) {
    // Pass ngraph node to check if backend supports this op
    auto ret = op_backend->is_supported(**it);
    if (!ret) {
      is_supported = false;
      return Status::OK();
    }
  }
  return Status::OK();
}

// Check if op is supported by backend using is_supported API
Status IsSupportedByBackend(const Node* node,
                            const ng::runtime::Backend* op_backend,
                            bool& is_supported) {
  // Constant Op, ReluGrad Op do not have default Constructor
  // in ngraph, so passing a dummy node
  auto constant = ngraph::op::Constant::create(ngraph::element::f32,
                                               ngraph::Shape{}, {2.0f});
  auto shape_a = ngraph::Shape{2, 5};
  auto A = make_shared<ngraph::op::Parameter>(ngraph::element::f32, shape_a);
  auto delta_val =
      make_shared<ngraph::op::Parameter>(ngraph::element::f32, shape_a);
  auto relu = make_shared<ngraph::op::ReluBackprop>(A, delta_val);
  // Map:: TF ops to NG Ops to track if all the Ngraph ops
  // are supported by backend
  // Update this Map if a new TF Op translation is
  // implemented or a new Ngraph Op has been added
  static std::map<std::string, std::set<shared_ptr<ng::Node>>> TFtoNgraphOpMap{
      {"Abs", {std::make_shared<ngraph::op::Abs>()}},
      {"Add", {std::make_shared<ngraph::op::Add>()}},
      {"AddN", {std::make_shared<ngraph::op::Add>()}},
      {"AddV2", {std::make_shared<ngraph::op::Add>()}},
      {"Any", {std::make_shared<ngraph::op::Any>()}},
      {"All", {std::make_shared<ngraph::op::All>()}},
      {"ArgMax", {std::make_shared<ngraph::op::ArgMax>()}},
      {"ArgMin", {std::make_shared<ngraph::op::ArgMin>()}},
      {"AvgPool", {std::make_shared<ngraph::op::AvgPool>()}},
      {"AvgPoolGrad", {std::make_shared<ngraph::op::AvgPoolBackprop>()}},
      {"BatchMatMul",
       {std::make_shared<ngraph::op::BatchMatMul>(),
        std::make_shared<ngraph::op::Dot>(),
        std::make_shared<ngraph::op::Reshape>(),
        std::make_shared<ngraph::op::Slice>(),
        std::make_shared<ngraph::op::Concat>()}},
      {"BatchMatMulV2",
       {std::make_shared<ngraph::op::BatchMatMul>(),
        std::make_shared<ngraph::op::Dot>(),
        std::make_shared<ngraph::op::Reshape>(),
        std::make_shared<ngraph::op::Slice>(),
        std::make_shared<ngraph::op::Concat>()}},
      {"BiasAdd",
       {std::make_shared<ngraph::op::Add>(),
        std::make_shared<ngraph::op::Broadcast>()}},
      {"BiasAddGrad", {std::make_shared<ngraph::op::Sum>()}},
      {"Cast", {std::make_shared<ngraph::op::Convert>()}},
      {"ConcatV2", {std::make_shared<ngraph::op::Concat>()}},
      {"Const", {constant}},
      {"Conv2D",
       {std::make_shared<ngraph::op::Reshape>(),
        std::make_shared<ngraph::op::Convolution>()}},
      {"Conv2DBackpropFilter",
       {std::make_shared<ngraph::op::ConvolutionBackpropFilters>(),
        std::make_shared<ngraph::op::Reshape>()}},
      {"Conv2DBackpropInput",
       {std::make_shared<ngraph::op::ConvolutionBackpropData>(),
        std::make_shared<ngraph::op::Reshape>()}},
      {"Conv3D",
       {std::make_shared<ngraph::op::Convolution>(),
        std::make_shared<ngraph::op::Reshape>()}},
      {"Cos", {std::make_shared<ngraph::op::Cos>()}},
      {"DepthToSpace", {std::make_shared<ngraph::op::Reshape>()}},
      {"DepthwiseConv2dNative",
       {std::make_shared<ngraph::op::Slice>(),
        std::make_shared<ngraph::op::Convolution>(),
        std::make_shared<ngraph::op::Concat>(),
        std::make_shared<ngraph::op::Reshape>()}},
      {"Dequantize", {std::make_shared<ngraph::op::Dequantize>()}},
      {"Equal", {std::make_shared<ngraph::op::Equal>()}},
      {"Exp", {std::make_shared<ngraph::op::Exp>()}},
      {"ExpandDims", {std::make_shared<ngraph::op::Reshape>()}},
      {"Fill", {std::make_shared<ngraph::op::Broadcast>()}},
      {"Floor", {std::make_shared<ngraph::op::Floor>()}},
      {"FloorDiv", {std::make_shared<ngraph::op::Divide>()}},
      {"FloorMod",
       {std::make_shared<ngraph::op::Divide>(),
        std::make_shared<ngraph::op::Subtract>(),
        std::make_shared<ngraph::op::Multiply>()}},
      {"FusedBatchNorm",
       {std::make_shared<ngraph::op::BatchNormTraining>(),
        std::make_shared<ngraph::op::GetOutputElement>(), constant,
        std::make_shared<ngraph::op::Multiply>(),
        std::make_shared<ngraph::op::BatchNormInference>()}},
      {"FusedBatchNormV2",
       {std::make_shared<ngraph::op::BatchNormTraining>(),
        std::make_shared<ngraph::op::GetOutputElement>(), constant,
        std::make_shared<ngraph::op::Multiply>(),
        std::make_shared<ngraph::op::BatchNormInference>()}},
      {"FusedBatchNormV3",
       {std::make_shared<ngraph::op::BatchNormTraining>(),
        std::make_shared<ngraph::op::GetOutputElement>(), constant,
        std::make_shared<ngraph::op::Multiply>(),
        std::make_shared<ngraph::op::BatchNormInference>()}},
      {"FusedBatchNormGrad",
       {constant, std::make_shared<ngraph::op::GetOutputElement>(),
        std::make_shared<ngraph::op::BatchNormTrainingBackprop>()}},
      {"FusedBatchNormGradV3",
       {constant, std::make_shared<ngraph::op::GetOutputElement>(),
        std::make_shared<ngraph::op::BatchNormTrainingBackprop>()}},
      {"GatherNd", {std::make_shared<ngraph::op::GatherND>()}},
      {"GatherV2", {std::make_shared<ngraph::op::Gather>()}},
      {"_FusedConv2D",
       {std::make_shared<ngraph::op::Reshape>(),
        std::make_shared<ngraph::op::Convolution>(), constant,
        std::make_shared<ngraph::op::Minimum>(),
        std::make_shared<ngraph::op::Relu>(),
        std::make_shared<ngraph::op::Broadcast>(),
        std::make_shared<ngraph::op::Add>(),
        std::make_shared<ngraph::op::BatchNormInference>()}},
      {"_FusedMatMul",
       {std::make_shared<ngraph::op::Dot>(),
        std::make_shared<ngraph::op::Relu>(),
        std::make_shared<ngraph::op::Broadcast>(),
        std::make_shared<ngraph::op::Add>(), constant,
        std::make_shared<ngraph::op::Minimum>(),
        std::make_shared<ngraph::op::Reshape>()}},
      {"Greater", {std::make_shared<ngraph::op::Greater>()}},
      {"GreaterEqual", {std::make_shared<ngraph::op::GreaterEq>()}},
      {"Identity", {}},
      {"IsFinite",
       {constant, std::make_shared<ngraph::op::NotEqual>(),
        std::make_shared<ngraph::op::Equal>(),
        std::make_shared<ngraph::op::And>()}},
      {"L2Loss",
       {constant, std::make_shared<ngraph::op::Multiply>(),
        std::make_shared<ngraph::op::Sum>(),
        std::make_shared<ngraph::op::Divide>()}},
      {"LogSoftmax",
       {std::make_shared<ngraph::op::Broadcast>(),
        std::make_shared<ngraph::op::Max>(),
        std::make_shared<ngraph::op::Subtract>(),
        std::make_shared<ngraph::op::Exp>(),
        std::make_shared<ngraph::op::Log>(),
        std::make_shared<ngraph::op::Sum>()}},
      {"Less", {std::make_shared<ngraph::op::Less>()}},
      {"LessEqual", {std::make_shared<ngraph::op::LessEq>()}},
      {"Log", {std::make_shared<ngraph::op::Log>()}},
      {"LogicalAnd", {std::make_shared<ngraph::op::And>()}},
      {"LogicalNot", {std::make_shared<ngraph::op::Not>()}},
      {"LogicalOr", {std::make_shared<ngraph::op::Or>()}},
      {"MatMul",
       {std::make_shared<ngraph::op::Reshape>(),
        std::make_shared<ngraph::op::Dot>()}},
      {"Max", {std::make_shared<ngraph::op::Max>()}},
      {"Maximum", {std::make_shared<ngraph::op::Maximum>()}},
      {"MaxPool",
       {std::make_shared<ngraph::op::Reshape>(),
        std::make_shared<ngraph::op::MaxPool>()}},
      {"MaxPool3D",
       {std::make_shared<ngraph::op::Reshape>(),
        std::make_shared<ngraph::op::MaxPool>()}},
      {"MaxPoolGrad",
       {std::make_shared<ngraph::op::Reshape>(),
        std::make_shared<ngraph::op::MaxPoolBackprop>()}},
      {"Mean",
       {std::make_shared<ngraph::op::Reshape>(), constant,
        std::make_shared<ngraph::op::Divide>(),
        std::make_shared<ngraph::op::Sum>()}},
      {"Min", {std::make_shared<ngraph::op::Min>()}},
      {"Minimum", {std::make_shared<ngraph::op::Minimum>()}},
      {"Mul", {std::make_shared<ngraph::op::Multiply>()}},
      {"Neg", {std::make_shared<ngraph::op::Negative>()}},
      {"OneHot",
       {std::make_shared<ngraph::op::OneHot>(),
        std::make_shared<ngraph::op::Convert>(),
        std::make_shared<ngraph::op::Select>(),
        std::make_shared<ngraph::op::Reshape>(),
        std::make_shared<ngraph::op::Broadcast>()}},
      {"Pack",
       {std::make_shared<ngraph::op::Concat>(),
        std::make_shared<ngraph::op::Reshape>()}},
      {"Pad", {constant, std::make_shared<ngraph::op::Pad>()}},
      {"Pow", {std::make_shared<ngraph::op::Power>()}},
      {"PreventGradient", {}},
      {"Prod", {std::make_shared<ngraph::op::Product>()}},
      {"QuantizeAndDequantizeV2",
       {constant, std::make_shared<ngraph::op::Quantize>(),
        std::make_shared<ngraph::op::Dequantize>()}},
      // Next few are CPU only ops
      {"QuantizedAvgPool",
       {std::make_shared<ngraph::op::AvgPool>(),
        std::make_shared<ngraph::op::Reshape>()}},
      {"QuantizedConcat",
       {constant, std::make_shared<ngraph::op::Reshape>(),
        std::make_shared<ngraph::op::Min>(),
        std::make_shared<ngraph::op::Max>(),
        std::make_shared<ngraph::op::Abs>(),
        std::make_shared<ngraph::op::Minimum>(),
        std::make_shared<ngraph::op::Maximum>(),
        std::make_shared<ngraph::op::Divide>(),
        std::make_shared<ngraph::op::Dequantize>(),
        std::make_shared<ngraph::op::Quantize>(),
        std::make_shared<ngraph::op::Concat>()}},
      {"QuantizedConcatV2",
       {constant, std::make_shared<ngraph::op::Reshape>(),
        std::make_shared<ngraph::op::Min>(),
        std::make_shared<ngraph::op::Max>(),
        std::make_shared<ngraph::op::Abs>(),
        std::make_shared<ngraph::op::Minimum>(),
        std::make_shared<ngraph::op::Maximum>(),
        std::make_shared<ngraph::op::Divide>(),
        std::make_shared<ngraph::op::Dequantize>(),
        std::make_shared<ngraph::op::Quantize>(),
        std::make_shared<ngraph::op::Concat>()}},
      {"QuantizedConv2DWithBiasAndReluAndRequantize",
       {constant, std::make_shared<ngraph::op::Broadcast>(),
        std::make_shared<ngraph::op::Abs>(),
        std::make_shared<ngraph::op::Minimum>(),
        std::make_shared<ngraph::op::Maximum>(),
        std::make_shared<ngraph::op::Divide>(),
        std::make_shared<ngraph::op::Multiply>(),
        std::make_shared<ngraph::op::Quantize>(),
        std::make_shared<ngraph::op::QuantizedConvolutionBias>(),
        std::make_shared<ngraph::op::Reshape>()}},
      {"QuantizedConv2DWithBiasAndRequantize",
       {constant, std::make_shared<ngraph::op::Broadcast>(),
        std::make_shared<ngraph::op::Abs>(),
        std::make_shared<ngraph::op::Minimum>(),
        std::make_shared<ngraph::op::Maximum>(),
        std::make_shared<ngraph::op::Divide>(),
        std::make_shared<ngraph::op::Multiply>(),
        std::make_shared<ngraph::op::Quantize>(),
        std::make_shared<ngraph::op::QuantizedConvolutionBias>(),
        std::make_shared<ngraph::op::Reshape>()}},
      {"QuantizedConv2DWithBiasSignedSumAndReluAndRequantize",
       {constant, std::make_shared<ngraph::op::Broadcast>(),
        std::make_shared<ngraph::op::Abs>(),
        std::make_shared<ngraph::op::Minimum>(),
        std::make_shared<ngraph::op::Maximum>(),
        std::make_shared<ngraph::op::Divide>(),
        std::make_shared<ngraph::op::Multiply>(),
        std::make_shared<ngraph::op::Quantize>(),
        std::make_shared<ngraph::op::QuantizedConvolutionBiasSignedAdd>(),
        std::make_shared<ngraph::op::Reshape>(),
        std::make_shared<ngraph::op::Convert>()}},
      {"QuantizedConv2DWithBiasSumAndReluAndRequantize",
       {constant, std::make_shared<ngraph::op::Broadcast>(),
        std::make_shared<ngraph::op::Abs>(),
        std::make_shared<ngraph::op::Minimum>(),
        std::make_shared<ngraph::op::Maximum>(),
        std::make_shared<ngraph::op::Divide>(),
        std::make_shared<ngraph::op::Multiply>(),
        std::make_shared<ngraph::op::Quantize>(),
        std::make_shared<ngraph::op::QuantizedConvolutionBiasAdd>(),
        std::make_shared<ngraph::op::Reshape>(),
        std::make_shared<ngraph::op::Convert>()}},
      {"QuantizedMaxPool",
       {std::make_shared<ngraph::op::Reshape>(),
        std::make_shared<ngraph::op::MaxPool>()}},
      // End of CPU only ops
      {"QuantizeV2",
       {constant, std::make_shared<ngraph::op::Minimum>(),
        std::make_shared<ngraph::op::Abs>(),
        std::make_shared<ngraph::op::Maximum>(),
        std::make_shared<ngraph::op::Quantize>()}},
      {
          "RandomUniform",
          {constant, std::make_shared<ngraph::op::RandomUniform>()},
      },
      {"Rank", {constant}},
      {"RealDiv", {std::make_shared<ngraph::op::Divide>()}},
      {"Reciprocal", {constant, std::make_shared<ngraph::op::Power>()}},
      {"Relu", {std::make_shared<ngraph::op::Relu>()}},
      {"Relu6",
       {constant, std::make_shared<ngraph::op::Minimum>(),
        std::make_shared<ngraph::op::Relu>()}},
      {"ReluGrad", {relu}},
      {"Reshape", {std::make_shared<ngraph::op::Reshape>()}},
      {"Rsqrt", {constant, std::make_shared<ngraph::op::Power>()}},
      {"RsqrtGrad",
       {constant, std::make_shared<ngraph::op::Power>(),
        std::make_shared<ngraph::op::Multiply>()}},
      {"Select",
       {std::make_shared<ngraph::op::Reshape>(),
        std::make_shared<ngraph::op::Broadcast>(),
        std::make_shared<ngraph::op::Select>()}},
      {"Reshape", {constant}},
      {"Shape", {constant}},
      {"Sigmoid",
       {constant, std::make_shared<ngraph::op::Exp>(),
        std::make_shared<ngraph::op::Negative>(),
        std::make_shared<ngraph::op::Add>(),
        std::make_shared<ngraph::op::Divide>()}},
      {"SigmoidGrad",
       {constant, std::make_shared<ngraph::op::Multiply>(),
        std::make_shared<ngraph::op::Subtract>()}},
      {"Sin", {std::make_shared<ngraph::op::Sin>()}},
      {"Size", {constant}},
      {"Sign", {std::make_shared<ngraph::op::Sign>()}},
      {"Slice", {std::make_shared<ngraph::op::Slice>()}},
      {"Snapshot", {}},
      {"Softmax", {std::make_shared<ngraph::op::Softmax>()}},
      {"SoftmaxCrossEntropyWithLogits",
       {std::make_shared<ngraph::op::Broadcast>(),
        std::make_shared<ngraph::op::Max>(),
        std::make_shared<ngraph::op::Subtract>(),
        std::make_shared<ngraph::op::Exp>(),
        std::make_shared<ngraph::op::Sum>(),
        std::make_shared<ngraph::op::Divide>(),
        std::make_shared<ngraph::op::Convert>(),
        std::make_shared<ngraph::op::Multiply>(),
        std::make_shared<ngraph::op::Log>()}},
      {"Softplus",
       {constant, std::make_shared<ngraph::op::Exp>(),
        std::make_shared<ngraph::op::Log>(),
        std::make_shared<ngraph::op::Add>()}},
      {"SpaceToDepth",
       {std::make_shared<ngraph::op::Slice>(),
        std::make_shared<ngraph::op::Concat>()}},
      {"SparseSoftmaxCrossEntropyWithLogits",
       {std::make_shared<ngraph::op::Broadcast>(),
        std::make_shared<ngraph::op::Max>(),
        std::make_shared<ngraph::op::Subtract>(),
        std::make_shared<ngraph::op::Exp>(),
        std::make_shared<ngraph::op::Sum>(),
        std::make_shared<ngraph::op::Divide>(),
        std::make_shared<ngraph::op::OneHot>(),
        std::make_shared<ngraph::op::Convert>(),
        std::make_shared<ngraph::op::Multiply>(),
        std::make_shared<ngraph::op::Log>()}},
      {"Split", {std::make_shared<ngraph::op::Slice>()}},
      {"SplitV", {std::make_shared<ngraph::op::Slice>()}},
      {"Sqrt", {std::make_shared<ngraph::op::Sqrt>()}},
      {"Square", {std::make_shared<ngraph::op::Multiply>()}},
      {"SquaredDifference",
       {std::make_shared<ngraph::op::Subtract>(),
        std::make_shared<ngraph::op::Multiply>()}},
      {"Squeeze", {std::make_shared<ngraph::op::Reshape>()}},
      {"StridedSlice",
       {std::make_shared<ngraph::op::Reverse>(),
        std::make_shared<ngraph::op::Slice>(),
        std::make_shared<ngraph::op::Reshape>()}},
      {"Sub", {std::make_shared<ngraph::op::Subtract>()}},
      {"Sum", {std::make_shared<ngraph::op::Sum>()}},
      {"Tanh", {std::make_shared<ngraph::op::Tanh>()}},
      {"TanhGrad",
       {constant, std::make_shared<ngraph::op::Subtract>(),
        std::make_shared<ngraph::op::Multiply>()}},
      {"Tile", {constant, std::make_shared<ngraph::op::Concat>()}},
      {"TopKV2",
       {std::make_shared<ngraph::op::TopK>(),
        std::make_shared<ngraph::op::GetOutputElement>()}},
      {"Transpose", {constant, std::make_shared<ngraph::op::Reshape>()}},
      {"UnsortedSegmentSum",
       {constant, std::make_shared<ngraph::op::ScatterAdd>()}},
      {"Unpack",
       {std::make_shared<ngraph::op::Slice>(),
        std::make_shared<ngraph::op::Reshape>()}},
      {"ZerosLike", {constant}},
      {"HorovodAllreduce", {std::make_shared<ngraph::op::AllReduce>()}},
      {"HorovodBroadcast",
       {std::make_shared<ngraph::op::BroadcastDistributed>()}},
      {"NoOp", {}},
  };

  return IsSupportedByBackend(node, op_backend, TFtoNgraphOpMap, is_supported);
}

}  // namespace ngraph_bridge
}  // namespace tensorflow
