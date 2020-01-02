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

#pragma once

#ifndef NGRAPH_TF_EXTRICATE_SUBGRAPH_H_
#define NGRAPH_TF_EXTRICATE_SUBGRAPH_H_

#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/lib/core/errors.h"

#include "ngraph/ngraph.hpp"

#include "logging/ngraph_log.h"
#include "ngraph_bridge/ngraph_assign_clusters.h"
#include "ngraph_bridge/ngraph_mark_for_clustering.h"

namespace tensorflow {

namespace ngraph_bridge {

typedef struct SubgraphExtractionResults {
  std::set<tensorflow::GraphDef*> subgraphs_for_encapsulation;

  // A map from cluster indices to the expected device name for nodes
  // in that cluster.
  std::map<int, std::string> device_name_map;

  // We *should* eventually have a way of monitoring the device and the backend
  // together
  std::map<int, std::string> backend_name_map;

} SubgraphExtractionResults;

Status ExtricateSubgraph(Graph*, SubgraphExtractionResults&);

}  // namespace ngraph_bridge
}  // namespace tensorflow

#endif  // NGRAPH_TF_EXTRICATE_SUBGRAPH_H_
