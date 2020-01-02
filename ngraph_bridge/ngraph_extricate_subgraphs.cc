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

#include "ngraph_bridge/ngraph_extricate_subgraphs.h"

using namespace std;

namespace tensorflow {

namespace ngraph_bridge {

Status ExtricateSubgraph(Graph* graph, SubgraphExtractionResults& results) {
  std::map<int, std::string>& device_name_map = results.device_name_map;
  std::map<int, std::string>& backend_name_map = results.backend_name_map;

  for (auto node : graph->op_nodes()) {
    int cluster_idx;

    if (GetNodeCluster(node, &cluster_idx) != Status::OK()) {
      continue;
    }

    string node_backend;
    if (GetNodeBackend(node, &node_backend) != Status::OK()) {
      continue;
    }

    auto it = device_name_map.find(cluster_idx);

    if (it != device_name_map.end()) {
      if (it->second != node->assigned_device_name()) {
        std::stringstream ss_err;
        ss_err << "Node " << node->name() << " in cluster " << cluster_idx
               << " has assigned device " << node->assigned_device_name()
               << " but another node with assigned device " << it->second
               << " has already been seen in the same cluster";

        return errors::Internal(ss_err.str());
      }
    } else {
      NGRAPH_VLOG(3) << "setting cluster " << cluster_idx
                     << " requested device to '" << node->assigned_device_name()
                     << "'";
      device_name_map[cluster_idx] = node->assigned_device_name();
    }

    auto itr = backend_name_map.find(cluster_idx);

    if (itr != backend_name_map.end()) {
      if (itr->second != node_backend) {
        std::stringstream ss_err;
        ss_err << "Node " << node->name() << " in cluster " << cluster_idx
               << " has assigned backend " << node_backend
               << " but another node with assigned backend " << it->second
               << " has already been seen in the same cluster";

        return errors::Internal(ss_err.str());
      }
    } else {
      NGRAPH_VLOG(3) << "setting cluster " << cluster_idx
                     << " requested backend to '" << node_backend << "'";
      backend_name_map[cluster_idx] = node_backend;
    }
  }

  return Status::OK();
}

}  // namespace ngraph_bridge
}  // namespace tensorflow
