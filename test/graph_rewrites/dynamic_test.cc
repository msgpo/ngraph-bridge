/*******************************************************************************
 * Copyright 2019-2020 Intel Corporation
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

#include "gtest/gtest.h"

#include "tensorflow/core/graph/node_builder.h"

#include "ngraph_bridge/ngraph_api.h"
#include "ngraph_bridge/ngraph_mark_for_clustering.h"
#include "test/test_utilities.h"

// using namespace std;
namespace ng = ngraph;
using namespace ngraph;

namespace tensorflow {

namespace ngraph_bridge {

namespace testing {

// Set using C API, get using C API
TEST(DisableOps, GettingSettingTest1) {
  ASSERT_EQ(config::ngraph_is_dynamic(), false);
  config::ngraph_use_dynamic();
  ASSERT_EQ(config::ngraph_is_dynamic(), true);
  config::ngraph_use_static();
  ASSERT_EQ(config::ngraph_is_dynamic(), false);
}

// Set using Cpp API, get using Cpp API
TEST(DisableOps, GettingSettingTest2) {
  ASSERT_EQ(config::IsDynamic(), false);
  config::UseDynamic();
  ASSERT_EQ(config::IsDynamic(), true);
  config::UseStatic();
  ASSERT_EQ(config::IsDynamic(), false);
}

// TODO remove me
TEST(NGTest, range_subgraph) {
  // Create a graph for f(start,stop,step) = Range(start,stop,step).
  auto start = make_shared<op::Parameter>(element::i32, Shape{});
  auto stop = make_shared<op::Parameter>(element::i32, Shape{});
  auto step = make_shared<op::Parameter>(element::i32, Shape{});
  auto start_2 = make_shared<op::Parameter>(element::i32, Shape{});
  PartialShape out_max_shape{15};
  PartialShape out_max_shape_2{10};

  // subgraph
  auto range = make_shared<op::Range>(start, stop, step);
  auto negative = make_shared<op::Negative>(range);
  auto abs = make_shared<op::Abs>(negative);
  auto sum = make_shared<op::Sum>(abs, AxisSet{0});
  auto range_2 = make_shared<op::Range>(start_2, sum, step);
  auto negative_2 = make_shared<op::Negative>(range_2);

  // range->output(0).set_max_partial_shape(out_max_shape);
  // range_2->output(0).set_max_partial_shape(out_max_shape_2);
  auto f = make_shared<Function>(NodeVector{negative_2},
                                 ParameterVector{start, stop, step, start_2});

  auto backend = runtime::Backend::create("CPU", true);

  ASSERT_EQ(backend->executable_can_create_tensors(), false);

  auto backend1 = runtime::Backend::create("CPU", false);
  ASSERT_EQ(backend1->executable_can_create_tensors(), true);

  // even if backend's executable can create tensors, dyn_wrapped backend can't

  auto handle = backend->compile(f);
  /*pass::Manager passes;
  passes.register_pass<pass::MinMaxShapePropagation>();
  passes.run_passes(f);*/
  /*

      auto t_start = backend->create_tensor(element::i32, Shape{});
      //copy_data(t_start, vector<int32_t>{0});
      auto t_stop = backend->create_tensor(element::i32, Shape{});
      //copy_data(t_stop, vector<int32_t>{10});
      auto t_step = backend->create_tensor(element::i32, Shape{});
      //copy_data(t_step, vector<int32_t>{1});
      auto t_start_2 = backend->create_tensor(element::i32, Shape{});
      //copy_data(t_start_2, vector<int32_t>{40});
      auto result = backend->create_dynamic_tensor(element::i32,
     PartialShape::dynamic());



      vector<int32_t> expected_result{-40, -41, -42, -43, -44};

      //EXPECT_EQ(out_max_shape, range->get_output_shape(0));
      //EXPECT_EQ(out_max_shape, range->output(0).get_max_partial_shape());
      //EXPECT_EQ(out_max_shape, negative->get_output_shape(0));

      //EXPECT_EQ(out_max_shape, negative->output(0).get_max_partial_shape());
      //EXPECT_EQ(out_max_shape, abs->output(0).get_max_partial_shape());
      //EXPECT_EQ(out_max_shape, sum->output(0).get_max_partial_shape());
      //EXPECT_EQ(out_max_shape_2, range_2->output(0).get_max_partial_shape());
      //EXPECT_EQ(out_max_shape_2,
     negative_2->output(0).get_max_partial_shape());
      handle->call_with_validate({result}, {t_start, t_stop, t_step,
     t_start_2});
      //EXPECT_EQ(PartialShape{5}, result->get_shape());
      //ASSERT_EQ(expected_result, read_vector<int32_t>(result));
      */
}
}
}
}