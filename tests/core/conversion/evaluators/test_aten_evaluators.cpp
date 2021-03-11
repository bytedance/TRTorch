#include <string>
#include "core/compiler.h"
#include "gtest/gtest.h"
#include "tests/util/util.h"
#include "torch/csrc/jit/ir/irparser.h"

TEST(Evaluators, IntEvaluatesCorrectly) {
  const auto graph = R"IR(
      graph(%0 : Tensor):
        %lu.1 : int = aten::Int(%0)
        return (%lu.1))IR";

  const std::vector<std::vector<int64_t>> input_shapes = {{1}};
  std::vector<torch::jit::IValue> jit_inputs_ivalues1, jit_inputs_ivalues2;
  std::vector<torch::jit::IValue> trt_inputs_ivalues1, trt_inputs_ivalues2;
  for (auto in_shape : input_shapes) {
    auto in1 = at::rand(in_shape, {at::kCUDA});
    jit_inputs_ivalues1.push_back(in1.clone());
    trt_inputs_ivalues1.push_back(in1.clone());
    auto in2 = at::randint(-10, 10, in_shape, {at::kCUDA});
    jit_inputs_ivalues2.push_back(in2.clone());
    trt_inputs_ivalues2.push_back(in2.clone());
  }

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, &*g);

  auto jit_results1 = trtorch::tests::util::EvaluateGraphJIT(g, jit_inputs_ivalues1);
  auto trt_results1 = trtorch::tests::util::EvaluateGraph(g->block(), trt_inputs_ivalues1);
  ASSERT_TRUE(jit_results1[0] == trt_results1[0]);

  auto jit_results2 = trtorch::tests::util::EvaluateGraphJIT(g, jit_inputs_ivalues2);
  auto trt_results2 = trtorch::tests::util::EvaluateGraph(g->block(), trt_inputs_ivalues2);
  ASSERT_TRUE(jit_results2[0] == trt_results2[0]);
}

TEST(Evaluators, IsFloatingPointEvaluatesCorrectly) {
  const auto graph = R"IR(
      graph(%0 : Tensor):
        %lu.1 : bool = aten::is_floating_point(%0)
        return (%lu.1))IR";

  const std::vector<std::vector<int64_t>> input_shapes = {{10, 10}};
  std::vector<torch::jit::IValue> jit_inputs_ivalues;
  std::vector<torch::jit::IValue> trt_inputs_ivalues;
  for (auto in_shape : input_shapes) {
    auto in = at::rand(in_shape, {at::kCUDA});
    jit_inputs_ivalues.push_back(in.clone());
    trt_inputs_ivalues.push_back(in.clone());
  }

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, &*g);
  auto jit_results = trtorch::tests::util::EvaluateGraphJIT(g, jit_inputs_ivalues);
  auto trt_results = trtorch::tests::util::EvaluateGraph(g->block(), trt_inputs_ivalues);
  ASSERT_TRUE(jit_results[0] == trt_results[0]);
}

TEST(Evaluators, DivIntEvaluatesCorrectly) {
  const auto graph = R"IR(
      graph():
        %1 : int = prim::Constant[value=9]()
        %2 : int = prim::Constant[value=4]()
        %3 : float = aten::div(%1, %2)
        return (%3))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, &*g);
  auto jit_results = trtorch::tests::util::EvaluateGraphJIT(g, {});
  auto trt_results = trtorch::tests::util::EvaluateGraph(g->block(), {});

  ASSERT_TRUE(jit_results[0] == trt_results[0]);
}

TEST(Evaluators, DivFloatEvaluatesCorrectly) {
  const auto graph = R"IR(
      graph():
        %1 : float = prim::Constant[value=9.1]()
        %2 : float = prim::Constant[value=4.2]()
        %3 : float = aten::div(%1, %2)
        return (%3))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, &*g);

  auto jit_results = trtorch::tests::util::EvaluateGraphJIT(g, {});
  auto trt_results = trtorch::tests::util::EvaluateGraph(g->block(), {});

  ASSERT_TRUE(jit_results[0] == trt_results[0]);
}

TEST(Evaluators, ATenArangeIntEvaluatesCorrectly) {
  const auto graph = R"IR(
      graph():
        %0 : int = prim::Constant[value=51]()
        %1 : None = prim::Constant()
        %2 : Tensor = aten::arange(%0, %1, %1, %1, %1)
        return (%2))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, &*g);

  auto jit_results = trtorch::tests::util::EvaluateGraphJIT(g, {});
  auto trt_results = trtorch::tests::util::EvaluateGraph(g->block(), {});

  ASSERT_TRUE(trtorch::tests::util::almostEqual(jit_results[0].toTensor(), trt_results[0].toTensor(), 2e-6));
}

TEST(Evaluators, ATenArangeFloatEvaluatesCorrectly) {
  const auto graph = R"IR(
      graph():
        %0 : float = prim::Constant[value=51.2]()
        %1 : None = prim::Constant()
        %2 : Tensor = aten::arange(%0, %1, %1, %1, %1)
        return (%2))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, &*g);

  auto jit_results = trtorch::tests::util::EvaluateGraphJIT(g, {});
  auto trt_results = trtorch::tests::util::EvaluateGraph(g->block(), {});
  ASSERT_TRUE(trtorch::tests::util::almostEqual(jit_results[0].toTensor(), trt_results[0].toTensor(), 2e-6));
}
