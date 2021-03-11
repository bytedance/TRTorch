#include <time.h>
#include <iostream>
#include <string>
#include "torch/script.h"
#include "trtorch/trtorch.h"
// #include "nvToolsExt.h"

#include <typeinfo>
#include "core/compiler.h"
#include "gtest/gtest.h"
#include "tests/util/util.h"
#include "torch/csrc/jit/ir/irparser.h"

TEST(Converters, ModelTestCorrectly) {
  // Done 1 input
  //   std::string path =
  //       "/home/vincentzh/Projects/ByteDanceModel/model/vc_dolly_grou_heyi/dolly.feature.blaid_heyi_22/dolly.feature.blaid.zip";
  //   torch::Tensor in0 = torch::randn({64, 1, 64, 64}, torch::kCUDA).to(torch::kFloat);
  //   torch::Tensor in1 = torch::randn({64, 1, 64, 64}, torch::kCUDA).to(torch::kFloat);

  // Done 1 input
    std::string path =
    "/home/ryan/work/data/TRTModel/dolly.feature.moco_603277_luoyuchu_19/moco_64_quan.zip";
    std::string path_trt =
    "/home/ryan/work/data/TRTModel/dolly.feature.moco_603277_luoyuchu_19/moco_64_quan_trt.zip";
    torch::Tensor in0 = torch::randn({1, 1, 64, 64}, torch::kCUDA).to(torch::kFloat);
    // torch::Tensor in1 = torch::randn({1, 1, 64, 64}, torch::kCUDA).to(torch::kFloat);

  // Done 1 input
  //   std::string path =
  //   "/home/vincentzh/Projects/ByteDanceModel/model/vc_dolly_grou_heyi/labvc.dungeon.moco_simclr_text_cls_shengtao_9/labvc.dungeon.moco_simclr_text_cls.pt";
  //   torch::Tensor in0 = torch::randn({64, 1, 3, 224, 224}, torch::kCUDA).to(torch::kFloat);
  //   torch::Tensor in1 = torch::randn({1, 300, 128}, torch::kCUDA).to(torch::kFloat);

  // Done 2 input
  //   std::string path =
  //   "/home/vincentzh/Projects/ByteDanceModel/model/vc_dolly_grou_heyi/labvc.dungeon.searchmatcher_liuyongchao.eric_7/labvc.dungeon.searchmatcher.pt";
  //   torch::Tensor in0 = torch::randn({1, 10, 128}, torch::kCUDA).to(torch::kFloat);
  //   torch::Tensor in1 = torch::randn({1, 10, 128}, torch::kCUDA).to(torch::kFloat);

  // Done 1 input
  //   std::string path =
  //   "/home/vincentzh/Projects/ByteDanceModel/model/traced/yet_another_torch/torchnative_fp32_3x224x224/model_512.ts";
  //   torch::Tensor in0 = torch::randn({1, 3, 224, 224}, torch::kCUDA).to(torch::kFloat);
  //   torch::Tensor in1 = torch::randn({1, 3, 224, 224}, torch::kCUDA).to(torch::kFloat);

  // Done 2 input
  //   std::string path =
  //   "/home/vincentzh/Projects/ByteDanceModel/model/vc_dolly_grou_heyi/labvc.dungeon.deepmatcher_zhudefa_29/labvc.dungeon.deepmatcher.pt";
  //   torch::Tensor in0 = torch::randn({1, 300, 128}, torch::kCUDA).to(torch::kFloat);
  //   torch::Tensor in1 = torch::randn({1, 300, 128}, torch::kCUDA).to(torch::kFloat);

  // Done 1 input
  // std::string path =
  //     "/home/vincentzh/Projects/ByteDanceModel/model/vc_dolly_grou_heyi/labvc.dungeon.moco_simclr_text_shengtao_4/labvc.dungeon.moco_simclr_text.pt";
  // torch::Tensor in0 = torch::randn({64, 1, 3, 224, 224}, torch::kCUDA).to(torch::kFloat);
  // torch::Tensor in1 = torch::randn({1, 300, 128}, torch::kCUDA).to(torch::kFloat);

  // 1 input
  // std::string path =
  // "/home/vincentzh/Projects/ByteDanceModel/model/traced/resnest/torchnative_fp32_3x224x224/model.ts"; torch::Tensor
  // in0 = torch::randn({1, 3, 224, 224}, torch::kCUDA).to(torch::kFloat); torch::Tensor in1 = torch::randn({1, 300,
  // 128}, torch::kCUDA).to(torch::kFloat);

  // 1 input done
  // std::string path = "/home/vincentzh/Projects/ByteDanceModel/model/vc_dolly_grou_heyi/dolly.feature.bac_sunwanxuan_8/bac_model_deploy_v1.pt";
  // torch::Tensor in0 = torch::randn({1, 48, 224, 224}, torch::kCUDA).to(torch::kFloat); 
  // torch::Tensor in1 = torch::randn({1, 300,  128}, torch::kCUDA).to(torch::kFloat);

  // 9 input
  // std::string path =
  // "/home/ryan/work/data/TRTModel/end2end_models_douyin_first_review_v3_offline_zhangxuan.x_33/douyin_first_review_model_fp32.pt.zip";
  // torch::Tensor in0 = torch::randn({8, 3, 224, 224}, torch::kCUDA).to(torch::kFloat);
  // torch::Tensor in1 = torch::randn({8, 3, 224, 224}, torch::kCUDA).to(torch::kFloat);
  // torch::Tensor in2 = torch::randn({8, 64}, torch::kCUDA).to(torch::kFloat);
  // torch::Tensor in3 = torch::randn({8, 64}, torch::kCUDA).to(torch::kFloat);
  // torch::Tensor in4 = torch::randn({8, 64}, torch::kCUDA).to(torch::kFloat);
  // torch::Tensor in5 = torch::randn({8, 128}, torch::kCUDA).to(torch::kFloat);
  // torch::Tensor in6 = torch::randn({8, 128}, torch::kCUDA).to(torch::kFloat);
  // torch::Tensor in7 = torch::randn({8, 128}, torch::kCUDA).to(torch::kFloat);
  // torch::Tensor in8 = torch::randn({8, 52}, torch::kCUDA).to(torch::kFloat);

  // 9 input
  // std::string path =
  // "/home/vincentzh/Projects/ByteDanceModel/model/0202_new_high_dau/end2end_models_douyin_first_review_v3_wangke.88_7/douyin_first_review_model_fp32.pt.zip";
  // std::string path = "/home/ryan/work/data/TRTModel/end2end_models_douyin_first_review_v3_wangke.88_7/douyin_first_review_model_fp32.pt.zip"
  // std::string path = "/home/ryan/work/data/TRTModel/end2end_models_douyin_report_v3_user_verify_service_24/douyin_report_model_fp32.pt.zip"
  
  // torch::Tensor in0 = torch::randn({128, 203}, torch::kCUDA).to(torch::kLong);
  // torch::Tensor in1 = torch::randn({128, 203}, torch::kCUDA).to(torch::kLong);
  // torch::Tensor in2 = torch::randn({8, 64}, torch::kCUDA).to(torch::kLong);
  // torch::Tensor in3 = torch::randn({8, 64}, torch::kCUDA).to(torch::kLong);
  // torch::Tensor in4 = torch::randn({8, 64}, torch::kCUDA).to(torch::kLong);
  // torch::Tensor in5 = torch::randn({8, 128}, torch::kCUDA).to(torch::kLong);
  // torch::Tensor in6 = torch::randn({8, 128}, torch::kCUDA).to(torch::kLong);
  // torch::Tensor in7 = torch::randn({8, 128}, torch::kCUDA).to(torch::kLong);
  // torch::Tensor in8 = torch::randn({8, 61}, torch::kCUDA).to(torch::kLong);

  // 7 input
  // std::string path =
  //     "/home/vincentzh/Projects/ByteDanceModel/model/0202_new_high_dau/end2end_models_douyin_report_v3_user_verify_service_24/douyin_report_model_fp32.pt.zip";
  // torch::Tensor in0 = torch::randn({128, 3, 224, 224}, torch::kCUDA).to(torch::kFloat);
  // torch::Tensor in1 = torch::randn({128, 64}, torch::kCUDA).to(torch::kFloat);
  // torch::Tensor in2 = torch::randn({128, 64}, torch::kCUDA).to(torch::kFloat);
  // torch::Tensor in3 = torch::randn({128, 64}, torch::kCUDA).to(torch::kFloat);
  // torch::Tensor in4 = torch::randn({128, 128}, torch::kCUDA).to(torch::kFloat);
  // torch::Tensor in5 = torch::randn({128, 128}, torch::kCUDA).to(torch::kFloat);
  // torch::Tensor in6 = torch::randn({128, 128}, torch::kCUDA).to(torch::kFloat);

  // std::string path =
  // "/home/vincentzh/Projects/ByteDanceModel/model/vc_dolly_grou_heyi/labvc.dungeon.searchnet_liuyongchao.eric_1/labvc.dungeon.searchnet.pt";
  // torch::Tensor in0 = torch::randn({64, 1, 3, 224, 224}, torch::kCUDA).to(torch::kFloat);
  // torch::Tensor in1 = torch::randn({1, 300, 128}, torch::kCUDA).to(torch::kFloat);

  // std::string path =
  // "/home/vincentzh/Projects/ByteDanceModel/model/vc_dolly_grou_heyi/labvc.dungeon.regnet_zhudefa_7/labvc.dungeon.moco_simclr.pt";
  // torch::Tensor in0 = torch::randn({64, 1, 3, 224, 224}, torch::kCUDA).to(torch::kFloat);
  // torch::Tensor in1 = torch::randn({1, 300, 128}, torch::kCUDA).to(torch::kFloat);

  //   std::string path = "/home/vincent/Projects/ByteDanceModel/model/traced/image_sp/test.zip"

  // std::string path = "/home/ryan/work/data/TRTModel/rerank-roberta-6_luoyuchu_29/classic_v1.zip";
  // torch::Tensor in0 = torch::randint(0, 4, {128, 203}, torch::kCUDA).to(torch::kInt64);
  // torch::Tensor in1 = torch::randint(0, 4, {128, 203}, torch::kCUDA).to(torch::kInt64);
  // torch::Tensor in2 = torch::randint(0, 4, {128, 203}, torch::kCUDA).to(torch::kInt32);
  // torch::Tensor in3 = torch::randint(0, 4, {128, 203}, torch::kCUDA).to(torch::kInt32);

  std::vector<at::Tensor> inputs;
  // std::vector<at::Tensor> inputs_trt;
  inputs.push_back(in0);
  // inputs.push_back(in1);
  // inputs_trt.push_back(in0);
  // inputs_trt.push_back(in1);
  // inputs.push_back(in2);
  // inputs.push_back(in3);
  // inputs.push_back(in4);
  // inputs.push_back(in5);
  // inputs.push_back(in6);
  // inputs.push_back(in7);
  // inputs.push_back(in8);

  torch::jit::Module mod;
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    mod = torch::jit::load(path);
  } catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
  }
  mod.eval();
  mod.to(torch::kCUDA);

  std::vector<torch::jit::IValue> inputs_;
  // std::vector<torch::jit::IValue> inputs_trt_;
  for (auto in : inputs) {
    inputs_.push_back(torch::jit::IValue(in.clone()));
  }
  // for (auto in : inputs_trt) {
  //   inputs_trt_.push_back(torch::jit::IValue(in.clone()));
  // }

  // nvtxRangePushA("ComplieGraph");
  auto trt_mod = trtorch::CompileGraph(mod, std::vector<trtorch::CompileSpec::InputRange>{in0.sizes()});
  // trt_mod.save(path_trt);
  // nvtxRangePop();
  // std::cout << "trtorch::CompileGraph" << std::endl;
  // auto trt_mod = trtorch::CompileGraph(mod, std::vector<trtorch::CompileSpec::InputRange>{in0.sizes(),
  //   in1.sizes(), in2.sizes(), in3.sizes(), in4.sizes(), in5.sizes(), in6.sizes(), in7.sizes(), in8.sizes()});

  // auto trt_mod = trtorch::CompileGraph(
  //     mod,
  //     std::vector<trtorch::CompileSpec::InputRange>{
  //         in0.sizes(), in1.sizes(), in2.sizes(), in3.sizes(), in4.sizes(), in5.sizes(), in6.sizes()});

  // clock_t start_time, end_time;
  // start_time = clock();
  // for (int i = 0; i < 10; i++) {
  //   mod.forward(inputs_);
  // }
  // end_time = clock();
  // std::cout << "Original torchscript elapse time(100 times) " << (double)(end_time - start_time) / CLOCKS_PER_SEC
  //           << std::endl;
  // auto before = (double)(end_time - start_time) / CLOCKS_PER_SEC;
  
  // start_time = clock();
  // for (int i = 0; i < 10; i++) {
  //   mod.forward(inputs_trt_);
  // }
  // end_time = clock();
  // std::cout << "TRTorch elapse time(100 times) " << (double)(end_time - start_time) / CLOCKS_PER_SEC << std::endl;
  // auto after = (double)(end_time - start_time) / CLOCKS_PER_SEC;
  // std::cout << "TRTorch speed up " << (double)before / after << std::endl;


  // nvtxRangePushA("mod.forward");
  auto out = mod.forward(inputs_);
  // nvtxRangePop();
  // std::cout << "mod forward" << std::endl;
  auto trt_out = mod.forward(inputs_);
  // std::cout << "trt_mod forward" << std::endl;

  // const std::vector<std::vector<int64_t>> input_shapes = {{1, 1, 64, 64}};
  // std::vector<torch::jit::IValue> jit1_inputs_ivalues;
  // std::vector<torch::jit::IValue> trt1_inputs_ivalues;
  // for (auto in_shape : input_shapes) {
  //   auto in = at::randint(5, in_shape, {at::kCUDA});
  //   jit1_inputs_ivalues.push_back(in.clone());
  //   trt1_inputs_ivalues.push_back(in.clone());
  // }

  //   std::vector<torch::jit::IValue> jit2_inputs_ivalues;
  //   std::vector<torch::jit::IValue> trt2_inputs_ivalues;
  //   for (auto in_shape : input_shapes) {
  //     auto in = at::randint(5, in_shape, {at::kCUDA});
  //     jit2_inputs_ivalues.push_back(in.clone());
  //     trt2_inputs_ivalues.push_back(in.clone());
  //   }

  // auto out = trtorch::tests::util::RunModuleForward(mod, jit1_inputs_ivalues);
  // auto trt_mod = trtorch::CompileGraph(mod, input_shapes);
  // auto trt_out = trtorch::tests::util::RunModuleForward(trt_mod, trt1_inputs_ivalues);

  ASSERT_TRUE(trtorch::tests::util::almostEqual(out.toTensor(), trt_out.toTensor(), 1e-2));
  // ASSERT_TRUE(trtorch::tests::util::almostEqual(out.toTuple()->elements()[0].toTensor(), trt_out.toTuple()->elements()[0].toTensor(), 1e-2));

  // ASSERT_TRUE(trtorch::tests::util::almostEqual(
  //     out.toTuple()->elements()[0].toTensor(), trt_out.toTuple()->elements()[0].toTensor(), 1e-2));
  // ASSERT_TRUE(trtorch::tests::util::almostEqual(
  //     out.toTuple()->elements()[1].toTensor(), trt_out.toTuple()->elements()[1].toTensor(), 1e-2));
}