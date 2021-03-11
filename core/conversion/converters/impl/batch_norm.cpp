#include "core/conversion/converters/converters.h"
#include "core/util/prelude.h"
#include "torch/torch.h"
#include "plugins/layer_norm_plugin.h"

namespace trtorch {
namespace core {
namespace conversion {
namespace converters {
namespace impl {
namespace {

auto batch_norm_registrations TRTORCH_UNUSED = RegisterNodeConversionPatterns().pattern({
    R"SIG(aten::batch_norm(Tensor input, Tensor? gamma, Tensor? beta,
                            Tensor? mean, Tensor? var,
                            bool training, float momentum, float eps, bool cudnn_enabled) -> (Tensor))SIG",
    [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
      auto input = args[0].ITensor(); // assumes non-static input Tensor
      auto orig_shape = input->getDimensions();
      auto shape = util::toVec(orig_shape);
      auto tensor_type = util::toATenDType(input->getType());
      auto options = torch::TensorOptions().dtype(tensor_type);

      torch::Tensor gamma, beta, mean, var;

      if (ctx->input_is_dynamic) {
        gamma = args[1].unwrapToTensor();
        beta = args[2].unwrapToTensor();
        mean = args[3].unwrapToTensor();
        var = args[4].unwrapToTensor();
      } else {
        gamma = args[1].unwrapToTensor(at::full({shape}, 1, {options}));
        beta = args[2].unwrapToTensor(at::full({shape}, 1, {options}));
        mean = args[3].unwrapToTensor(at::full({shape}, 0, {options}));
        var = args[4].unwrapToTensor(at::full({shape}, 0, {options}));
      }

      auto eps = args[7].unwrapToDouble(1e-5f);

      LOG_DEBUG("momentum disregarded");
      LOG_DEBUG("training disregarded");
      LOG_DEBUG("cudnn disregarded");

      auto should_unpack = util::toVec(orig_shape).size() < 4;
      if (should_unpack) {
        // expand spatial dims from 1D to 2D
        auto new_shape = util::toDimsPad(util::toVec(orig_shape), 4);
        LOG_DEBUG(
            "Input shape is less than 4D got: "
            << orig_shape << ", inserting shuffle layer to reshape to 4D tensor shape: " << new_shape);
        auto in_shuffle = ctx->net->addShuffle(*input);
        in_shuffle->setReshapeDimensions(new_shape);
        in_shuffle->setName(std::string("[Reshape input to " + util::toStr(new_shape) + ']').c_str());
        input = in_shuffle->getOutput(0);
      }

      auto scale = gamma / torch::sqrt(var + eps);
      auto bias = beta - mean * scale;

      auto scale_weights = Weights(ctx, scale);
      auto bias_weights = Weights(ctx, bias);

      auto power = Weights(ctx, at::ones_like(scale));
      auto bn = ctx->net->addScaleNd(
          *input, nvinfer1::ScaleMode::kCHANNEL, bias_weights.data, scale_weights.data, power.data, 1);
      bn->setName(util::node_info(n).c_str());
      auto out_tensor = bn->getOutput(0);

      if (should_unpack) {
        LOG_DEBUG("Inserting shuffle layer to reshape to back to original shape: " << orig_shape);
        auto out_shuffle = ctx->net->addShuffle(*out_tensor);
        out_shuffle->setReshapeDimensions(orig_shape);
        out_shuffle->setName(std::string("[Reshape output to " + util::toStr(orig_shape) + ']').c_str());
        out_tensor = out_shuffle->getOutput(0);
      }

      ctx->AssociateValueAndTensor(n->outputs()[0], out_tensor);
      return true;
    }}).pattern({R"SIG(aten::layer_norm(Tensor input, int[] normalized_shape,
                Tensor? weight=None, Tensor? bias=None, 
                float eps=1.0e-05, bool cudnn_enable=True) -> (Tensor))SIG",
    [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
      auto input = args[0].ITensor(); // assumes non-static input Tensor
      auto tensor_type = util::toATenDType(input->getType());
      auto options = torch::TensorOptions().dtype(tensor_type);
      auto normalized_shape = args[1].unwrapToIntList().vec();
      auto weight = args[2].isITensor() ? args[2].ITensor() : 
                            tensor_to_const(ctx, at::full({normalized_shape}, 1, {options}));
      auto bias = args[3].isITensor() ? args[3].ITensor() : 
                             tensor_to_const(ctx, at::full({normalized_shape}, 0, {options})); 
      auto eps = args[4].unwrapToDouble(1e-5f);
      auto orig_shape = input->getDimensions();
      auto shape = util::toVec(orig_shape);

      // auto should_unpack = util::toVec(orig_shape).size() < 4;
    
      // if (should_unpack) {
      //   // expand spatial dims from 1D to 2D
      //   auto new_shape = util::toDimsPad(util::toVec(orig_shape), 4);
      //   LOG_DEBUG(
      //       "Input shape is less than 4D got: "
      //       << orig_shape << ", inserting shuffle layer to reshape to 4D tensor shape: " << new_shape);
      //   auto in_shuffle = ctx->net->addShuffle(*input);
      //   in_shuffle->setReshapeDimensions(new_shape);
      //   in_shuffle->setName(std::string("[Reshape input to " + util::toStr(new_shape) + ']').c_str());
      //   input = in_shuffle->getOutput(0);
      // }
      auto creator = new plugins::LayerNormPluginCreator();
      auto plugin = creator->createPlugin(
          "LayerNorm",
          normalized_shape,
          eps);
      nvinfer1::ITensor* inputs[] = {input, weight, bias}; 

      auto layer_norm =
          ctx->net->addPluginV2(inputs, 3, *plugin);
      TRTORCH_CHECK(layer_norm, "Unable to create layer_norm plugin from node" << *n);

      layer_norm->setName(util::node_info(n).c_str());
      ctx->AssociateValueAndTensor(n->outputs()[0], layer_norm->getOutput(0));
      LOG_DEBUG("Output tensor shape: " << layer_norm->getOutput(0)->getDimensions());
      return true;
    }});

} // namespace
} // namespace impl
} // namespace converters
} // namespace conversion
} // namespace core
} // namespace trtorch
