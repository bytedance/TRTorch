#ifndef TRTORCH_LAYER_NORM_PLUGIN_H
#define TRTORCH_LAYER_NORM_PLUGIN_H

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAEvent.h>
#include <cuda_runtime_api.h>
#include <cudnn.h>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "NvInfer.h"
#include "core/util/prelude.h"
#include "torch/torch.h"

using namespace nvinfer1;

namespace trtorch {
namespace core {
namespace conversion {
namespace converters {
namespace impl {
namespace plugins {

class LayerNormPlugin : public nvinfer1::IPluginV2DynamicExt {
 private:
  at::TensorOptions tensor_options_;
  DataType dtype_;

  std::vector<int64_t> normalized_shape_;
  float eps_;

 protected:
  // To prevent compiler warnings
  using nvinfer1::IPluginV2DynamicExt::canBroadcastInputAcrossBatch;
  using nvinfer1::IPluginV2DynamicExt::configurePlugin;
  using nvinfer1::IPluginV2DynamicExt::enqueue;
  using nvinfer1::IPluginV2DynamicExt::getOutputDimensions;
  using nvinfer1::IPluginV2DynamicExt::getWorkspaceSize;
  using nvinfer1::IPluginV2DynamicExt::isOutputBroadcastAcrossBatch;
  using nvinfer1::IPluginV2DynamicExt::supportsFormat;

 public:
  LayerNormPlugin(std::vector<int64_t> normalized_shape, float eps);

  LayerNormPlugin(const char* data, size_t length);

  LayerNormPlugin() = delete;

  //   std::vector<int64_t> getInputShape();

  //   std::vector<int64_t> getOutputShape();

  //   std::vector<int64_t> getOutputSize();

  int getNbOutputs() const override;

  const char* getPluginType() const override;

  const char* getPluginVersion() const override;

  const char* getPluginNamespace() const override;

  void setPluginNamespace(const char* pluginNamespace) override{};

  nvinfer1::IPluginV2DynamicExt* clone() const override;

  nvinfer1::DimsExprs getOutputDimensions(
      int outputIndex,
      const nvinfer1::DimsExprs* inputs,
      int nbInputs,
      nvinfer1::IExprBuilder& exprBuilder) override;

  nvinfer1::DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const override;

  int initialize() override;

  void terminate() override {}

  void serialize(void* buffer) const;

  std::string serializeToString() const;

  size_t getSerializationSize() const override;

  void destroy() override {}

  bool supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs)
      override;

  void configurePlugin(
      const nvinfer1::DynamicPluginTensorDesc* in,
      int nbInputs,
      const nvinfer1::DynamicPluginTensorDesc* out,
      int nbOutputs) override;

  size_t getWorkspaceSize(
      const nvinfer1::PluginTensorDesc* inputs,
      int nbInputs,
      const nvinfer1::PluginTensorDesc* outputs,
      int nbOutputs) const override;

  int enqueue(
      const nvinfer1::PluginTensorDesc* inputDesc,
      const nvinfer1::PluginTensorDesc* outputDesc,
      const void* const* inputs,
      void* const* outputs,
      void* workspace,
      cudaStream_t stream) override;
};

class LayerNormPluginCreator : public nvinfer1::IPluginCreator {
 private:
  std::string name_;

 public:
  LayerNormPluginCreator() = default;

  const char* getPluginNamespace() const override;

  void setPluginNamespace(const char* libNamespace) override{};

  const char* getPluginName() const override;

  const char* getPluginVersion() const override;

  nvinfer1::IPluginV2* createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc) override;

  LayerNormPlugin* createPlugin(
      const char* name,
      std::vector<int64_t> normalized_shape,
      float eps);

  nvinfer1::IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) override;

  const nvinfer1::PluginFieldCollection* getFieldNames() override;
};

} // namespace plugins
} // namespace impl
} // namespace converters
} // namespace conversion
} // namespace core
} // namespace trtorch

#endif // TRTORCH_LAYER_NORM_PLUGIN_H