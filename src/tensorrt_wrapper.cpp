// Copyright 2021 TIER IV, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "tensorrt_wrapper.hpp"

#include <NvOnnxParser.h>

#include <iostream>
#include <sstream>
#include <fstream>
#include <memory>
#include <string>
#include <limits>
#include <unordered_map>

namespace centerpoint
{
TensorRTWrapper::TensorRTWrapper(const CenterPointConfig & config) : config_(config)
{
}

TensorRTWrapper::~TensorRTWrapper()
{
  context_.reset();
  runtime_.reset();
  plan_.reset();
  engine_.reset();
}

bool TensorRTWrapper::init(
  const std::string & onnx_path, const std::string & engine_path, const std::string & precision)
{
  runtime_ =
    TrtUniquePtr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger_));
  if (!runtime_) {
    std::cout << "Failed to create runtime" << std::endl;
    return false;
  }

  // DLA
  // runtime_->setDLACore(0);

  std::cout << "Net : " << onnx_path << std::endl;
  // printNetworkInfo(onnx_path, precision);

  bool success;
  std::ifstream engine_file(engine_path);
  if (engine_file.is_open()) {
    success = loadEngine(engine_path);
  } else {
    success = parseONNX(onnx_path, engine_path, precision);
  }
  success &= createContext();

  return success;
}

bool TensorRTWrapper::createContext()
{
  if (!engine_) {
    std::cout << "Failed to create context: Engine was not created" << std::endl;
    return false;
  }

  context_ =
    TrtUniquePtr<nvinfer1::IExecutionContext>(engine_->createExecutionContext());
  if (!context_) {
    std::cout << "Failed to create context" << std::endl;
    return false;
  }
  // context_->setProfiler(&model_profiler_);

  return true;
}

bool TensorRTWrapper::parseONNX(
  const std::string & onnx_path, const std::string & engine_path, const std::string & precision,
  const size_t workspace_size)
{
  auto builder =
    TrtUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger_));
  if (!builder) {
    std::cout << "Failed to create builder" << std::endl;
    return false;
  }

  auto config =
    TrtUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
  if (!config) {
    std::cout << "Failed to create config" << std::endl;
    return false;
  }

  // DLA
  // config->setDefaultDeviceType(nvinfer1::DeviceType::kDLA);
  // config->setDLACore(0);

#if (NV_TENSORRT_MAJOR * 1000) + (NV_TENSORRT_MINOR * 100) + NV_TENSOR_PATCH >= 8400
  config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, workspace_size);
#else
  config->setMaxWorkspaceSize(workspace_size);
#endif

  config->setFlag(nvinfer1::BuilderFlag::kGPU_FALLBACK);

  config->setFlag(nvinfer1::BuilderFlag::kPREFER_PRECISION_CONSTRAINTS);

  if (precision == "fp16") {
    if (builder->platformHasFastFp16()) {
      std::cout <<  "Using TensorRT FP16 Inference" << std::endl;
      config->setFlag(nvinfer1::BuilderFlag::kFP16);
      // config->setFlag(nvinfer1::BuilderFlag::kOBEY_PRECISION_CONSTRAINTS);
    } else {
      std::cout << "TensorRT FP16 Inference isn't supported in this environment" << std::endl;
    }
  }

  // builder->setMaxBatchSize(1);

  if (precision == "int8") {
    if (builder->platformHasFastInt8()) {
      std::cout <<  "Using TensorRT INT8 Inference" << std::endl;
      // config->setFlag(nvinfer1::BuilderFlag::kPREFER_PRECISION_CONSTRAINTS);
      config->setFlag(nvinfer1::BuilderFlag::kINT8);
      // config->setFlag(nvinfer1::BuilderFlag::kOBEY_PRECISION_CONSTRAINTS);
      // builder->setInt8Mode(true);

      // not being used
      std::string s("?");
      std::string cache_file;
      DummyBatchStream dbs; 
      float max = 100.0, min = -100.0;
      std::vector<int> v{};

      if(config_.mode_ == 0) {
        v = std::vector<int>{40000, 9, 32};
        cache_file = "/home/adehome/quantization_lib/model/cal_encoder.txt";
        min = -80.0, max = 80.0;
      }
      else {
        v = std::vector<int>{32, 560, 560};
        cache_file = "/home/adehome/quantization_lib/model/cal_head.txt";
        min = 0.0, max = 5.0;
      }

      // min and max have to be measured somehow
      dbs.setDim(v);
      dbs.setMinMax(min, max);

      calibrator.reset(new Int8EntropyCalibrator(dbs, cache_file, true));
      config->setInt8Calibrator(calibrator.get());
      // setLayerPrecision(network);
    } else {
      std::cout << "TensorRT INT8 Inference isn't supported in this environment" << std::endl;
    }
  }

  const auto flag =
    1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
  auto network =
    TrtUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(flag));
  // if (precision == "int8") {
  //   setLayerPrecision(network);
  //   bool res = setDynamicRange(network);
  //   if(res) {
  //     std::cout <<  "Set to int8" << std::endl;
  //   }
  // }
  if (!network) {
    std::cout <<  "Failed to create network" << std::endl;
    return false;
  }

  auto parser = TrtUniquePtr<nvonnxparser::IParser>(
    nvonnxparser::createParser(*network, logger_));
  parser->parseFromFile(onnx_path.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kERROR));

  if (!setProfile(*builder, *network, *config)) {
    std::cout <<  "Failed to set profile" << std::endl;
    return false;
  }

  std::cout << 
    "Applying optimizations and building TRT CUDA engine (" << onnx_path << ") ..." << std::endl;
  plan_ = TrtUniquePtr<nvinfer1::IHostMemory>(
    builder->buildSerializedNetwork(*network, *config));
  if (!plan_) {
    std::cout <<  "Failed to create serialized network" << std::endl;
    return false;
  }

  std::cout <<  "deserializeCudaEngine" << std::endl;
  engine_ = TrtUniquePtr<nvinfer1::ICudaEngine>(
    runtime_->deserializeCudaEngine(plan_->data(), plan_->size()));
  if (!engine_) {
    std::cout <<  "Failed to create engine" << std::endl;
    return false;
  }

  return saveEngine(engine_path);
}

bool TensorRTWrapper::saveEngine(const std::string & engine_path)
{
  std::cout <<  "Writing to " << engine_path << std::endl;
  std::ofstream file(engine_path, std::ios::out | std::ios::binary);
  file.write(reinterpret_cast<const char *>(plan_->data()), plan_->size());
  return true;
}

bool TensorRTWrapper::loadEngine(const std::string & engine_path)
{
  std::ifstream engine_file(engine_path);
  std::stringstream engine_buffer;
  engine_buffer << engine_file.rdbuf();
  std::string engine_str = engine_buffer.str();
  engine_ = TrtUniquePtr<nvinfer1::ICudaEngine>(runtime_->deserializeCudaEngine(
    reinterpret_cast<const void *>(engine_str.data()), engine_str.size()));
  std::cout <<  "Loaded engine from " << engine_path << std::endl;
  return true;
}

bool TensorRTWrapper::setDynamicRange(TrtUniquePtr<nvinfer1::INetworkDefinition>& network)
{
  float max_range = 50.0;
  // set dynamic range for network input tensors
  for (int i = 0; i < network->getNbInputs(); ++i)
  {
      std::string tName = network->getInput(i)->getName();
      if (!network->getInput(i)->setDynamicRange(-max_range, max_range))
      {
          return false;
      }
  }

  // set dynamic range for layer output tensors
  for (int i = 0; i < network->getNbLayers(); ++i)
  {
      auto lyr = network->getLayer(i);
      for (int j = 0, e = lyr->getNbOutputs(); j < e; ++j)
      {
          std::string tName = lyr->getOutput(j)->getName();
          if (lyr->getType() == nvinfer1::LayerType::kCONSTANT)
          {
              nvinfer1::IConstantLayer* cLyr = static_cast<nvinfer1::IConstantLayer*>(lyr);
              auto wts = cLyr->getWeights();
              double max = std::numeric_limits<double>::min();
              for (int64_t wb = 0, we = wts.count; wb < we; ++wb)
              {
                  double val{};
                  switch (wts.type)
                  {
                    case nvinfer1::DataType::kFLOAT: val = static_cast<const float*>(wts.values)[wb]; break;
                    case nvinfer1::DataType::kBOOL: val = static_cast<const bool*>(wts.values)[wb]; break;
                    case nvinfer1::DataType::kINT8: val = static_cast<const int8_t*>(wts.values)[wb]; break;
                    // case nvinfer1::DataType::kHALF: val = static_cast<const half_float::half*>(wts.values)[wb]; break;
                    case nvinfer1::DataType::kHALF: val = 1.0; break;
                    case nvinfer1::DataType::kINT32: val = static_cast<const int32_t*>(wts.values)[wb]; break;
                  }
                  max = std::max(max, std::abs(val));
              }

              if (!lyr->getOutput(j)->setDynamicRange(-max, max))
              {
                  return false;
              }
          }
          else
          {
              // Calibrator generated dynamic range for network tensor can be overriden or set using below API
              if (!lyr->getOutput(j)->setDynamicRange(-max_range, max_range))
              {
                  return false;
              }
          }
      }
  }

  return true;
}

void TensorRTWrapper::setLayerPrecision(TrtUniquePtr<nvinfer1::INetworkDefinition>& network)
{
    for (int i = 0; i < network->getNbLayers(); ++i)
    {
        auto layer = network->getLayer(i);

        // Don't set the precision on non-computation layers as they don't support
        // int8.
        if (layer->getType() != nvinfer1::LayerType::kCONSTANT && layer->getType() != nvinfer1::LayerType::kCONCATENATION
            && layer->getType() != nvinfer1::LayerType::kSHAPE && layer->getType() != nvinfer1::LayerType::kSHUFFLE
            && layer->getType() != nvinfer1::LayerType::kGATHER)
        {
            // set computation precision of the layer
            layer->setPrecision(nvinfer1::DataType::kINT8);
        }

        for (int j = 0; j < layer->getNbOutputs(); ++j)
        {
            std::string tensorName = layer->getOutput(j)->getName();
            // set output type of execution tensors and not shape tensors.
            if (layer->getOutput(j)->isExecutionTensor())
            {
                layer->setOutputType(j, nvinfer1::DataType::kINT8);
            }
        }
    }
}

void TensorRTWrapper::printNetworkInfo(const std::string & onnx_file_path, const std::string & precision_)
{
  std::size_t max_workspace_size_ = (1ULL << 30);
  std::unordered_map<nvinfer1::DataType, std::string> int2type = {{nvinfer1::DataType::kFLOAT, "kFLOAT"}, {nvinfer1::DataType::kHALF, "kHALF"}, {nvinfer1::DataType::kINT8, "kINT8"}, {nvinfer1::DataType::kINT32, "kINT32"}};
  auto builder = TrtUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger_));
  if (!builder) {
    logger_.log(nvinfer1::ILogger::Severity::kERROR, "Fail to create builder");
    return;
  }

  const auto explicitBatch =
    1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);

  auto network =
    TrtUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
  if (!network) {
    logger_.log(nvinfer1::ILogger::Severity::kERROR, "Fail to create network");
    return;
  }

  auto config = TrtUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
  if (!config) {
    logger_.log(nvinfer1::ILogger::Severity::kERROR, "Fail to create builder config");
    return;
  }
  
  std::cout << "Using " << precision_ << std::endl;
  if (precision_ == "fp16") {
    config->setFlag(nvinfer1::BuilderFlag::kFP16);
  }
  else if (precision_ == "int8") {
    config->setFlag(nvinfer1::BuilderFlag::kINT8);
  }
  config->setFlag(nvinfer1::BuilderFlag::kPREFER_PRECISION_CONSTRAINTS);
#if (NV_TENSORRT_MAJOR * 1000) + (NV_TENSORRT_MINOR * 100) + NV_TENSOR_PATCH >= 8400
  config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, max_workspace_size_);
#else
  config->setMaxWorkspaceSize(max_workspace_size_);
#endif

  auto parser = TrtUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, logger_));
  if (!parser->parseFromFile(
        onnx_file_path.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kERROR))) {
    return;
  }
  // setLayerPrecision(network);
  int num = network->getNbLayers();
  float total_gflops = 0.0;
  int total_params = 0;
  for (int i = 0; i < num; i++) {
    nvinfer1::ILayer * layer = network->getLayer(i);
    auto layer_type = layer->getType();
    std::string name = layer->getName();
    // model_profiler_.setProfDict(layer);

    if (layer_type == nvinfer1::LayerType::kCONSTANT) {
      std::cout << "C" << i << " [Constant]" << std::endl;
      std::cout << " " << int2type[layer->getPrecision()] << std::endl;
      continue;
    }
    nvinfer1::ITensor * in = layer->getInput(0);
    nvinfer1::Dims dim_in = in->getDimensions();
    nvinfer1::ITensor * out = layer->getOutput(0);
    nvinfer1::Dims dim_out = out->getDimensions();

    if (layer_type == nvinfer1::LayerType::kCONVOLUTION) {
      nvinfer1::IConvolutionLayer * conv = (nvinfer1::IConvolutionLayer *)layer;
      nvinfer1::Dims k_dims = conv->getKernelSizeNd();
      nvinfer1::Dims s_dims = conv->getStrideNd();
      int groups = conv->getNbGroups();
      int stride = s_dims.d[0];
      int num_weights = (dim_in.d[1] / groups) * dim_out.d[1] * k_dims.d[0] * k_dims.d[1];
      float gflops = (2 * num_weights) * (dim_in.d[3] / stride * dim_in.d[2] / stride / 1e9);
      ;
      total_gflops += gflops;
      total_params += num_weights;
      std::cout << "L" << i << " [conv " << k_dims.d[0] << "x" << k_dims.d[1] << " (" << groups
                << ") "
                << "/" << s_dims.d[0] << "] " << dim_in.d[3] << "x" << dim_in.d[2] << "x"
                << dim_in.d[1] << " -> " << dim_out.d[3] << "x" << dim_out.d[2] << "x"
                << dim_out.d[1];
      std::cout << " weights:" << num_weights;
      std::cout << " GFLOPs:" << gflops;
      std::cout << " " << int2type[layer->getPrecision()] << std::endl;
      std::cout << std::endl;
    } else if (layer_type == nvinfer1::LayerType::kPOOLING) {
      nvinfer1::IPoolingLayer * pool = (nvinfer1::IPoolingLayer *)layer;
      auto p_type = pool->getPoolingType();
      nvinfer1::Dims dim_stride = pool->getStrideNd();
      nvinfer1::Dims dim_window = pool->getWindowSizeNd();

      std::cout << "L" << i << " [";
      if (p_type == nvinfer1::PoolingType::kMAX) {
        std::cout << "max ";
      } else if (p_type == nvinfer1::PoolingType::kAVERAGE) {
        std::cout << "avg ";
      } else if (p_type == nvinfer1::PoolingType::kMAX_AVERAGE_BLEND) {
        std::cout << "max avg blend ";
      }
      float gflops = dim_in.d[1] * dim_window.d[0] / dim_stride.d[0] * dim_window.d[1] /
                     dim_stride.d[1] * dim_in.d[2] * dim_in.d[3] / 1e9;
      total_gflops += gflops;
      std::cout << "pool " << dim_window.d[0] << "x" << dim_window.d[1] << "]";
      std::cout << " GFLOPs:" << gflops;
      std::cout << " " << int2type[layer->getPrecision()] << std::endl;
      std::cout << std::endl;
    } else if (layer_type == nvinfer1::LayerType::kRESIZE) {
      std::cout << "L" << i << " [resize]" << std::endl;
      std::cout << " " << int2type[layer->getPrecision()] << std::endl;
    } else if (layer_type == nvinfer1::LayerType::kELEMENTWISE) {
      std::cout << "E" << i << " [Elementwise]" << std::endl;
      std::cout << " " << int2type[layer->getPrecision()] << std::endl;
    } else if (layer_type == nvinfer1::LayerType::kSHUFFLE) {
      std::cout << "S" << i << " [Shuffle]" << std::endl;
      std::cout << " " << int2type[layer->getPrecision()] << std::endl;
    } else if (layer_type == nvinfer1::LayerType::kSLICE) {
      std::cout << "S" << i << " [Slice]" << std::endl;
      std::cout << " " << int2type[layer->getPrecision()] << std::endl;
    }
  }
  std::cout << "Total " << total_gflops << " GFLOPs" << std::endl;
  std::cout << "Total " << total_params / 1000.0 / 1000.0 << " M params" << std::endl;
  return;
}

}  // namespace centerpoint
