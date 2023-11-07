#ifndef CENTERPOINT_TENSORRT_WRAPPER_HPP
#define CENTERPOINT_TENSORRT_WRAPPER_HPP

#include "centerpoint_config.hpp"

#include <NvInfer.h>

#include <iostream>
#include <memory>
#include <string>

namespace centerpoint
{

class Logger : public nvinfer1::ILogger {
  public:
    void log(Severity severity, const char* msg) noexcept override {
        // suppress info-level message
        //if (severity == Severity::kERROR || severity == Severity::kINTERNAL_ERROR || severity == Severity::kINFO ) {
        if (severity == Severity::kERROR || severity == Severity::kINTERNAL_ERROR) {
            std::cerr << "trt_infer: " << msg << std::endl;
        }
    }
};

template <typename T>
struct InferDeleter
{
  void operator()(T * obj) const
  {
    if (obj) {
#if TENSORRT_VERSION_MAJOR >= 8
      delete obj;
#else
      obj->destroy();
#endif
    }
  }
};

template <typename T>
using TrtUniquePtr = std::unique_ptr<T, InferDeleter<T>>;

class TensorRTWrapper
{
public:
  explicit TensorRTWrapper(const CenterPointConfig & config);

  ~TensorRTWrapper();

  bool init(
    const std::string & onnx_path, const std::string & engine_path, const std::string & precision);

  TrtUniquePtr<nvinfer1::IExecutionContext> context_{nullptr};

protected:
  virtual bool setProfile(
    nvinfer1::IBuilder & builder, nvinfer1::INetworkDefinition & network,
    nvinfer1::IBuilderConfig & config) = 0;

  CenterPointConfig config_;
  Logger logger_;

private:
  bool parseONNX(
    const std::string & onnx_path, const std::string & engine_path, const std::string & precision,
    size_t workspace_size = (1ULL << 30));

  bool saveEngine(const std::string & engine_path);

  bool loadEngine(const std::string & engine_path);

  bool createContext();

  TrtUniquePtr<nvinfer1::IRuntime> runtime_{nullptr};
  TrtUniquePtr<nvinfer1::IHostMemory> plan_{nullptr};
  TrtUniquePtr<nvinfer1::ICudaEngine> engine_{nullptr};
};

}  // namespace centerpoint

#endif  // CENTERPOINT_TENSORRT_WRAPPER_HPP
