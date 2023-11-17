#ifndef CENTERPOINT_TENSORRT_WRAPPER_HPP
#define CENTERPOINT_TENSORRT_WRAPPER_HPP

#include "centerpoint_config.hpp"
#include "cuda_utils.hpp"
#include "utils.hpp"
#include "simple_profiler.hpp"


#include <NvInfer.h>

#include <iostream>
#include <memory>
#include <string>
#include <iterator>
#include <fstream>

namespace centerpoint
{

class DummyBatchStream
{
public:
    DummyBatchStream(std::string& dataFile, std::vector<int>& dim)
     //!< We already know the dimensions of MNIST images.
    {
        mDims = nvinfer1::Dims{3, {dim[0], dim[1], dim[2]}};
        std::size_t size = dim[0] * dim[1] * dim[2];
        mData.resize(size);

        random_input(mData);
        // srand(static_cast <unsigned> (time(0)));
        // for(int i = 0; i < size; ++i) {
        //   mData[i] = (rand() / ( RAND_MAX / (10.0) ) );
        // }

    }

    void reset(int firstBatch)
    {
        mBatchCount = firstBatch;
    }

    bool next()
    {
        if (mBatchCount >= mMaxBatches)
        {
            return false;
        }
        ++mBatchCount;
        return true;
    }

    void skip(int skipCount)
    {
        mBatchCount += skipCount;
    }

    float* getBatch()
    {
        return mData.data();
    }

    int getBatchesRead() const
    {
        return mBatchCount;
    }

    int getBatchSize() const
    {
        return mBatchSize;
    }

    nvinfer1::Dims getDims() const
    {
        return nvinfer1::Dims{4, {mBatchSize, mDims.d[0], mDims.d[1], mDims.d[2]}};
    }

private:
    // void readDataFile(const std::string& dataFilePath)
    // {
    //     std::ifstream file{dataFilePath.c_str(), std::ios::binary};

    //     int magicNumber, numImages, imageH, imageW;
    //     file.read(reinterpret_cast<char*>(&magicNumber), sizeof(magicNumber));
    //     // All values in the MNIST files are big endian.
    //     magicNumber = samplesCommon::swapEndianness(magicNumber);
    //     ASSERT(magicNumber == 2051 && "Magic Number does not match the expected value for an MNIST image set");

    //     // Read number of images and dimensions
    //     file.read(reinterpret_cast<char*>(&numImages), sizeof(numImages));
    //     file.read(reinterpret_cast<char*>(&imageH), sizeof(imageH));
    //     file.read(reinterpret_cast<char*>(&imageW), sizeof(imageW));

    //     numImages = samplesCommon::swapEndianness(numImages);
    //     imageH = samplesCommon::swapEndianness(imageH);
    //     imageW = samplesCommon::swapEndianness(imageW);

    //     // The MNIST data is made up of unsigned bytes, so we need to cast to float and normalize.
    //     int numElements = numImages * imageH * imageW;
    //     std::vector<uint8_t> rawData(numElements);
    //     file.read(reinterpret_cast<char*>(rawData.data()), numElements * sizeof(uint8_t));
    //     mData.resize(numElements);
    //     std::transform(
    //         rawData.begin(), rawData.end(), mData.begin(), [](uint8_t val) { return static_cast<float>(val) / 255.f; });
    // }

    // void readLabelsFile(const std::string& labelsFilePath)
    // {
    //     std::ifstream file{labelsFilePath.c_str(), std::ios::binary};
    //     int magicNumber, numImages;
    //     file.read(reinterpret_cast<char*>(&magicNumber), sizeof(magicNumber));
    //     // All values in the MNIST files are big endian.
    //     magicNumber = samplesCommon::swapEndianness(magicNumber);
    //     ASSERT(magicNumber == 2049 && "Magic Number does not match the expected value for an MNIST labels file");

    //     file.read(reinterpret_cast<char*>(&numImages), sizeof(numImages));
    //     numImages = samplesCommon::swapEndianness(numImages);

    //     std::vector<uint8_t> rawLabels(numImages);
    //     file.read(reinterpret_cast<char*>(rawLabels.data()), numImages * sizeof(uint8_t));
    //     mLabels.resize(numImages);
    //     std::transform(
    //         rawLabels.begin(), rawLabels.end(), mLabels.begin(), [](uint8_t val) { return static_cast<float>(val); });
    // }

    int mBatchSize{1};
    int mBatchCount{0}; //!< The batch that will be read on the next invocation of next()
    int mMaxBatches{100};
    nvinfer1::Dims mDims{};
    std::vector<float> mData{};
};

class Int8EntropyCalibrator : public nvinfer1::IInt8MinMaxCalibrator
{
public:
  Int8EntropyCalibrator(
    DummyBatchStream & stream, const std::string calibration_cache_file,
    bool read_cache = true)
  : stream_(stream), calibration_cache_file_(calibration_cache_file), read_cache_(read_cache)
  {
    auto d = stream_.getDims();
    input_count_ = stream_.getBatchSize() * d.d[1] * d.d[2] * d.d[3];

    std::cout << "dim : " << d.d[1] << " , " << d.d[2] << " , " << d.d[3]  << std::endl;

    CHECK_CUDA_ERROR(cudaMalloc(&device_input_, input_count_ * sizeof(float)));
    auto algType = getAlgorithm();
    switch (algType) {
      case (nvinfer1::CalibrationAlgoType::kLEGACY_CALIBRATION):
        std::cout << "CalibrationAlgoType : kLEGACY_CALIBRATION" << std::endl;
        break;
      case (nvinfer1::CalibrationAlgoType::kENTROPY_CALIBRATION):
        std::cout << "CalibrationAlgoType : kENTROPY_CALIBRATION" << std::endl;
        break;
      case (nvinfer1::CalibrationAlgoType::kENTROPY_CALIBRATION_2):
        std::cout << "CalibrationAlgoType : kENTROPY_CALIBRATION_2" << std::endl;
        break;
      case (nvinfer1::CalibrationAlgoType::kMINMAX_CALIBRATION):
        std::cout << "CalibrationAlgoType : kMINMAX_CALIBRATION" << std::endl;
        break;
      default:
        std::cout << "No CalibrationAlgType" << std::endl;
        break;
    }
  }
  int getBatchSize() const noexcept override { return stream_.getBatchSize(); }

  virtual ~Int8EntropyCalibrator() { CHECK_CUDA_ERROR(cudaFree(device_input_)); }

  bool getBatch(void * bindings[], const char * names[], int nb_bindings) noexcept override
  {
    // std::cout << "getBatch" << std::endl;
    (void)names;
    (void)nb_bindings;

    if (!stream_.next()) {
      return false;
    }
    try {
      CHECK_CUDA_ERROR(cudaMemcpy(
        device_input_, stream_.getBatch(), input_count_ * sizeof(float), cudaMemcpyHostToDevice));
    } catch (const std::exception & e) {
      // Do nothing
    }
    bindings[0] = device_input_;
    return true;
  }

  const void * readCalibrationCache(size_t & length) noexcept override
  {
    // std::cout << "readCalibrationCache" << std::endl;
    calib_cache_.clear();
    std::ifstream input(calibration_cache_file_, std::ios::binary);
    input >> std::noskipws;
    if (read_cache_ && input.good()) {
      std::copy(
        std::istream_iterator<char>(input), std::istream_iterator<char>(),
        std::back_inserter(calib_cache_));
    }

    length = calib_cache_.size();
    if (length) {
      std::cout << "Using cached calibration table to build the engine" << std::endl;
    } else {
      std::cout << "New calibration table will be created to build the engine" << std::endl;
    }
    return length ? &calib_cache_[0] : nullptr;
  }

  void writeCalibrationCache(const void * cache, size_t length) noexcept override
  {
    // std::cout << "writeCalibrationCache" << std::endl;
    std::ofstream output(calibration_cache_file_, std::ios::binary);
    output.write(reinterpret_cast<const char *>(cache), length);
  }

private:
  DummyBatchStream stream_;
  const std::string calibration_cache_file_;
  bool read_cache_{true};
  size_t input_count_;
  void * device_input_{nullptr};
  std::vector<char> calib_cache_;
  std::vector<char> hist_cache_;
};

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

  bool setDynamicRange(TrtUniquePtr<nvinfer1::INetworkDefinition>& network);
  void setLayerPrecision(TrtUniquePtr<nvinfer1::INetworkDefinition>& network);

  void printNetworkInfo(const std::string & onnx_file_path, const std::string & precision_);

  TrtUniquePtr<nvinfer1::IRuntime> runtime_{nullptr};
  TrtUniquePtr<nvinfer1::IHostMemory> plan_{nullptr};
  TrtUniquePtr<nvinfer1::ICudaEngine> engine_{nullptr};

  std::unique_ptr<nvinfer1::IInt8Calibrator> calibrator;

  SimpleProfiler model_profiler_{"Model"};
};

}  // namespace centerpoint

#endif  // CENTERPOINT_TENSORRT_WRAPPER_HPP
