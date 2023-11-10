#include <sstream>
#include <fstream>
#include <iostream>
#include <fstream>
#include <memory>
#include <string>
#include <vector>
#include <chrono>
#include <numeric>

#include "centerpoint.hpp"
#include "cuda_utils.hpp"

// void GetDeviceInfo()
// {
//     cudaDeviceProp prop;

//     int count = 0;
//     cudaGetDeviceCount(&count);
//     printf("\nGPU has cuda devices: %d\n", count);
//     for (int i = 0; i < count; ++i) {
//         cudaGetDeviceProperties(&prop, i);
//         printf("----device id: %d info----\n", i);
//         printf("  GPU : %s \n", prop.name);
//         printf("  Capbility: %d.%d\n", prop.major, prop.minor);
//         printf("  Global memory: %luMB\n", prop.totalGlobalMem >> 20);
//         printf("  Const memory: %luKB\n", prop.totalConstMem  >> 10);
//         printf("  SM in a block: %luKB\n", prop.sharedMemPerBlock >> 10);
//         printf("  warp size: %d\n", prop.warpSize);
//         printf("  threads in a block: %d\n", prop.maxThreadsPerBlock);
//         printf("  block dim: (%d,%d,%d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
//         printf("  grid dim: (%d,%d,%d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
//     }
//     printf("\n");
// }

bool hasEnding(std::string const &fullString, std::string const &ending)
{
    if (fullString.length() >= ending.length()) {
        return (0 == fullString.compare (fullString.length() - ending.length(), ending.length(), ending));
    } else {
        return false;
    }
}

int loadData(const char *file, void **data, unsigned int *length)
{
    std::fstream dataFile(file, std::ifstream::in);

    if (!dataFile.is_open()) {
        std::cout << "Can't open files: "<< file<<std::endl;
        return -1;
    }

    unsigned int len = 0;
    dataFile.seekg (0, dataFile.end);
    len = dataFile.tellg();
    dataFile.seekg (0, dataFile.beg);

    char *buffer = new char[len];
    if (buffer==NULL) {
        std::cout << "Can't malloc buffer."<<std::endl;
        dataFile.close();
        exit(EXIT_FAILURE);
    }

    dataFile.read(buffer, len);
    dataFile.close();

    *data = (void*)buffer;
    *length = len;
    return 0;  
}

template<typename T>
double getAverage(std::vector<T> const& v) {
    if (v.empty()) {
        return 0;
    }
    return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
}

// void perf_report(){
//     float a = getAverage(timing_pre_);
//     float b = getAverage(timing_scn_engine_);
//     float c = getAverage(timing_trt_);
//     float d = getAverage(timing_post_);
//     float total = a + b + c + d;
//     std::cout << "\nPerf Report: "        << std::endl;
//     std::cout << "    Voxelization: "   << a << " ms." <<std::endl;
//     std::cout << "    3D Backbone: "    << b << " ms." << std::endl;
//     std::cout << "    RPN + Head: "     << c << " ms." << std::endl;
//     std::cout << "    Decode + NMS: "   << d << " ms." << std::endl;
//     std::cout << "    Total: "          << total << " ms." << std::endl;
// }

// void SaveBoxPred(std::vector<Bndbox> boxes, std::string file_name)
// {
//     std::ofstream ofs;
//     ofs.open(file_name, std::ios::out);
//     ofs.setf(std::ios::fixed, std::ios::floatfield);
//     ofs.precision(5);
//     if (ofs.is_open()) {
//         for (const auto box : boxes) {
//           ofs << box.x << " ";
//           ofs << box.y << " ";
//           ofs << box.z << " ";
//           ofs << box.w << " ";
//           ofs << box.l << " ";
//           ofs << box.h << " ";
//           ofs << box.vx << " ";
//           ofs << box.vy << " ";
//           ofs << box.rt << " ";
//           ofs << box.id << " ";
//           ofs << box.score << " ";
//           ofs << "\n";
//         }
//     }
//     else {
//       std::cerr << "Output file cannot be opened!" << std::endl;
//     }
//     ofs.close();
//     std::cout << "Saved prediction in: " << file_name << std::endl;
//     return;
// }

int main() {
    EventTimer timer_;
    centerpoint::CenterPointConfig config(3, 4, 40000, {-89.6, -89.6, -3.0, 89.6, 89.6, 5.0}, 
        {0.32, 0.32, 8.0}, 1, 9, 0.35, 0.5, {0.3, 0.0, 0.3}, 0);
    centerpoint::CenterPointConfig config_1(3, 4, 40000, {-89.6, -89.6, -3.0, 89.6, 89.6, 5.0}, 
        {0.32, 0.32, 8.0}, 1, 9, 0.35, 0.5, {0.3, 0.0, 0.3}, 1);
    std::string precision = "int8";
    std::string data_file = "../data/2.bin";
    std::string encoder_onnx = "/home/development/quantization_lib/model/pts_voxel_encoder_centerpoint.onnx";
    std::string encoder_engine = "/home/development/quantization_lib/model/pts_voxel_encoder_centerpoint.engine";
    std::string head_onnx = "/home/development/quantization_lib/model/pts_backbone_neck_head_centerpoint.onnx";
    std::string head_engine = "/home/development/quantization_lib/model/pts_backbone_neck_head_centerpoint.engine";

    std::vector<float> timing_pre_voxel_;
    std::vector<float> timing_pre_;
    std::vector<float> timing_voxel_trt_;
    std::vector<float> timing_seletc_;
    std::vector<float> timing_head_trt_;
    std::vector<float> timing_post_;

    unsigned int length = 0;
    void *data = NULL;
    std::shared_ptr<char> buffer((char *)data, std::default_delete<char[]>());
    loadData(data_file.data(), &data, &length);
    buffer.reset((char *)data);

    float* points = (float*)buffer.get();
    size_t points_size = length/sizeof(float)/4;

    std::cout << "point len : " << length << std::endl;

    // float *points_data = nullptr;
    // unsigned int points_data_size = points_size * 4 * sizeof(float);

    std::vector<float> points_vec(points_size);
    for(auto i = 0; i < points_size; ++i) {
        points_vec[i] = points[i];
        // std::cout << i <<", " <<points_vec[i] << std::endl;
    }

    const auto voxels_size =
        config.max_voxel_size_ * config.max_point_in_voxel_size_ * config.point_feature_size_;
    const auto coordinates_size = config.max_voxel_size_ * config.point_dim_size_;
    const auto encoder_in_feature_size_ =
        config.max_voxel_size_ * config.max_point_in_voxel_size_ * config.encoder_in_feature_size_;
    const auto pillar_features_size = config.max_voxel_size_ * config.encoder_out_feature_size_;
    const auto spatial_features_size_ =
        config.grid_size_x_ * config.grid_size_y_ * config.encoder_out_feature_size_;
    const auto grid_xy_size = config.down_grid_size_x_ * config.down_grid_size_y_;

    std::vector<float> voxels_(voxels_size);
    std::vector<int> coordinates_(coordinates_size);
    std::vector<float> num_points_per_voxel_(config.max_voxel_size_);
    std::unique_ptr<centerpoint::VoxelGenerator> vg_ptr_(new centerpoint::VoxelGenerator(config));

    cudaStream_t stream_{nullptr};
    cudaStreamCreate(&stream_);

    cuda::unique_ptr<float[]> voxels_d_ = cuda::make_unique<float[]>(voxels_size);
    cuda::unique_ptr<int[]> coordinates_d_ = cuda::make_unique<int[]>(coordinates_size);
    cuda::unique_ptr<float[]> num_points_per_voxel_d_ = cuda::make_unique<float[]>(config.max_voxel_size_);
    cuda::unique_ptr<float[]> encoder_in_features_d_ = cuda::make_unique<float[]>(encoder_in_feature_size_);
    cuda::unique_ptr<float[]> pillar_features_d_ = cuda::make_unique<float[]>(pillar_features_size);
    cuda::unique_ptr<float[]> spatial_features_d_ = cuda::make_unique<float[]>(spatial_features_size_);
    cuda::unique_ptr<float[]> head_out_heatmap_d_ = cuda::make_unique<float[]>(grid_xy_size * config.class_size_);
    cuda::unique_ptr<float[]> head_out_offset_d_ = cuda::make_unique<float[]>(grid_xy_size * config.head_out_offset_size_);
    cuda::unique_ptr<float[]> head_out_z_d_ = cuda::make_unique<float[]>(grid_xy_size * config.head_out_z_size_);
    cuda::unique_ptr<float[]> head_out_dim_d_ = cuda::make_unique<float[]>(grid_xy_size * config.head_out_dim_size_);
    cuda::unique_ptr<float[]> head_out_rot_d_ = cuda::make_unique<float[]>(grid_xy_size * config.head_out_rot_size_);
    cuda::unique_ptr<float[]> head_out_vel_d_ = cuda::make_unique<float[]>(grid_xy_size * config.head_out_vel_size_);

    std::unique_ptr<centerpoint::VoxelEncoderTRT> encoder_trt_ptr_ = std::make_unique<centerpoint::VoxelEncoderTRT>(config);
    encoder_trt_ptr_->init(encoder_onnx, encoder_engine, precision);
    encoder_trt_ptr_->context_->setBindingDimensions(
        0,
        nvinfer1::Dims3(
        config.max_voxel_size_, config.max_point_in_voxel_size_, config.encoder_in_feature_size_));

    std::vector<std::size_t> out_channel_sizes = {
        config.class_size_,        config.head_out_offset_size_, config.head_out_z_size_,
        config.head_out_dim_size_, config.head_out_rot_size_,    config.head_out_vel_size_};
    std::unique_ptr<centerpoint::HeadTRT> head_trt_ptr_ = std::make_unique<centerpoint::HeadTRT>(config_1);
    head_trt_ptr_->init(head_onnx, head_engine, precision);
    head_trt_ptr_->context_->setBindingDimensions(
        0, nvinfer1::Dims4(
            config.batch_size_, config.encoder_out_feature_size_, config.grid_size_y_,
            config.grid_size_x_));
    std::unique_ptr<centerpoint::PostProcessCUDA> post_proc_ptr_ = std::make_unique<centerpoint::PostProcessCUDA>(config);

    auto start = std::chrono::system_clock::now();
    int num_voxels_ = vg_ptr_->pointsToVoxels(points_vec, voxels_, coordinates_, num_points_per_voxel_);
    auto end = std::chrono::system_clock::now();
    timing_pre_voxel_.push_back(std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());

    std::cout << "voxel : " << num_voxels_ << std::endl;

    // memcpy from host to device (not copy empty voxels)
    CHECK_CUDA_ERROR(cudaMemcpyAsync(
    voxels_d_.get(), voxels_.data(), voxels_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(
    coordinates_d_.get(), coordinates_.data(), coordinates_size * sizeof(int),
    cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(
    num_points_per_voxel_d_.get(), num_points_per_voxel_.data(), num_voxels_ * sizeof(float),
    cudaMemcpyHostToDevice));

    CHECK_CUDA_ERROR(cudaMemsetAsync(
    encoder_in_features_d_.get(), 0, encoder_in_feature_size_ * sizeof(float), stream_));

    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream_));

    timer_.start(stream_);
    CHECK_CUDA_ERROR(centerpoint::generateFeatures_launch(
        voxels_d_.get(), num_points_per_voxel_d_.get(), coordinates_d_.get(), num_voxels_,
        config.max_voxel_size_, config.voxel_size_x_, config.voxel_size_y_, config.voxel_size_z_,
        config.range_min_x_, config.range_min_y_, config.range_min_z_, encoder_in_features_d_.get(),
        stream_));
    timing_pre_.push_back(timer_.stop("GenerateFeatures_kernel", true));

    std::vector<void *> encoder_buffers{encoder_in_features_d_.get(), pillar_features_d_.get()};
    timer_.start(stream_);
    encoder_trt_ptr_->context_->enqueueV2(encoder_buffers.data(), stream_, nullptr);
    timing_voxel_trt_.push_back(timer_.stop("Encoder_trt", true));

    timer_.start(stream_);
    CHECK_CUDA_ERROR(centerpoint::scatterFeatures_launch(
        pillar_features_d_.get(), coordinates_d_.get(), num_voxels_, config.max_voxel_size_,
        config.encoder_out_feature_size_, config.grid_size_x_, config.grid_size_y_,
        spatial_features_d_.get(), stream_));
    timing_seletc_.push_back(timer_.stop("Select_kernel", true));

    CHECK_CUDA_ERROR(
    cudaMemsetAsync(spatial_features_d_.get(), 0, spatial_features_size_ * sizeof(float), stream_));

    std::vector<void *> head_buffers = {spatial_features_d_.get(), head_out_heatmap_d_.get(),
                                        head_out_offset_d_.get(),  head_out_z_d_.get(),
                                        head_out_dim_d_.get(),     head_out_rot_d_.get(),
                                        head_out_vel_d_.get()};
    timer_.start(stream_);
    head_trt_ptr_->context_->enqueueV2(head_buffers.data(), stream_, nullptr);
    timing_head_trt_.push_back(timer_.stop("Head_trt", true));
    // cuda::unique_ptr<float[]> head_out_heatmap_d_ = cuda::make_unique<float[]>(grid_xy_size * config_.class_size_);
    // cuda::unique_ptr<float[]> head_out_offset_d_ = cuda::make_unique<float[]>(grid_xy_size * config_.head_out_offset_size_);
    // cuda::unique_ptr<float[]> head_out_z_d_ = cuda::make_unique<float[]>(grid_xy_size * config_.head_out_z_size_);
    // cuda::unique_ptr<float[]> head_out_dim_d_ = cuda::make_unique<float[]>(grid_xy_size * config_.head_out_dim_size_);
    // cuda::unique_ptr<float[]> head_out_rot_d_ = cuda::make_unique<float[]>(grid_xy_size * config_.head_out_rot_size_);
    // cuda::unique_ptr<float[]> head_out_vel_d_ = cuda::make_unique<float[]>(grid_xy_size * config_.head_out_vel_size_);

    std::vector<centerpoint::Box3D> det_boxes3d;
    timer_.start(stream_);
    CHECK_CUDA_ERROR(post_proc_ptr_->generateDetectedBoxes3D_launch(
        head_out_heatmap_d_.get(), head_out_offset_d_.get(), head_out_z_d_.get(), head_out_dim_d_.get(),
        head_out_rot_d_.get(), head_out_vel_d_.get(), det_boxes3d, stream_));
    timing_post_.push_back(timer_.stop("Post_kernel", true));
    std::cout << "detect size : " << det_boxes3d.size() << std::endl;


    float a = getAverage(timing_pre_voxel_);
    float b = getAverage(timing_pre_);
    float c = getAverage(timing_voxel_trt_);
    float d = getAverage(timing_seletc_);
    float e = getAverage(timing_head_trt_);
    float f = getAverage(timing_post_);
    float total = a + b + c + d + e + f;
    std::cout << "\nPerf Report: "        << std::endl;
    std::cout << "    Pre_Voxelization: "   << a << " ms." <<std::endl;
    std::cout << "    GenerateFeatures: "    << b << " ms." << std::endl;
    std::cout << "    Encoder_trt: "     << c << " ms." << std::endl;
    std::cout << "    Select: "   << d << " ms." << std::endl;
    std::cout << "    Head_trt: "   << e << " ms." << std::endl;
    std::cout << "    Post: "   << f << " ms." << std::endl;
    std::cout << "    Total: "          << total << " ms." << std::endl;


    if (stream_) {
        cudaStreamSynchronize(stream_);
        cudaStreamDestroy(stream_);
    }
    return 0;
}