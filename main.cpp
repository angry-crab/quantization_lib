#include <sstream>
#include <fstream>
#include <iostream>
#include <fstream>
#include <memory>
#include <string>
#include <vector>
#include <chrono>
#include <numeric>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

#include "centerpoint.hpp"
#include "cuda_utils.hpp"
#include "geometry_utils.hpp"

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

// int loadData(const char *file, void **data, unsigned int *length)
// {
//     std::fstream dataFile(file, std::ifstream::in);

//     if (!dataFile.is_open()) {
//         std::cout << "Can't open files: "<< file<<std::endl;
//         return -1;
//     }

//     unsigned int len = 0;
//     dataFile.seekg (0, dataFile.end);
//     len = dataFile.tellg();
//     dataFile.seekg (0, dataFile.beg);

//     char *buffer = new char[len];
//     if (buffer==NULL) {
//         std::cout << "Can't malloc buffer."<<std::endl;
//         dataFile.close();
//         exit(EXIT_FAILURE);
//     }

//     dataFile.read(buffer, len);
//     dataFile.close();

//     *data = (void*)buffer;
//     *length = len;
//     return 0;  
// }

void read_cloud(std::string input, std::vector<float>& buffer) {
    // Ptr cloud(new PointCloud);
    std::fstream file(input.c_str(), std::ios::in | std::ios::binary);
    if(!file.good()){
        std::cerr << "Could not read file: " << input << std::endl;
        // return cloud;
    }
    file.seekg(0, std::ios::beg);
    for (int i=0; file.good() && !file.eof(); i++) {
        float x;
        // PointType point;
        // file.read((char *) &point.x, 3*sizeof(float));
        // file.read((char *) &point.intensity, sizeof(float));
        // cloud->push_back(point);

        file.read((char *) &x, sizeof(float));
        buffer.push_back(x);
    }
    file.close();
    std::cout << "Read " + input + " Done !" << std::endl;
    // return cloud;
}

void write_cloud(std::string output, std::vector<float>& data) {
    pcl::PointCloud<pcl::PointXYZ> cloud;
    
    for(int i = 0; i < data.size(); i += 5) {
        pcl::PointXYZ point(data[i], data[i+1], data[i+2]);
        cloud.push_back(point);
    }

    pcl::io::savePCDFileASCII (output, cloud);
}

void dumpDetectionsAsMesh(
  const std::vector<centerpoint::Box3D> & objects,
  const std::string output_path)
{
  std::ofstream ofs(output_path, std::ofstream::out);
  std::stringstream vertices_stream;
  std::stringstream faces_stream;
  int index = 0;
  int num_detections = static_cast<int>(objects.size());

  ofs << "ply" << std::endl;
  ofs << "format ascii 1.0" << std::endl;
  ofs << "comment created by lidar_centerpoint" << std::endl;
  ofs << "element vertex " << 8 * num_detections << std::endl;
  ofs << "property float x" << std::endl;
  ofs << "property float y" << std::endl;
  ofs << "property float z" << std::endl;
  ofs << "element face " << 11 * num_detections << std::endl;
  ofs << "property list uchar uint vertex_indices" << std::endl;
  ofs << "end_header" << std::endl;

  auto streamFace = [&faces_stream](int v1, int v2, int v3) {
    faces_stream << std::to_string(3) << " " << std::to_string(v1) << " " << std::to_string(v2)
                 << " " << std::to_string(v3) << std::endl;
  };

  for (const auto & object : objects) {
    Eigen::Affine3d pose_affine;
    Box3DtoAffine(object, pose_affine);

    std::vector<Eigen::Vector3d> vertices = getVertices(object, pose_affine);

    for (const auto & vertex : vertices) {
      vertices_stream << vertex.x() << " " << vertex.y() << " " << vertex.z() << std::endl;
    }

    streamFace(index + 1, index + 3, index + 4);
    streamFace(index + 3, index + 5, index + 6);
    streamFace(index + 0, index + 7, index + 5);
    streamFace(index + 7, index + 2, index + 4);
    streamFace(index + 5, index + 3, index + 1);
    streamFace(index + 7, index + 0, index + 2);
    streamFace(index + 2, index + 1, index + 4);
    streamFace(index + 4, index + 3, index + 6);
    streamFace(index + 5, index + 7, index + 6);
    streamFace(index + 6, index + 7, index + 4);
    streamFace(index + 0, index + 5, index + 1);
    index += 8;
  }

  ofs << vertices_stream.str();
  ofs << faces_stream.str();

  ofs.close();
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

void SaveBoxPred(std::vector<centerpoint::Box3D>& boxes, std::string file_name)
{
    std::ofstream ofs;
    ofs.open(file_name, std::ios::out);
    ofs.setf(std::ios::fixed, std::ios::floatfield);
    ofs.precision(5);
    if (ofs.is_open()) {
        for (const auto box : boxes) {
          ofs << box.label << " ";
          ofs << box.score << " ";
          ofs << box.x << " ";
          ofs << box.y << " ";
          ofs << box.z << " ";
          ofs << box.length << " ";
          ofs << box.width << " ";
          ofs << box.height << " ";
          ofs << box.yaw << " ";
          ofs << box.vel_x << " ";
          ofs << box.vel_y << " ";
          ofs << "\n";
        }
    }
    else {
      std::cerr << "Output file cannot be opened!" << std::endl;
    }
    ofs.close();
    std::cout << "Saved prediction in: " << file_name << std::endl;
    return;
}

int main() {
    EventTimer timer_;
    centerpoint::CenterPointConfig config(3, 4, 40000, {-76.8, -76.8, -4.0, 76.8, 76.8, 6.0}, 
        {0.32, 0.32, 10.0}, 1, 9, 0.35, 0.5, {0.3, 0.3, 0.3, 0.3, 0.0}, 0);
    centerpoint::CenterPointConfig config_1(3, 4, 40000, {-76.8, -76.8, -4.0, 76.8, 76.8, 6.0}, 
        {0.32, 0.32, 10.0}, 1, 9, 0.35, 0.5, {0.3, 0.3, 0.3, 0.3, 0.0}, 1);

    std::string input_path = "../data/test/";
    std::string output_path = "../data/";
    std::string detected_object_output_path = "../data/test/objects/";
    std::string precision = "int8";
    std::string data_file = "../data/38.pcd.bin";
    std::string encoder_onnx = "../model/pts_voxel_encoder_centerpoint.onnx";
    std::string encoder_engine = "../model/pts_voxel_encoder_centerpoint.engine";
    std::string head_onnx = "../model/pts_backbone_neck_head_centerpoint.onnx";
    std::string head_engine = "../model/pts_backbone_neck_head_centerpoint.engine";

    std::vector<float> timing_pre_voxel_;
    std::vector<float> timing_pre_;
    std::vector<float> timing_voxel_trt_;
    std::vector<float> timing_seletc_;
    std::vector<float> timing_head_trt_;
    std::vector<float> timing_post_;

    // unsigned int length = 0;
    // void *data = NULL;
    // std::shared_ptr<char> buffer((char *)data, std::default_delete<char[]>());
    // loadData(data_file.data(), &data, &length);
    // buffer.reset((char *)data);

    // float* points = (float*)buffer.get();
    // size_t points_size = length/sizeof(float)/4;

    // std::cout << "point len : " << length << std::endl;

    // float *points_data = nullptr;
    // unsigned int points_data_size = points_size * 4 * sizeof(float);

    std::vector<float> points_vec;
    read_cloud(data_file, points_vec);
    std::cout << "point len : " << points_vec.size() << std::endl;
    // for(auto i = 0; i < 1000; ++i) {
    //     std::cout  <<points_vec[i] << std::endl;
    // }

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

    for(int i = 0; i < 200; ++i) {

        std::fill(voxels_.begin(), voxels_.end(), 0);
        std::fill(coordinates_.begin(), coordinates_.end(), -1);
        std::fill(num_points_per_voxel_.begin(), num_points_per_voxel_.end(), 0);
        CHECK_CUDA_ERROR(cudaMemsetAsync(
            encoder_in_features_d_.get(), 0, encoder_in_feature_size_ * sizeof(float), stream_));
        CHECK_CUDA_ERROR(
            cudaMemsetAsync(spatial_features_d_.get(), 0, spatial_features_size_ * sizeof(float), stream_));
        
        CHECK_CUDA_ERROR(cudaStreamSynchronize(stream_));

        std::vector<float> points_vec;
        read_cloud(input_path + std::to_string(i) + ".pcd.bin", points_vec);
        std::cout << "point len : " << points_vec.size() << std::endl;
        
        // centerpoint::random_input(points_vec);
        // std::cout << points_vec[0] << " " << points_vec[1] << " " << points_vec[100] << " " << points_vec[200] << std::endl;
        auto start = std::chrono::high_resolution_clock::now();
        int num_voxels_ = vg_ptr_->pointsToVoxels(points_vec, voxels_, coordinates_, num_points_per_voxel_);

        // std::cout << "voxel_: ";
        // for(int j=0; j<1000; ++j) {
        //     std::cout << voxels_[j] << " , ";
        // }
        // std::cout << std::endl;

        // std::cout << "coordinates_: ";
        // for(int j=0; j<1000; ++j) {
        //     std::cout << coordinates_[j] << " , ";
        // }
        // std::cout << std::endl;

        // std::cout << "num_points_per_voxel_: ";
        // for(int j=0; j<1000; ++j) {
        //     std::cout << num_points_per_voxel_[j] << " , ";
        // }
        // std::cout << std::endl;

        auto end = std::chrono::high_resolution_clock::now();
        auto count = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        timing_pre_voxel_.push_back(count/1000000.0);

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

        CHECK_CUDA_ERROR(cudaStreamSynchronize(stream_));

        timer_.start(stream_);
        CHECK_CUDA_ERROR(centerpoint::generateFeatures_launch(
            voxels_d_.get(), num_points_per_voxel_d_.get(), coordinates_d_.get(), num_voxels_,
            config.max_voxel_size_, config.voxel_size_x_, config.voxel_size_y_, config.voxel_size_z_,
            config.range_min_x_, config.range_min_y_, config.range_min_z_, encoder_in_features_d_.get(),
            stream_));
        timing_pre_.push_back(timer_.stop("GenerateFeatures_kernel", true));

        // have to collect min and max for int8 calibration
        std::vector<float> encoder_in_feature(encoder_in_feature_size_);
        CHECK_CUDA_ERROR(cudaMemcpyAsync(
        encoder_in_feature.data(), encoder_in_features_d_.get(), encoder_in_feature_size_ * sizeof(float),
        cudaMemcpyDeviceToHost, stream_));
        CHECK_CUDA_ERROR(cudaStreamSynchronize(stream_));

        float e_min = 100000.0, e_max = -100000.0;
        for(float i : encoder_in_feature) {
            e_min = std::min(e_min, i);
            e_max = std::max(e_max, i);
        }

        std::cout << "encoder_in_features_d_ : max = " << e_max << " min = " << e_min << std::endl;

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

        // have to collect min and max for int8 calibration
        std::vector<float> spatial_features(spatial_features_size_);
        CHECK_CUDA_ERROR(cudaMemcpyAsync(
        spatial_features.data(), spatial_features_d_.get(), spatial_features_size_ * sizeof(float),
        cudaMemcpyDeviceToHost, stream_));
        CHECK_CUDA_ERROR(cudaStreamSynchronize(stream_));

        float s_min = 100000.0, s_max = -100000.0;
        for(float i : spatial_features) {
            s_min = std::min(s_min, i);
            s_max = std::max(s_max, i);
        }

        std::cout << "spatial_features_d_ : max = " << s_max << " min = " << s_min << std::endl;

        std::vector<void *> head_buffers = {spatial_features_d_.get(), head_out_heatmap_d_.get(),
                                            head_out_offset_d_.get(),  head_out_z_d_.get(),
                                            head_out_dim_d_.get(),     head_out_rot_d_.get(),
                                            head_out_vel_d_.get()};
        timer_.start(stream_);
        head_trt_ptr_->context_->enqueueV2(head_buffers.data(), stream_, nullptr);
        timing_head_trt_.push_back(timer_.stop("Head_trt", true));

        std::vector<centerpoint::Box3D> det_boxes3d;
        timer_.start(stream_);
        CHECK_CUDA_ERROR(post_proc_ptr_->generateDetectedBoxes3D_launch(
            head_out_heatmap_d_.get(), head_out_offset_d_.get(), head_out_z_d_.get(), head_out_dim_d_.get(),
            head_out_rot_d_.get(), head_out_vel_d_.get(), det_boxes3d, stream_));
        timing_post_.push_back(timer_.stop("Post_kernel", true));
        std::cout << "detect size : " << det_boxes3d.size() << std::endl;

        // Uncomment these if outputs needed
        
        // write_cloud(output_path + std::to_string(i) + ".pcd", points_vec);
        // dumpDetectionsAsMesh(det_boxes3d, output_path + std::to_string(i) + ".ply");
        SaveBoxPred(det_boxes3d, detected_object_output_path + std::to_string(i) + ".txt")
    }

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