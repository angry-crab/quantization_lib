#include <sstream>
#include <fstream>
#include <iostream>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "centerpoint.hpp"

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
    centerpoint::CenterPointConfig config(3, 4, 40000, {-89.6, -89.6, -3.0, 89.6, 89.6, 5.0}, 
        {0.32, 0.32, 8.0}, 1, 9, 0.35, 0.5, {0.3, 0.0, 0.3});
    std::string data_file = "../data/2.bin";

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
    for(int i = 0; i < points_size; ++i) {
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

    std::cout << "?" << std::endl;

    int res = vg_ptr_->pointsToVoxels(points_vec, voxels_, coordinates_, num_points_per_voxel_);

    std::cout << "voxel : " << res << std::endl;

    return 0;
}