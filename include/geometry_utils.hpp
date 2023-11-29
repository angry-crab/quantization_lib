#ifndef CENTERPOINT_GEOMETRY_UTILS_HPP
#define CENTERPOINT_GEOMETRY_UTILS_HPP

#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "utils.hpp"

namespace centerpoint
{

constexpr double pi = 3.14159265358979323846;

std::vector<Eigen::Vector3d> getVertices(
  const Box3D & shape, const Eigen::Affine3d & pose)
{
    std::vector<Eigen::Vector3d> vertices;
    Eigen::Vector3d vertex;

    vertex.x() = -shape.length / 2.0;
    vertex.y() = -shape.width / 2.0;
    vertex.z() = shape.height / 2.0;
    vertices.push_back(pose * vertex);

    vertex.x() = -shape.length / 2.0;
    vertex.y() = shape.width / 2.0;
    vertex.z() = shape.height / 2.0;
    vertices.push_back(pose * vertex);

    vertex.x() = -shape.length / 2.0;
    vertex.y() = shape.width / 2.0;
    vertex.z() = -shape.height / 2.0;
    vertices.push_back(pose * vertex);

    vertex.x() = shape.length / 2.0;
    vertex.y() = shape.width / 2.0;
    vertex.z() = shape.height / 2.0;
    vertices.push_back(pose * vertex);

    vertex.x() = shape.length / 2.0;
    vertex.y() = shape.width / 2.0;
    vertex.z() = -shape.height / 2.0;
    vertices.push_back(pose * vertex);

    vertex.x() = shape.length / 2.0;
    vertex.y() = -shape.width / 2.0;
    vertex.z() = shape.height / 2.0;
    vertices.push_back(pose * vertex);

    vertex.x() = shape.length / 2.0;
    vertex.y() = -shape.width / 2.0;
    vertex.z() = -shape.height / 2.0;
    vertices.push_back(pose * vertex);

    vertex.x() = -shape.length / 2.0;
    vertex.y() = -shape.width / 2.0;
    vertex.z() = -shape.height / 2.0;
    vertices.push_back(pose * vertex);

    return vertices;
}

void Box3DtoAffine(const Box3D& box, Eigen::Affine3d& affine) {
    Eigen::Quaterniond q;
    float roll = 0.0, pitch = 0.0, yaw = -box.yaw - pi / 2;
    q = Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX())
        * Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY())
        * Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ());
    
    Eigen::Matrix4d trans;
    trans.setIdentity();
    trans.block<3,3>(0,0) = q.toRotationMatrix();
    Eigen::Vector3d translation(box.x, box.y, box.z);
    trans.block<3,1>(0,3) = translation;

    affine.matrix() = trans;

    // std::cout << translation << std::endl;
}


}  // namespace centerpoint

#endif  // CENTERPOINT_GEOMETRY_UTILS_HPP
