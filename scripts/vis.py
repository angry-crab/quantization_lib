#!/usr/bin/env python
# Copyright 2022 TIER IV, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import time

import open3d as o3d


def main(args=None):

    pcd_path = "../data/0.pcd"
    detections_path = "../data/0.ply"

    mesh = o3d.io.read_triangle_mesh(detections_path)
    pcd = o3d.io.read_point_cloud(pcd_path)

    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])

    detection_lines = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
    detection_lines.paint_uniform_color([1.0, 0.0, 1.0])

    o3d.visualization.draw_geometries([mesh_frame, pcd, detection_lines])



if __name__ == "__main__":
    main()
