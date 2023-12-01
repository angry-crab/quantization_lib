# quantization_lib

Docker
nvcr.io/nvidia/tensorrt:22.07-py3 (For desktop)

sudo apt update
sudo apt install libeigen3-dev -y
sudo apt install libpcl-dev -y

pip3 install open3d --ignore-installed

To see plot from Docker:
xhost local:root

python3 scripts/vis.py 
OR use CloudCompare to load .pcd and .ply files