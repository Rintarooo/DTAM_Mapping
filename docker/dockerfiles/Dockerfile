# https://hub.docker.com/r/nvidia/cudagl/tags?page=1&ordering=last_updated&name=10.1
FROM nvidia/cudagl:10.1-devel-ubuntu18.04

# set environment variable
ENV DEBIAN_FRONTEND noninteractive
ENV CUDA_DEBUGGER_SOFTWARE_PREEMPTION 1
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility,graphics

## check opencv version with cuda version
## https://qiita.com/tomoaki_teshima/items/294c9ffb03a3dcc4cfef
# ARG OPENCV_VERSION='4.4.0'
ARG OPENCV_VERSION='3.4.11'

## check Compute Capability(CC) with the url of the official cuda website below
## https://developer.nvidia.com/cuda-gpus
## if you don't know your GPU name, check by $ nvidia-settings or $ nvidia-smi
## GPU_ARCH should be the value lower than your GPU CC, otherwise it won't work when you use cuda enabled opencv.
## I am using "NVIDIA GeForce GTX TITAN X". According to the website above, CC is 5.2. So, GPU_ARCH should be lower than 5.2
## https://qiita.com/sonots/items/1c60ccb6ffec6d04b223
ARG GPU_ARCH='5.2'
# ARG GPU_ARCH='5.0'

ARG PCL_VERSION="1.11.0"

WORKDIR /opt

# Build tools + debugger
RUN apt update && \
    apt install -y \
    sudo \
    tzdata \
    git \
    cmake \
    wget \
    unzip \
    build-essential \
    curl \
    vim \
    gdb \
    gnuplot \
    nvidia-cuda-gdb


# Media I/O:
RUN sudo apt install -y \
    zlib1g-dev \
    libjpeg-dev \
    libwebp-dev \
    libpng-dev \
    libtiff5-dev \
    libopenexr-dev \
    libgdal-dev \
    libgtk2.0-dev

# for OpenGL
# https://grainrigi.hatenablog.com/entry/2017/08/27/144218
RUN sudo apt install -y \
    libglew-dev \
    libglfw3-dev

# Video I/O:
RUN sudo apt install -y \
    libdc1394-22-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libtheora-dev \
    libvorbis-dev \
    libxvidcore-dev \
    libx264-dev \
    yasm \
    libopencore-amrnb-dev \
    libopencore-amrwb-dev \
    libv4l-dev \
    libxine2-dev \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    libopencv-highgui-dev \
    ffmpeg

# ======== Installing Parallelism and linear algebra library(Eigen) ========
RUN sudo apt install -y \
    libtbb-dev \
    libeigen3-dev

# ======== Installing Python3 ========
RUN sudo apt install -y \
    python3-dev \
    python3-tk \
    python3-numpy


# ======== Installing CUDA enabled OpenCV ========
RUN wget https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.zip && \
    unzip ${OPENCV_VERSION}.zip && rm ${OPENCV_VERSION}.zip && \
    mv opencv-${OPENCV_VERSION} OpenCV && \
    cd OpenCV && \
    wget https://github.com/opencv/opencv_contrib/archive/${OPENCV_VERSION}.zip && \
    unzip ${OPENCV_VERSION}.zip && \
    mkdir build && \
    cd build && \
    cmake \
      -D WITH_TBB=ON \
      -D CMAKE_BUILD_TYPE=RELEASE \
      -D BUILD_EXAMPLES=ON \
      -D WITH_FFMPEG=ON \
      -D WITH_V4L=ON \
      -D WITH_OPENGL=ON \
      -D WITH_CUDA=ON \
      -D CUDA_ARCH_BIN=${GPU_ARCH} \
      -D CUDA_ARCH_PTX=${GPU_ARCH} \
      -D WITH_CUBLAS=ON \
      -D WITH_CUFFT=ON \
      -D WITH_EIGEN=ON \
      -D EIGEN_INCLUDE_PATH=/usr/include/eigen3 \
      -D OPENCV_EXTRA_MODULES_PATH=../opencv_contrib-${OPENCV_VERSION}/modules/ \
      .. && \
    make all -j$(nproc) && \
    make install

# ======== Installing VTK ========
RUN apt-get update && apt-get install -y \
    libxt-dev 

RUN wget https://www.vtk.org/files/release/8.2/VTK-8.2.0.tar.gz \
    && tar -xf VTK-8.2.0.tar.gz \
    && cd VTK-8.2.0 && mkdir build && cd build \
    && cmake .. -DVTK_MODULE_ENABLE_VTK_RenderingContextOpenGL2=YES \
                -DCMAKE_BUILD_TYPE=Release \
    && make -j$(nproc) \
    && make install


# ======== Installing PCL ========
# PCL dependencies
RUN apt-get install -y \
    libeigen3-dev \
    libflann-dev \
    libglu1-mesa-dev \
    freeglut3-dev \
    mesa-common-dev \
    libboost-all-dev \
    libusb-1.0-0-dev \
    libusb-dev \
    libopenni-dev \
    libopenni2-dev \
    libpcap-dev \
    libpng-dev \
    mpi-default-dev \
    openmpi-bin \
    openmpi-common \
    libqhull-dev \
    libgtest-dev

RUN wget https://github.com/PointCloudLibrary/pcl/archive/pcl-${PCL_VERSION}.tar.gz \
    && tar -xf pcl-${PCL_VERSION}.tar.gz \
    && cd pcl-pcl-${PCL_VERSION} \
    && mkdir build \
    && cd build \
    && cmake .. -DCMAKE_BUILD_TYPE=Release \
                -DVTK_RENDERING_BACKEND=OpenGL2 \
                -DWITH_CUDA:BOOL=ON \
    && make -j$(nproc)\
    && make install
                # -DBUILD_CUDA=ON \
                # -DBUILD_GPU=ON \

RUN apt-get update && apt-get install -y pcl-tools
