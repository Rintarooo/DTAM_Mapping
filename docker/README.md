<!-- ```bash
docker run --rm -it --runtime=nvidia --cap-add=SYS_PTRACE --security-opt="seccomp=unconfined" -v $HOME/coding/:/opt -e CUDA_DEBUGGER_SOFTWARE_PREEMPTION=1 rin/cuda:10.1-cudnn7-ubuntu18.04-opencv3.4.11-CC5.0-gdb
```
<br>
 -->
<!-- 
### run container
```bash
docker run --rm -it --runtime=nvidia --cap-add=SYS_PTRACE --security-opt="seccomp=unconfined" -v $HOME/coding/:/opt -e CUDA_DEBUGGER_SOFTWARE_PREEMPTION=1 rin/cuda:10.1-cudnn7-ubuntu18.04-opencv3.4.11-CC5.0-pcl1.11.0
```
<br>

### run container with GUI
```bash
docker run --rm -it --runtime=nvidia --cap-add=SYS_PTRACE --security-opt="seccomp=unconfined" -v $HOME/coding/:/opt -e CUDA_DEBUGGER_SOFTWARE_PREEMPTION=1 -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix rin/cuda:10.1-cudnn7-ubuntu18.04-opencv3.4.11-CC5.0-pcl1.11.0
```
<br>

If you cannot display images(like `bad X server connection. `), you may need to run `xhost si:localuser:$USER` or worst case `xhost local:root` before running docker container if get errors like Error: cannot open display
<br>

Ref: https://github.com/turlucode/ros-docker-gui

### run container with GUI to show PCL viewer and fix openGL error
add `-e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics`
```bash
docker run --rm -it --runtime=nvidia --cap-add=SYS_PTRACE --security-opt="seccomp=unconfined" -v $HOME/coding/:/opt -e CUDA_DEBUGGER_SOFTWARE_PREEMPTION=1 -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics rin/cuda:10.1-cudnn7-ubuntu18.04-opencv3.4.11-CC5.0-pcl1.11.0
```
<br>


### explanation for docker command

`-e CUDA_DEBUGGER_SOFTWARE_PREEMPTION=1`:

if the error, `fatal:  All CUDA devices are used for display and cannot be used while debugging. (error code = CUDBG_ERROR_ALL_DEVICES_WATCHDOGGED(0x18)
` happens during debugging, set the following environment command
```bash
export CUDA_DEBUGGER_SOFTWARE_PREEMPTION=1
```
 -->





# Usage
* update 2021,12,23
modify `dockerfiles/Dockerfile` depending on your environment
(such as your desired OpenCV version, Compute Capability(CC) of your GPU).
```bash
docker build -t $(id -un)/cudagl:10.1-devel-ubuntu18.04-opencv3.4.11-cc5.2-pcl1.11.0 ./dockerfiles/
```
<br>


make sure you built successfully
```bash
docker images | head
```
<br>

run container, `-v` option is mounting your local dir `$HOME/coding` into `/opt/` in container
```bash
docker run --rm -it --runtime=nvidia --cap-add=SYS_PTRACE --security-opt="seccomp=unconfined" -v $HOME/coding/:/opt -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix $(id -un)/cudagl:10.1-devel-ubuntu18.04-opencv3.4.11-cc5.2-pcl1.11.0
```
<br>

If you cannot display images(like `bad X server connection. `), you may need to run `xhost si:localuser:$USER` or worst case `xhost local:root` before running docker container if get errors like Error: cannot open display
<br>

Ref: https://github.com/turlucode/ros-docker-gui


## Environment
I leave my own environment below. 
* OS:
    * Linux(Ubuntu 20.04.3 LTS) 
* GPU:
    * NVIDIA® GeForce® GTX TITAN X
* CPU:
    * Intel® Core i9-7920X CPU @ 2.90GHz
    * `$ nproc` outputs 24  
* NVIDIA® Driver = 470.86
* Docker = 20.10.12
* [nvidia-docker2](https://github.com/NVIDIA/nvidia-docker)(for GPU)