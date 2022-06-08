# DTAM implementation (Mapping only)

re-implementation for [DTAM: Dense Tracking and Mapping[Newcombe+, 2011]](https://www.doc.ic.ac.uk/~ajd/Publications/newcombe_etal_iccv2011.pdf) for Mapping only using C++/CUDA

![image1](https://user-images.githubusercontent.com/51239551/150547085-6b7eb87d-8c34-4c25-9585-48c8c7297d31.png)

<!-- ![image2](https://user-images.githubusercontent.com/51239551/172653443-5fdbf493-28fa-41df-8eed-0622fe199c57.png)
 -->
<img src="https://user-images.githubusercontent.com/51239551/172653443-5fdbf493-28fa-41df-8eed-0622fe199c57.png" width="870">


## Usage

### set up your environment using Docker (optional)
follow the instruction in `docker/README.md`
```bash
cat docker/README.md
```
<br>

### download `fountain-P11` dataset
follow the instruction in `input/fountain/README.md`
```bash
cat input/fountain/README.md
```
<br>

### build
```bash
./run.sh
```
<br>

### run
```bash
./build/main input/json/fountain.json
```
<br>

### change parameters as you like
```bash
vim input/json/fountain.json
```
<br>

## debug CUDA kernel (optional)

### 1. memory checker

```bash
cuda-memcheck ./build/main input/json/fountain.json
```
<br>

### 2. cuda-gdb

```bash
cuda-gdb ./build/main
```
<br>

set breakpoint at cuda kernel `updateCostVolume`
```bash
(cuda-gdb) b updateCostVolume
```
<br>

run program with argument
```bash
(cuda-gdb) r input/json/fountain.json
```
<br>