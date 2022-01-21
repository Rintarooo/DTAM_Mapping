# DTAM implementation (Mapping only)

![image](https://user-images.githubusercontent.com/51239551/150547085-6b7eb87d-8c34-4c25-9585-48c8c7297d31.png)

implementation for DTAM[Newcombe+, 2011] using C++/CUDA

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