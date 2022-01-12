# DTAM implementation(Mapping only)

in progress.

## Usage


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

## debug

### memory checker

```bash
cuda-memcheck ./build/main input/json/fountain.json
```
<br>

### cuda-gdb

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