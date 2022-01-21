# fountain-P11 dataset
url: https://documents.epfl.ch/groups/c/cv/cvlab-unit/www/data/multiview/denseMVS.html

## Usage
```bash
cd input/fountain/
mkdir urd

curl -O https://documents.epfl.ch/groups/c/cv/cvlab-unit/www/data/multiview/data/fountain_dense/urd/fountain_dense_images.tar.gz
tar -zxvf fountain_dense_images.tar.gz

curl -O https://documents.epfl.ch/groups/c/cv/cvlab-unit/www/data/multiview/data/fountain_dense/urd/fountain_dense_cameras.tar.gz
tar -zxvf fountain_dense_cameras.tar.gz

curl -O https://documents.epfl.ch/groups/c/cv/cvlab-unit/www/data/multiview/data/fountain_dense/urd/fountain_dense_p.tar.gz
tar -zxvf fountain_dense_p.tar.gz

mv fountain_dense/urd/* urd/

# to generate p11.txt
python3 gen_fou.py
```

## File format

`urd/0000.png.camera`

### K(intrinsic matrix)

```bash
fx, 0, cx,
0, fy, cy,
0, 0, 1
```

```bash
# line 1~3
2759.48 0 1520.69 
0 2764.16 1006.81 
0 0 1
```

### Rwc(Rotation matrix from Camera to World, with respect to the World)

```bash
Rwc = 
[R00, R01, R02,
R10, R11, R12,
R20, R21, R22]
```

```bash
# line 5~7
0.450927 -0.0945642 -0.887537 
-0.892535 -0.0401974 -0.449183 
0.00679989 0.994707 -0.102528 
```



### twc(translation vector from Camera to World, the location of the optic center of camera in the world frame)

```bash
twc = 
[tx, ty, tz]
```

```bash
# line 8
-7.28137 -7.57667 0.204446 
```


### width and height of `urd/0000.png`

```bash
width height
```

```bash
# line 9
3072 2048
```


<br>
<br>


`urd/0000.png.P`

### perspective projection matrix P
```bash
P = K[Rcw|tcw] = K[Rwc^(-1)| -Rwc^(-1)*twc] = K[Rwc^t| -Rwc^t*twc]
```
