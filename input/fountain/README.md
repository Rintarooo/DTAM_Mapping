# fountain-P11 dataset
url: https://documents.epfl.ch/groups/c/cv/cvlab-unit/www/data/multiview/denseMVS.html

## Usage
```bash
curl -O https://documents.epfl.ch/groups/c/cv/cvlab-unit/www/data/multiview/data/fountain_dense/urd/fountain_dense_images.tar.gz
tar -zxvf fountain_dense_images.tar.gz

curl -O https://documents.epfl.ch/groups/c/cv/cvlab-unit/www/data/multiview/data/fountain_dense/urd/fountain_dense_cameras.tar.gz
tar -zxvf fountain_dense_cameras.tar.gz

curl -O https://documents.epfl.ch/groups/c/cv/cvlab-unit/www/data/multiview/data/fountain_dense/urd/fountain_dense_p.tar.gz
tar -zxvf fountain_dense_p.tar.gz

python3 gen_fou.py
```

## File format
`urd/0000.png.camera`

### K(intrinstic parameters)

fx, 0.0,  cx,
0.0, fy,  cy,
0.0,0.0, 1.0

2759.48 0 1520.69 
0 2764.16 1006.81 
0 0 1 


0 0 0

### Rwc(Rotation matrix, from Camera to World, with respect to the World)

R00, R01, R02,
R10, R11, R12,
R20, R21, R22

0.450927 -0.0945642 -0.887537 
-0.892535 -0.0401974 -0.449183 
0.00679989 0.994707 -0.102528 

### twc(translation, the location of the optic center of camera in the world frame)

tx, ty, tz

-7.28137 -7.57667 0.204446 

### width height

3072 2048

<br>
<br>


`urd/0000.png.P`

```bash
P = K[Rcw|tcw] = K[Rwc^(-1)|-Rwc^(-1)*twc] = K[Rwc^t|-Rwc^t*twc]
```
