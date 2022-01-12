
## dataset of ICL NUIM
https://www.doc.ic.ac.uk/~ahanda/VaFRIC/iclnuim.html
paper: https://www.doc.ic.ac.uk/~ahanda/VaFRIC/icra2014.pdf

Living Room 'lr kt2'
Number of Images: 882.
Size: 1.9G
Frame Rate: 30Hz
Total Time: 30 secs

## Usage
download images
```bash
curl -O http://www.doc.ic.ac.uk/~ahanda/living_room_traj2_frei_png.tar.gz
tar -zxvf living_room_traj2_frei_png.tar.gz
```

download Poses TUM RGB-D format: TrajectoryGT
```bash
curl -O https://www.doc.ic.ac.uk/~ahanda/VaFRIC/livingRoom2.gt.freiburg
```

download Global Poses [R | t]: Global_RT_Trajectory_GT
```bash
curl -O https://www.doc.ic.ac.uk/~ahanda/VaFRIC/livingRoom2n.gt.sim
```
