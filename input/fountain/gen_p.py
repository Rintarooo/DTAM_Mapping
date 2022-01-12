# https://github.com/qq456cvb/SfM/blob/23d36d70591ce024483c4136fd718421c4b129df/SFM/utils.py
# https://github.com/akshitj1/dtam/blob/bd027b6f76215cc71af382f5e7015f11a726cace/mapper.cpp
import os

	
def write_imgf_R_t(txtdir = 'urd/', imgdir = 'urd/', dst = "fou11.txt"):
	with open(dst, 'w') as f:
		f.write("fountain-P11 dataset\nimgpath R(rotation) t(translation)\n")
		for i in range(11):
			src = txtdir + '00{:02d}.png.camera'.format(i)
			if not os.path.isfile(src):
				raise FileNotFoundError(src)
			imgfilename = src[-15:-7]
			# print(imgfilename)
			if not os.path.isdir(imgdir):
				raise NotADirectoryError(imgdir)
			if not os.path.isfile(imgdir + imgfilename):
				raise FileNotFoundError(imgdir + imgfilename)
			# print(imgdir + imgfilename)
			lines = open(src).readlines()
			
			f.write(imgdir+src[-15:-7])# "imgdir+00{:02d}.png".format(i)
			f.write("\t")
			for line in lines[4:7]:
				R =line.split()
				f.write(f'{R[0]}\t{R[1]}\t{R[2]}\t')
			t = lines[7].split()
			f.write(f'{t[0]}\t{t[1]}\t{t[2]}\n')
			

if __name__ == '__main__':
	txtdir, imgdir, dst = 'urd/', 'urd/', "p11.txt"
	write_imgf_R_t(txtdir, imgdir, dst)