#include "CostVolume.cuh"

#define NORM_SAD 1/(powf(3, 9.)*9.)
#define NORM_SSD 1/(powf(3, 9.)*9.*255)

static texture<float4, cudaTextureType2D, cudaReadModeElementType> cur_imgTex;
static texture<float4, cudaTextureType2D, cudaReadModeElementType> ref_imgTex;

static __global__ void updateCostVolume_pixel(const float* K, const float* Kinv, const float* Tmr,
										int rows, int cols,
										float near, float far, int layers, int layerStep,
										float* Cost, float count,
										float* Cmin, float* Cmax, float* CminIdx)
{
	const int ur = blockIdx.x * blockDim.x + threadIdx.x;
	const int vr = blockIdx.y * blockDim.y + threadIdx.y;
	const int i = ur + vr*cols;// cols = width
	const float4 Ir = tex2D(ref_imgTex, ur+0.5, vr+0.5);
	const float2 uv_r = make_float2(ur, vr);

	const float depthStep = (near - far)/(layers-1);
	int minl = layers-1;
	float Cost_min = 1e+30, Cost_max = 0;// FLT_MAX
	for(int l=layers-1; l >= 0; l--) {
		// const float d = far + __int2float_rn(l)*depthStep;// d is inverse depth, l is cost volume layer index
		const float zr = 1.0/(far + float(l)*depthStep);// zr is depth in r camera frame 
		const float3 xyz_r = pix2cam(uv_r, Kinv, zr);
		const float3 xyz_m = cam2cam(xyz_r, Tmr);
		const float2 uv_m = cam2pix(xyz_m, K);
		const float um = uv_m.x, vm = uv_m.y;
		const float4 Im = tex2D(cur_imgTex, um+0.5, vm+0.5);
		// printf("Im.x:%f, Ir.x:%f, Im.y:%f, Ir.y:%f, Im.z:%f Ir.z:%f\n", Im.x, Ir.x, Im.y, Ir.y, Im.z, Ir.z);
		float rho = fabsf(Im.x - Ir.x) + fabsf(Im.y - Ir.y) + fabsf(Im.z - Ir.z);
		rho /= 3;
		if((um > cols-1 || um < 0 || vm > rows-1 || vm < 0)) rho = Cost[i+l*layerStep];
		Cost[i+l*layerStep] = (Cost[i+l*layerStep]*(count-1) + rho) / count;
		const float Cost_l = Cost[i+l*layerStep];
		if(Cost_l < Cost_min) {
			Cost_min = Cost_l;
			minl = l;
		}
		Cost_max = fmaxf(Cost_l, Cost_max);  
	}

	Cmin[i]	   = Cost_min;
	CminIdx[i] = far + float(minl)*depthStep;// inverse depth which minimize photomeric error
	Cmax[i]	   = Cost_max;
	if(minl == 0 || minl == layers-1) return;// first or last inverse depth was best
	const float A = Cost[i+(minl-1)*layerStep];
	const float B = Cost_min;
	const float C = Cost[i+(minl+1)*layerStep];
	float delta = ((A+C)==2*B)? 0.0f : ((C-A)*depthStep)/(2*(A-2*B+C));
	delta = (fabsf(delta) > depthStep)? 0.0f : delta;// if the gradient descent step is less than one whole inverse depth interval, reject interpolation
	CminIdx[i] += delta;
}

static __global__ void updateCostVolume_SAD(const float* K, const float* Kinv, const float* Tmr,
										int rows, int cols,
										float near, float far, int layers, int layerStep,
										float* Cost, float count,
										float* Cmin, float* Cmax, float* CminIdx)
{
	const int ur = blockIdx.x * blockDim.x + threadIdx.x;
	const int vr = blockIdx.y * blockDim.y + threadIdx.y;
	const int i = ur + vr*cols;// cols = width
	const float2 uv_r = make_float2(ur, vr);

	float4 Ir_patch[3][3];// https://github.com/HKUST-Aerial-Robotics/open_quadtree_mapping/blob/no_cudaopencv/src/depth_extract.cu#L167-L174
	for(int xx = -1; xx <= 1; xx++){
		for(int yy = -1; yy <= 1; yy++){
			Ir_patch[xx+1][yy+1] = tex2D(ref_imgTex, ur+xx+0.5, vr+yy+0.5);
		}
	}

	const float depthStep = (near - far)/(layers-1);
	int minl = layers-1;
	float Cost_min = 1e+30, Cost_max = 0;// FLT_MAX
	for(int l=layers-1; l >= 0; l--) {
		const float zr = 1.0/(far + float(l)*depthStep);
		const float3 xyz_r = pix2cam(uv_r, Kinv, zr);
		const float3 xyz_m = cam2cam(xyz_r, Tmr);
		const float2 uv_m = cam2pix(xyz_m, K);
		const float um = uv_m.x, vm = uv_m.y;
		// if((um < cols-1) && (um >= 1) && (vm < rows-1) && (vm >= 1)){
		float rho = 0.0f;
		for(int uu = -1; uu <= 1; uu++){
			for(int vv = -1; vv <= 1; vv++){
				const float4 Im = tex2D(cur_imgTex, um+uu+0.5, vm+vv+0.5);
				// float4 Ir = tex2D(ref_imgTex, ur+uu+0.5, vr+vv+0.5);
				const float4 Ir = Ir_patch[uu+1][vv+1];
				rho += fabsf(Im.x - Ir.x) + fabsf(Im.y - Ir.y) + fabsf(Im.z - Ir.z);
				// tmp += (Im.x - Ir.x)*(Im.x - Ir.x) + fabsf(Im.y - Ir.y) + fabsf(Im.z - Ir.z);
				// tmp /= 3;// for RGB
			}
		}
		rho *= NORM_SAD;
		// printf("Im.x:%f, Ir.x:%f, Im.y:%f, Ir.y:%f, Im.z:%f Ir.z:%f\n", Im.x, Ir.x, Im.y, Ir.y, Im.z, Ir.z);
		// float rho = tmp*NORM;
		// rho = tmp/(powf(3, 9.)*9.);
		// rho = tmp/9.;// for 3*3 patch
		
		// 0,1,2,...,n-3,n-2,n-1
		// if((um >= cols-2 || um < 2 || vm >= rows-2 || vm < 2)){
		if((um > cols-3 || um < 2 || vm > rows-3 || vm < 2)) rho = Cost[i+l*layerStep];
		Cost[i+l*layerStep] = (Cost[i+l*layerStep]*(count-1) + rho) / count; // TODO: maintain per pixel count? Not necessary. 
		const float Cost_l = Cost[i+l*layerStep];
		if(Cost_l < Cost_min) {
			Cost_min = Cost_l;
			minl = l;
		}
		Cost_max = fmaxf(Cost_l, Cost_max);  
	}

	Cmin[i]	   = Cost_min;
	CminIdx[i] = far + float(minl)*depthStep;// inverse depth which minimize photomeric error
	Cmax[i]	   = Cost_max;
	if(minl == 0 || minl == layers-1) // first or last inverse depth was best
		return;

	const float A = Cost[i+(minl-1)*layerStep];
	const float B = Cost_min;
	const float C = Cost[i+(minl+1)*layerStep];
	float delta = ((A+C)==2*B)? 0.0f : ((C-A)*depthStep)/(2*(A-2*B+C));
	delta = (fabsf(delta) > depthStep)? 0.0f : delta;// if the gradient descent step is less than one whole inverse depth interval, reject interpolation
	CminIdx[i] += delta;
}


static __global__ void updateCostVolume_SSD(const float* K, const float* Kinv, const float* Tmr,
										int rows, int cols,
										float near, float far, int layers, int layerStep,
										float* Cost, float count,
										float* Cmin, float* Cmax, float* CminIdx)
{
	const int ur = blockIdx.x * blockDim.x + threadIdx.x;
	const int vr = blockIdx.y * blockDim.y + threadIdx.y;
	const int i = ur + vr*cols;// cols = width
	const float2 uv_r = make_float2(ur, vr);
	float4 Ir_patch[3][3];
	for(int xx = -1; xx <= 1; xx++){
		for(int yy = -1; yy <= 1; yy++){
			Ir_patch[xx+1][yy+1] = tex2D(ref_imgTex, ur+xx+0.5, vr+yy+0.5);
		}
	}
	const float depthStep = (near - far)/(layers-1);
	int minl = layers-1;
	float Cost_min = 1e+30, Cost_max = 0;
	for(int l=layers-1; l >= 0; l--) {
		const float zr = 1.0/(far + float(l)*depthStep);
		const float3 xyz_r = pix2cam(uv_r, Kinv, zr);
		const float3 xyz_m = cam2cam(xyz_r, Tmr);
		const float2 uv_m = cam2pix(xyz_m, K);
		const float um = uv_m.x, vm = uv_m.y;
		float rho = 0.0f;
		for(int uu = -1; uu <= 1; uu++){
			for(int vv = -1; vv <= 1; vv++){
				const float4 Im = tex2D(cur_imgTex, um+uu+0.5, vm+vv+0.5);
				const float4 Ir = Ir_patch[uu+1][vv+1];
				rho += (Im.x - Ir.x)*(Im.x - Ir.x) + (Im.y - Ir.y)*(Im.y - Ir.y) + (Im.z - Ir.z)*(Im.z - Ir.z);
			}
		}
		rho *= NORM_SSD;
		// float rho = tmp*NORM_SSD;
		if((um > cols-3 || um < 2 || vm > rows-3 || vm < 2)) rho = Cost[i+l*layerStep];
		Cost[i+l*layerStep] = (Cost[i+l*layerStep]*(count-1) + rho) / count; // TODO: maintain per pixel count? Not necessary. 
		const float Cost_l = Cost[i+l*layerStep];
		if(Cost_l < Cost_min) {
			Cost_min = Cost_l;
			minl = l;
		}
		Cost_max = fmaxf(Cost_l, Cost_max);  
	}

	Cmin[i]	   = Cost_min;
	CminIdx[i] = far + float(minl)*depthStep;// inverse depth which minimize photomeric error
	Cmax[i]	   = Cost_max;
	if(minl == 0 || minl == layers-1) // first or last inverse depth was best
		return;

	const float A = Cost[i+(minl-1)*layerStep];
	const float B = Cost_min;
	const float C = Cost[i+(minl+1)*layerStep];
	float delta = ((A+C)==2*B)? 0.0f : ((C-A)*depthStep)/(2*(A-2*B+C));
	delta = (fabsf(delta) > depthStep)? 0.0f : delta;// if the gradient descent step is less than one whole inverse depth interval, reject interpolation
	CminIdx[i] += delta;
}

static __global__ void updateCostVolume_float(const float* K, const float* Kinv, const float* Tmr,
										int rows, int cols,
										float near, float far, int layers, int layerStep,
										float* Cost, float count,
										float* Cmin, float* Cmax, float* CminIdx,
										// float* reference_image_gray_, float* current_image_gray_)
										const float4* reference_image, const float4* current_image)			
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int i = x + y*cols;// cols=width

	// const float Ir = reference_image_gray_[i];
	float4 Ir = reference_image[i];

	const float ur = x, vr = y;
	float2 uv_r = make_float2(ur, vr);

	const float depthStep = (near - far)/(layers-1);// near_ = 10.0, far = 0.1;
	int minl = layers-1;
	float Cost_min = FLT_MAX, Cost_max = 0;
	for(int l=layers-1; l >= 0; l--) {
		// float d = far + float(l)*depthStep;// d is inverse depth, l is cost volume index(layer)	   
		// float d = far + __int2float_rn(l)*depthStep;// d is inverse depth, l is cost volume index(layer)
		const float zr = 1.0/(far + float(l)*depthStep);
		float3 xyz_r = pix2cam(uv_r, Kinv, zr);
		float3 xyz_m = cam2cam(xyz_r, Tmr);
		float2 uv_m = cam2pix(xyz_m, K);
		const float um = uv_m.x, vm = uv_m.y;

		float rho;
		if((um > cols-1 || um < 0 || vm > rows-1 || vm < 0)){
			rho = Cost[i+l*layerStep];
		}
		else{
			// // float Im = current_image_gray_[(int)(um + vm*cols)];
			// float Im = current_image_gray_[(__float2int_rd)(um + vm*cols)];
			// // printf("Im.x:%f, Ir.x:%f, Im.y:%f, Ir.y:%f, Im.z:%f Ir.z:%f\n", Im.x, Ir.x, Im.y, Ir.y, Im.z, Ir.z);
			// rho = fabsf(Im - Ir);
			float4 Im = current_image[(__float2int_rn)(um + vm*cols)];
			rho = fabsf(Im.x - Ir.x) + fabsf(Im.y - Ir.y) + fabsf(Im.z - Ir.z);
			rho /= 3;
		
		}
		Cost[i+l*layerStep] = (Cost[i+l*layerStep]*(count-1) + rho) / count; // TODO: maintain per pixel count? Not necessary. 
		float Cost_l = Cost[i+l*layerStep];
		if(Cost_l < Cost_min) {
			Cost_min = Cost_l;
			minl = l;
		}
		Cost_max = fmaxf(Cost_l, Cost_max);  
	}

	Cmin[i]	   = Cost_min;
	CminIdx[i] = far + float(minl)*depthStep;// inverse depth which minimize photomeric error
	Cmax[i]	   = Cost_max;
	if(minl == 0 || minl == layers-1) // first or last inverse depth was best
		return;

	float A = Cost[i+(minl-1)*layerStep];
	float B = Cost_min;
	float C = Cost[i+(minl+1)*layerStep];
	float delta = ((A+C)==2*B)? 0.0f : ((C-A)*depthStep)/(2*(A-2*B+C));
	delta = (fabsf(delta) > depthStep)? 0.0f : delta;// if the gradient descent step is less than one whole inverse depth interval, reject interpolation
	CminIdx[i] += delta;
}




/*
static __global__ void updateCostVolume_texture_ZNCC(float* K, float* Kinv, float* Tmr,
										int rows, int cols,
										float near, float far, int layers, int layerStep,
										float* Cost, float count,
										float* Cmin, float* Cmax, float* CminIdx,
										float4* reference_image, float4* current_image)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int i = x + y*cols;// cols=width

	const float ur = x;
	const float vr = y;

	float4 Ir_patch[3][3];// https://github.com/HKUST-Aerial-Robotics/open_quadtree_mapping/blob/no_cudaopencv/src/depth_extract.cu#L167-L174
	// float4 Ir_mean;
	// Ir_mean.x = 0;
	// Ir_mean.y = 0;
	// Ir_mean.z = 0;
	float4 Ir_mean = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

	for(int xx = -1; xx <= 1; xx++){
			for(int yy = -1; yy <= 1; yy++){
				Ir_patch[xx+1][yy+1] = tex2D(ref_imgTex, ur+xx+0.5, vr+yy+0.5);
				float4 tmp_Ir = Ir_patch[xx+1][yy+1];
				Ir_mean += tmp_Ir;
			}
	}
	Ir_mean /= 9;


	const float depthStep = (near - far)/(layers-1);// near_ = 10.0, far = 0.1;
	
		
	int minl = layers-1;
	float Cost_min = FLT_MAX, Cost_max = 0;
	for(int l=layers-1; l >= 0; l--) {
		float d = far + float(l)*depthStep;// d is inverse depth, l is cost volume index(layer)
	   
		float zr = 1.0/d; // zr is depth, divide by 0 is evaluated as Inf, as per IEEE-754
		float xr = (Kinv[0]*ur + Kinv[2])*zr;
		float yr = (Kinv[4]*vr + Kinv[5])*zr;
		float xm = Tmr[0]*xr + Tmr[1]*yr + Tmr[2]*zr  + Tmr[3];
		float ym = Tmr[4]*xr + Tmr[5]*yr + Tmr[6]*zr  + Tmr[7];
		float zm = Tmr[8]*xr + Tmr[9]*yr + Tmr[10]*zr + Tmr[11];
		float um = K[0]*(xm/zm) + K[2];
		float vm = K[4]*(ym/zm) + K[5];
		float rho;
		
		// if((um < cols-1) && (um >= 1) && (vm < rows-1) && (vm >= 1)){
		float tmp = 0.0, Im_mean = 0.0;
		for(int uu = -1; uu <= 1; uu++){
			for(int vv = -1; vv <= 1; vv++){
				float4 Im = tex2D(cur_imgTex, um+uu+0.5, vm+vv+0.5);
				// float4 Ir = tex2D(ref_imgTex, ur+uu+0.5, vr+vv+0.5);
				float4 Ir = Ir_patch[uu+1][vv+1];
				tmp += fabsf(Im.x - Ir.x) + fabsf(Im.y - Ir.y) + fabsf(Im.z - Ir.z);
				tmp /= 3;
			}
		}
		// printf("Im.x:%f, Ir.x:%f, Im.y:%f, Ir.y:%f, Im.z:%f Ir.z:%f\n", Im.x, Ir.x, Im.y, Ir.y, Im.z, Ir.z);
		rho = tmp/9.;
		
		// if((um > cols-2 || um < 2 || vm > rows-2 || vm < 2)){
		if((um >= cols-2 || um < 2 || vm >= rows-2 || vm < 2)){
			rho = Cost[i+l*layerStep];
		}
		Cost[i+l*layerStep] = (Cost[i+l*layerStep]*(count-1) + rho) / count; // TODO: maintain per pixel count? Not necessary. 
		float Cost_l = Cost[i+l*layerStep];
		if(Cost_l < Cost_min) {
			Cost_min = Cost_l;
			minl = l;
		}
		Cost_max = fmaxf(Cost_l, Cost_max);  
	}

	Cmin[i]	   = Cost_min;
	CminIdx[i] = far + float(minl)*depthStep;// inverse depth which minimize photomeric error
	Cmax[i]	   = Cost_max;
	if(minl == 0 || minl == layers-1) // first or last inverse depth was best
		return;

	float A = Cost[i+(minl-1)*layerStep];
	float B = Cost_min;
	float C = Cost[i+(minl+1)*layerStep];
	float delta = ((A+C)==2*B)? 0.0f : ((C-A)*depthStep)/(2*(A-2*B+C));
	delta = (fabsf(delta) > depthStep)? 0.0f : delta;// if the gradient descent step is less than one whole inverse depth interval, reject interpolation
	CminIdx[i] += delta;
}
*/

	
void updateCostVolumeCaller(const float* K, const float* Kinv, const float* Tmr,
							int rows, int cols, int imageStep,
							float near, float far, int layers, int layerStep,
							float* Cdata, float count,
							float* Cmin, float* Cmax, float* CminIdx,
							const float* reference_image_gray_, const float* current_image_gray_,//)
							const float* gaussian_ref_gray_, const float*gaussian_cur_gray_,
							const float4* reference_image, const float4* current_image,
							std::string similarity)
{
	dim3 dimBlock(16, 16);
	// 16 * 16: block size, nubmer of threads per block

	// dim3 dimGrid((cols + dimBlock.x - 1) / dimBlock.x,
	// 			 (rows + dimBlock.y - 1) / dimBlock.y);
	//// dim3 dimGrid: grid size, number of blocks
	dim3 dimGrid(divRoundUp(cols, dimBlock.x), divRoundUp(rows, dimBlock.y));
	// (cuda-gdb) cuda block
	// block (36,29,0) 640/16=36

	// // Set texture reference parameters
	// cudaChannelFormatDesc channelDesc = //cudaCreateChannelDesc<uchar4>();
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
	// cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);


	cur_imgTex.normalized     = false;
	cur_imgTex.addressMode[0] = cudaAddressModeClamp;	// out of border references return first or last element
	cur_imgTex.addressMode[1] = cudaAddressModeClamp;
	cur_imgTex.filterMode     = cudaFilterModeLinear;

	// Bind current_image to the texture reference
	// size_t offset;
	// read only from global memory 
	// cudaBindTexture2D(&offset, cur_imgTex, current_image_gray_, channelDesc, cols, rows, imageStep);
	// cudaBindTexture2D(0, cur_imgTex, current_image, channelDesc, cols, rows, imageStep);
	CV_CUDEV_SAFE_CALL(cudaBindTexture2D(0, cur_imgTex, current_image, channelDesc, cols, rows, imageStep));
	
	ref_imgTex.normalized     = false;
	ref_imgTex.addressMode[0] = cudaAddressModeClamp;	// out of border references return first or last element
	ref_imgTex.addressMode[1] = cudaAddressModeClamp;
	ref_imgTex.filterMode     = cudaFilterModeLinear;

	CV_CUDEV_SAFE_CALL(cudaBindTexture2D(0, ref_imgTex, reference_image, channelDesc, cols, rows, imageStep));
	

	// cudaDeviceSynchronize();
	// printf("cuda error: %s\n", cudaGetErrorString(cudaGetLastError()));
	CV_CUDEV_SAFE_CALL(cudaGetLastError());
	CV_CUDEV_SAFE_CALL(cudaDeviceSynchronize());


	if(similarity == "per_pixel"){
			// updateCostVolume<<<1, 16>>>(K, Kinv, Tmr,
			updateCostVolume_pixel<<<dimGrid, dimBlock>>>(K, Kinv, Tmr,
								rows, cols,
								near, far, layers, layerStep,
								Cdata, count,
								Cmin, Cmax, CminIdx);
	}
	else if(similarity == "SAD"){
			updateCostVolume_SAD<<<dimGrid, dimBlock>>>(K, Kinv, Tmr,
								rows, cols,
								near, far, layers, layerStep,
								Cdata, count,
								Cmin, Cmax, CminIdx);
	}
	else if(similarity == "SSD"){
			updateCostVolume_SSD<<<dimGrid, dimBlock>>>(K, Kinv, Tmr,
								rows, cols,
								near, far, layers, layerStep,
								Cdata, count,
								Cmin, Cmax, CminIdx);
	}
	else if(similarity == "undefined"){
			updateCostVolume_float<<<dimGrid, dimBlock>>>(
											K, Kinv, Tmr,
											rows, cols,
											near, far, layers, layerStep,
											Cdata, count,
											Cmin, Cmax, CminIdx,
											// reference_image_gray_, current_image_gray_);
											reference_image, current_image);
	}
	else{
		printf("set similarity, no cuda kernel func was called");
	}


	// cudaDeviceSynchronize();
	// // cudaSafeCall(cudaGetLastError());
	// printf("cuda error: %s\n", cudaGetErrorString(cudaGetLastError()));
	CV_CUDEV_SAFE_CALL(cudaGetLastError());
	CV_CUDEV_SAFE_CALL(cudaDeviceSynchronize());

	CV_CUDEV_SAFE_CALL(cudaUnbindTexture(cur_imgTex));
	CV_CUDEV_SAFE_CALL(cudaUnbindTexture(ref_imgTex));
}
