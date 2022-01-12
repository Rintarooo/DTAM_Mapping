#include "Regulariser.cuh"

// Using a different version of the accelerated search method:
// a_min must lie between [(d_i-d_min), (d_i+d_min)]// Eaux = data term(Cost Volume) + regularisation term + coupling term

__device__
static __forceinline__ int set_start_layer(float di, float r, float far, float depthStep, int layers){
    const float d_start = di - r;                                 
    const int start_layer = lrintf(floorf((d_start - far)/depthStep)) - 1;  
    return (start_layer<0)? 0 : start_layer;// start_layer >= 0                           
}

__device__
static __forceinline__ int set_end_layer(float di, float r, float far, float depthStep, int layers){
    const float d_end = di + r;                                   
    const int end_layer = lrintf(ceilf((d_end - far)/depthStep)) + 1;     
    // int end_layer = 255;// int layer = int((d_end - far)/depthStep) + 1;
    return  (end_layer>(layers-1))? (layers-1) : end_layer;// end_layer <= layers-1
}

// DTAM paper equ(10); total energy Eaux = data term(Cost Volume) + regularisation term + coupling term
// phd thesis p.136/324, equ(5.19), 5.3. Global Cost Volume Optimisation
// DTAM paper equ(13)(14)
__device__
static __forceinline__ float get_Eaux(float theta, float di, float aIdx, float far, float depthStep, float lambda, float scale_Eaux, float costval)
{
	const float ai = far + aIdx*depthStep;
	return scale_Eaux*(0.5f/theta)*((di-ai)*(di-ai)) + lambda*costval; 
	
	// return (0.5f/theta)*((di-ai)*(di-ai)) + lambda*costval; 
	// return 100*(0.5f/theta)*((di-ai)*(di-ai)) + lambda*costval; 
	// return 10000*(0.5f/theta)*((di-ai)*(di-ai)) + lambda*costval; 
	// return 1000000*(0.5f/theta)*((di-ai)*(di-ai)) + lambda*costval; 
	// return 10000000*(0.5f/theta)*((di-ai)*(di-ai)) + lambda*costval; 
}

// a which minimizes Eaux
static __global__ void minimizeEaux(const float* cost, int rows, int cols,
								 float* a, const float* d,
								 const float* d_Cmin, const float* C_min, const float*C_max,
								 float far, float near, int layers,
								 float theta, float lambda, float scale_Eaux, float*Eaux_)
{
	// thread coordinate
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int i = x + y*cols;

	const float depthStep = (near - far) / (layers - 1);
	const int layerStep = rows*cols;
	const float di = d[i];

	// lambda = 1/(1 + 0.5*(1./d_Cmin[i]));// For the first key-frame we set λ = 1, in DTAM paper // phd thesis pg.140/324,  λ should reflect the data term quality 
	
	// const float r = sqrtf(2*theta*lambda*(C_max[i] - C_min[i]));
	const float r = 2*theta*lambda*(C_max[i] - C_min[i]);// phd thesis pg.137/324, equ(5.24)
	const int start_layer = set_start_layer(di, r, far, depthStep, layers);
	const int end_layer = set_end_layer(di, r, far, depthStep, layers);
	int minl = 0;
	float Eaux_min = 1e+30;
	for(int l = start_layer; l <= end_layer; l++) {
		// // l = aIdx
		const float cost_total = get_Eaux(theta, di, (float)l, far, depthStep, lambda, scale_Eaux, cost[i+l*layerStep]);
		Eaux_[i+l*layerStep] = cost_total;
		if(cost_total < Eaux_min) {
			Eaux_min = cost_total;
			minl = l;
		}
	}

	a[i] = far + float(minl)*depthStep;
	if(minl == start_layer || minl == end_layer) return;// if(minl == 0 || minl == layers-1) // first or last was best

	// sublayer sampling as the minimum of the parabola with the 2 points around (minl, Eaux_min)
	const float A = get_Eaux(theta, di, minl-1, far, depthStep, lambda, scale_Eaux, cost[i+(minl-1)*layerStep]);
	const float B = Eaux_min;
	const float C = get_Eaux(theta, di, minl+1, far, depthStep, lambda, scale_Eaux, cost[i+(minl+1)*layerStep]);
	// float delta = ((A+C)==2*B)? 0.0f : ((A-C)*depthStep)/(2*(A-2*B+C));
	float delta = ((A+C)==2*B)? 0.0f : ((C-A)*depthStep)/(2*(A-2*B+C));
	delta = (fabsf(delta) > depthStep)? 0.0f : delta;
	// a[i] += delta;
	a[i] -= delta;
}

void minimizeEauxCaller(const float *cdata, int rows, int cols,
					 float *a, const float *d,
					 const float*d_Cmin, const float*C_min, const float*C_max,
					 float far, float near, int layers,
					 float theta, float lambda, float scale_Eaux, float*Eaux_)
{
	dim3 dimBlock(16, 16);
	dim3 dimGrid((cols + dimBlock.x - 1) / dimBlock.x,
				 (rows + dimBlock.y - 1) / dimBlock.y);

	minimizeEaux<<<dimGrid, dimBlock>>>(cdata, rows, cols,
									 a, d,
									 d_Cmin, C_min, C_max,
									 far, near, layers,
									 theta, lambda, scale_Eaux, Eaux_);

	CV_CUDEV_SAFE_CALL(cudaGetLastError());
	CV_CUDEV_SAFE_CALL(cudaDeviceSynchronize());
	// cudaDeviceSynchronize();
	// printf("cuda error: %s\n", cudaGetErrorString(cudaGetLastError()));
	// cudaSafeCall( cudaGetLastError() );
}
