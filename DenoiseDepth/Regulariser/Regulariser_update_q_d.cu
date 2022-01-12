#include "Regulariser.cuh"

// https://github.com/ankurhanda/TVL1Denoising/blob/master/primal_dual_udpate_copy.cu
//// update dual variable q, gradient ascent
static __global__ void update_q(const float *g, const float *d, // const input g, d
                                float *q, // input q
                                int w, int h, // dimensions: width, height
                                float sigma_q, float epsilon // parameters
                                )
{
	// thread coordinates
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	const int i  = (y * w + x);
	const int wh = (w*h);

	// gradients dd_x := $\partial_{x}^{+}d$ computed using forward differences
	// d[u+v]
	// -----------
	// | d[0+0] | d[1+0] | ... | d[w-1+0] |
	// -----------
	// | d[0+w] | d[1+w] | ... | d[w-1+w] |
	// -----------
	// -----------
	// -----------
	// | d[0+(h-1)*w] | d[1+(h-1)*w] | ... | d[w-1+(h-1)*w] |
	// -----------
	const float dd_x = (x==w-1)? 0.0f : d[i+1] - d[i];
	const float dd_y = (y==h-1)? 0.0f : d[i+w] - d[i];

	// DTAM paper, primal-dual update step
	const float qx = (q[i]    + sigma_q*g[i]*dd_x) / (1.0f + sigma_q*epsilon);
	const float qy = (q[i+wh] + sigma_q*g[i]*dd_y) / (1.0f + sigma_q*epsilon);
	// phd thesis p.136/324, equ(5.20) 
	// const float qx = (q[i]    + sigma_q*dd_x) / (1.0f + sigma_q*epsilon);
	// const float qy = (q[i+wh] + sigma_q*dd_y) / (1.0f + sigma_q*epsilon);

	// q reprojected **element-wise** as per Newcombe thesis pg. 76, 79 (sec. 3.5)
	// if the whole vector q had to be reprojected, a tree-reduction sum would have been required
	const float maxq = fmaxf(1.0f, sqrtf(qx*qx + qy*qy));
	// phd thesis p.79/324, float len = ... /lambda
	// const float maxq = fmaxf(1.0f, sqrtf(qx*qx + qy*qy)/0.05);
	q[i]    = qx / maxq;
	q[i+wh] = qy / maxq;
}

__device__
static __forceinline__ float div_q_x(const float *q, int w, int x, int i) {
	// 	p.79 phd thesis paper
	// 	if (x==0) return p[i];
	// 	else if (x==M- 1) return -p[i-1];
	//    else return p[i] - p[i-1];
	if (x == 0) return q[i];
	else if (x == w-1) return -q[i-1];
	else return q[i]- q[i-1];
}

__device__
static __forceinline__ float div_q_y(const float *q, int w, int h, int wh, int y, int i) {
	if (y == 0) return q[i+wh];// return q[i];
	else if (y == h-1) return -q[i+wh-w];// return -q[i-1];
	else return q[i+wh] - q[i+wh-w];// return q[i]- q[i-w];
}

//// update primal variable d, gradient descent 
static __global__ void update_d(const float *g, const float *a, const float *q, // const input g, a, q
								float *d, // input d
                                int w, int h, // dimensions: width, height
                                float sigma_d, float theta // parameters
                                )
{
	// thread coordinates
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	const int i  = (y * w + x);
	const int wh = (w*h);

	// div_q computed using backward differences
	// const float dqx_x = (x==0)? q[i]    - q[i+1]    : q[i]    - q[i-1];
	// const float dqy_y = (y==0)? q[i+wh] - q[i+wh+w] : q[i+wh] - q[i+wh-w];
	const float dqx_x = div_q_x(q, w, x, i);
	const float dqy_y = div_q_y(q, w, h, wh, y, i);
	const float div_q = dqx_x + dqy_y;

	d[i]  = (d[i] + sigma_d*(g[i]*div_q + a[i]/theta)) / (1.0f + sigma_d/theta);
	// d[i]  = (d[i] + sigma_d*(div_q + a[i]/theta)) / (1.0f + sigma_d/theta);
}

void update_q_dCaller(const float *g, const float *a,
                      float *q,  float *d,
                      int width, int height,
                      float sigma_q, float sigma_d, float epsilon, float theta
                      )
{
	dim3 dimBlock(16, 16);
	dim3 dimGrid((width  + dimBlock.x - 1) / dimBlock.x,
				 (height + dimBlock.y - 1) / dimBlock.y);

	update_q<<<dimGrid, dimBlock>>>(g, d, // const input g, d
									q, // input q
									width, height, // dimensions: width, height
									sigma_q, epsilon // parameters
									);
	CV_CUDEV_SAFE_CALL(cudaGetLastError());
	CV_CUDEV_SAFE_CALL(cudaDeviceSynchronize());
	
	update_d<<<dimGrid, dimBlock>>>(g, a, q, // const input g, a, q
									d, // input d
									width, height, // dimensions: width, height
									sigma_d, theta // parameters
									);
	CV_CUDEV_SAFE_CALL(cudaGetLastError());
	CV_CUDEV_SAFE_CALL(cudaDeviceSynchronize());
	// cudaDeviceSynchronize();
	// printf("cuda error: %s\n", cudaGetErrorString(cudaGetLastError()));
   // cudaSafeCall( cudaGetLastError() );
}