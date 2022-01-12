#include "Regulariser.cuh"

// TODO pass in alphaG, betaG as parameters
static __global__ void computeG(const float* img, float* g, int w, int h, float alphaG, float betaG)
{
  // thread coordinates
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  const int i  = (x + y*w);

  // gradients gx := $\partial_{x}^{+}img$ computed using forward differences
  const float gx = (x==w-1)? 0.0f : img[i+1] - img[i];// du
  const float gy = (y==h-1)? 0.0f : img[i+w] - img[i];// dv

  // DTAM paper Equation(5), $g(\mathbf{u}) = e^{-\alpha\|\nabla \mathbf{I_r(u)}\|_2^{\beta}}$
  g[i] = expf( -alphaG * powf(sqrtf(gx*gx + gy*gy), betaG) );
}


void computeGCaller(const float* img, float* g,
                    int width, int height, int pitch,
                    float alphaG, float betaG)
{
  // TODO set dimBlock based on warp size
  dim3 dimBlock(16, 16);
  dim3 dimGrid((width  + dimBlock.x - 1) / dimBlock.x,
               (height + dimBlock.y - 1) / dimBlock.y);
 
  computeG<<<dimGrid, dimBlock>>>(img, g, width, height, alphaG, betaG);
  CV_CUDEV_SAFE_CALL(cudaGetLastError());
  CV_CUDEV_SAFE_CALL(cudaDeviceSynchronize());
  // cudaDeviceSynchronize();
  // printf("cuda error: %s\n", cudaGetErrorString(cudaGetLastError()));
  // cudaSafeCall( cudaGetLastError() );
  
}