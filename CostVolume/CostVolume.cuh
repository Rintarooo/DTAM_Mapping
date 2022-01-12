#ifndef COSTVOLUME_CUH
#define COSTVOLUME_CUH 

// opencv
#include <opencv2/opencv.hpp>

// cuda
#include <opencv2/cudev/common.hpp>// CV_CUDEV_SAFE_CALL

__device__ __forceinline__
float3 pix2cam(const float2 &uv_r, const float* Kinv, float zr) 
{
/*
// http://www.cse.psu.edu/~rtc12/CSE486/lecture13.pdf
2D[ur, vr, 1]^T -> 3D[xr, yr, zr]^T, // 3D point x = π^-1(u,d) = 1/d*K^(-1)u^
[xr, yr, zr]^T = 1/d * [[1/fx,0,-cx/fx],[0,1/fy,-cy/fy],[0,0,1]] * [ur, vr, 1]^T
d is inverse depth, zr(=1/d) is depth, divide by 0 is evaluated as Inf, as per IEEE-754
*/
   return make_float3(
      (Kinv[0]*uv_r.x + Kinv[2])*zr,
      (Kinv[4]*uv_r.y + Kinv[5])*zr,
      zr);
}

__device__ __forceinline__
float3 cam2cam(const float3 &xyz_r, const float* Tmr) 
{
/*
3D[xr, yr, zr, 1]^T -> 3D[xm, ym, zm, 1]^T, // xm=Tmr*xr
[xm,ym,zm,1]^T = [[R00,R01,R02,t1],[R10,R11,R12,t2],[R20,R21,R22,t3],[0,0,0,1]] * [xr,yr,zr,1]^T
*/
   return make_float3(
      Tmr[0]*xyz_r.x + Tmr[1]*xyz_r.y + Tmr[2]*xyz_r.z  + Tmr[3],
      Tmr[4]*xyz_r.x + Tmr[5]*xyz_r.y + Tmr[6]*xyz_r.z  + Tmr[7],
      Tmr[8]*xyz_r.x + Tmr[9]*xyz_r.y + Tmr[10]*xyz_r.z + Tmr[11]);
}

__device__ __forceinline__
float2 cam2pix(const float3 &xyz_m, const float* K) 
{
/*
3D[xm, ym, zm, 1]^T -> 2D[um, vm, 1]^T, // u=π(K*x)=1/z*K*x
[um, vm, 1]^T = 1/zm * [[fx,0,cx],[0,fy,cy],[0,0,1]] * [xm, ym, zm]^T
*/
   return make_float2(
      K[0]*(xyz_m.x/xyz_m.z) + K[2],
      K[4]*(xyz_m.y/xyz_m.z) + K[5]);
}

// static 
__host__ __forceinline__ 
int divRoundUp(int value, int radix) {
   /*
   e.g., default: 7/3=2 -> 7/3=3 (actual digit is 7/3=2.3...)
   return the minimum number of blocks which covers the nubmer of threads
   https://on-demand.gputechconf.com/gtc/2013/jp/sessions/8003.pdf#page=12
   http://bttb.s1.valueserver.jp/wordpress/blog/2017/07/23/%E7%AB%B6%E3%83%97%E3%83%AD%E3%81%AE%E3%83%86%E3%82%AF%E3%83%8B%E3%83%83%E3%82%AF-%E5%89%B2%E3%82%8A%E7%AE%97%E3%81%AE%E5%88%87%E3%82%8A%E4%B8%8A%E3%81%92%E3%80%81%E5%9B%9B%E6%8D%A8%E4%BA%94%E5%85%A5/
   */
    return (value + radix - 1) / radix;
}


void updateCostVolumeCaller(const float* K, const float* Kinv, const float* Tmr,
                            int rows, int cols, int imageStep, 
                            float near, float far, int layers, int layerStep,
                            float* Cdata, float count,
                            float* Cmin, float* Cmax, float* CminIdx,
                            const float* reference_image_gray, const float* current_image_gray,
                            const float* gaussian_cur_gray, const float* gaussian_ref_gray,
                            const float4* reference_image, const float4* current_image,
                            std::string similarity);

#endif