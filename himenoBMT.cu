/********************************************************************

 This benchmark test program is measuring a cpu performance
 of floating point operation by a Poisson equation solver.

 If you have any question, please ask me via email.
 written by Ryutaro HIMENO, November 26, 2001.
 Version 3.0
 ----------------------------------------------
 Ryutaro Himeno, Dr. of Eng.
 Head of Computer Information Division,
 RIKEN (The Institute of Pysical and Chemical Research)
 Email : himeno@postman.riken.go.jp
 ---------------------------------------------------------------
 You can adjust the size of this benchmark code to fit your target
 computer. In that case, please chose following sets of
 (mimax,mjmax,mkmax):
 small : 33,33,65
 small : 65,65,129
 midium: 129,129,257
 large : 257,257,513
 ext.large: 513,513,1025
 This program is to measure a computer performance in MFLOPS
 by using a kernel which appears in a linear solver of pressure
 Poisson eq. which appears in an incompressible Navier-Stokes solver.
 A point-Jacobi method is employed in this solver as this method can 
 be easyly vectrized and be parallelized.
 ------------------
 Finite-difference method, curvilinear coodinate system
 Vectorizable and parallelizable on each grid point
 No. of grid points : imax x jmax x kmax including boundaries
 ------------------
 A,B,C:coefficient matrix, wrk1: source term of Poisson equation
 wrk2 : working area, OMEGA : relaxation parameter
 BND:control variable for boundaries and objects ( = 0 or 1)
 P: pressure
********************************************************************/

#include <stdio.h>

#ifdef SSMALL
#define MIMAX            33
#define MJMAX            33
#define MKMAX            65
#endif

#ifdef SMALL
#define MIMAX            65
#define MJMAX            65
#define MKMAX            129
#endif

#ifdef MIDDLE
#define MIMAX            129
#define MJMAX            129
#define MKMAX            257
#endif

#ifdef LARGE
#define MIMAX            257
#define MJMAX            257
#define MKMAX            513
#endif

#ifdef ELARGE
#define MIMAX            513
#define MJMAX            513
#define MKMAX            1025
#endif

double second();
float jacobi(int);
void initmt();
double fflop(int,int,int);
double mflops(int,double,double);

static float  p[MIMAX][MJMAX][MKMAX];
static float  a[4][MIMAX][MJMAX][MKMAX],
              b[3][MIMAX][MJMAX][MKMAX],
              c[3][MIMAX][MJMAX][MKMAX];
static float  bnd[MIMAX][MJMAX][MKMAX];
static float  wrk1[MIMAX][MJMAX][MKMAX],
              wrk2[MIMAX][MJMAX][MKMAX];

static int imax, jmax, kmax;
static float omega;

static cudaArray_t a_p;
static float *d_a;
static float *d_b;
static float *d_c;
static float *d_bnd;
static float *d_wrk1;
static size_t d_pitch;
static cudaArray_t a_wrk2;
static float *d_gosa;

#ifdef BINDLESS
cudaSurfaceObject_t t_p, t_wrk2;
#else
surface<void, cudaSurfaceType3D> t_src, t_dst;
#endif

int
main()
{
  int    i,j,k,nn;
  float  gosa;
  double cpu,cpu0,cpu1,flop,target;

  target= 60.0;
  omega= 0.8;
  imax = MIMAX-1;
  jmax = MJMAX-1;
  kmax = MKMAX-1;

  /*
   *    Initializing matrixes
   */
  initmt();
  printf("mimax = %d mjmax = %d mkmax = %d\n",MIMAX, MJMAX, MKMAX);
  printf("imax = %d jmax = %d kmax =%d\n",imax,jmax,kmax);

  nn= 3;
  printf(" Start rehearsal measurement process.\n");
  printf(" Measure the performance in %d times.\n\n",nn);

  cpu0= second();
  gosa= jacobi(nn);
  cpu1= second();
  cpu= cpu1 - cpu0;

  flop= fflop(imax,jmax,kmax);
  
  printf(" MFLOPS: %f time(s): %f %e\n\n",
         mflops(nn,cpu,flop),cpu,gosa);

  nn= (int)(target/(cpu/3.0));

  printf(" Now, start the actual measurement process.\n");
  printf(" The loop will be excuted in %d times\n",nn);
  printf(" This will take about one minute.\n");
  printf(" Wait for a while\n\n");

  /*
   *    Start measuring
   */
  cpu0 = second();
  gosa = jacobi(nn);
  cpu1 = second();

  cpu= cpu1 - cpu0;
  
  printf(" Loop executed for %d times\n",nn);
  printf(" Gosa : %e \n",gosa);
  printf(" MFLOPS measured : %f\tcpu : %f\n",mflops(nn,cpu,flop),cpu);
  printf(" Score based on Pentium III 600MHz : %f\n",
         mflops(nn,cpu,flop)/82,84);
  
  return (0);
}

void
initmt()
{
	int i,j,k;

  for(i=0 ; i<MIMAX ; i++)
    for(j=0 ; j<MJMAX ; j++)
      for(k=0 ; k<MKMAX ; k++){
        a[0][i][j][k]=0.0;
        a[1][i][j][k]=0.0;
        a[2][i][j][k]=0.0;
        a[3][i][j][k]=0.0;
        b[0][i][j][k]=0.0;
        b[1][i][j][k]=0.0;
        b[2][i][j][k]=0.0;
        c[0][i][j][k]=0.0;
        c[1][i][j][k]=0.0;
        c[2][i][j][k]=0.0;
        p[i][j][k]=0.0;
        wrk1[i][j][k]=0.0;
        bnd[i][j][k]=0.0;
      }

  for(i=0 ; i<imax ; i++)
    for(j=0 ; j<jmax ; j++)
      for(k=0 ; k<kmax ; k++){
        a[0][i][j][k]=1.0;
        a[1][i][j][k]=1.0;
        a[2][i][j][k]=1.0;
        a[3][i][j][k]=1.0/6.0;
        b[0][i][j][k]=0.0;
        b[1][i][j][k]=0.0;
        b[2][i][j][k]=0.0;
        c[0][i][j][k]=1.0;
        c[1][i][j][k]=1.0;
        c[2][i][j][k]=1.0;
        p[i][j][k]=(float)(i*i)/(float)((imax-1)*(imax-1));
        wrk1[i][j][k]=0.0;
        bnd[i][j][k]=1.0;
      }
  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
  cudaMalloc3DArray(&a_p, &channelDesc,
                    make_cudaExtent(kmax, jmax, imax),
                    cudaArraySurfaceLoadStore);
  cudaMemcpy3DParms p_p = { 0 };
  p_p.srcPtr = make_cudaPitchedPtr(p, MKMAX * sizeof(float), MKMAX, MJMAX);
  p_p.dstArray = a_p;
  p_p.extent = make_cudaExtent(kmax, jmax, imax);
  p_p.kind = cudaMemcpyHostToDevice;
  cudaMemcpy3D(&p_p);
  cudaMallocPitch(reinterpret_cast<void **>(&d_a), &d_pitch,
                  kmax * sizeof(float), 12 * imax * jmax);
  cudaMemcpy3DParms p_a = { 0 };
  for (size_t i = 0; i < 4; ++i) {
    p_a.srcPtr = make_cudaPitchedPtr(a[i], MKMAX * sizeof(float), MKMAX, MJMAX);
    p_a.dstPtr =
      make_cudaPitchedPtr(d_a + i * imax * jmax * (d_pitch / sizeof(float)),
                          d_pitch, kmax, jmax);
    p_a.extent = make_cudaExtent(kmax * sizeof(float), jmax, imax);
    p_a.kind = cudaMemcpyHostToDevice;
    cudaMemcpy3D(&p_a);
  }
  d_b = d_a + 4 * imax * jmax * (d_pitch / sizeof(float));
  cudaMemcpy3DParms p_b = { 0 };
  for (size_t i = 0; i < 3; ++i) {
    p_b.srcPtr = make_cudaPitchedPtr(b[i], MKMAX * sizeof(float), MKMAX, MJMAX);
    p_b.dstPtr =
      make_cudaPitchedPtr(d_b + i * imax * jmax * (d_pitch / sizeof(float)),
                          d_pitch, kmax, jmax);
    p_b.extent = make_cudaExtent(kmax * sizeof(float), jmax, imax);
    p_b.kind = cudaMemcpyHostToDevice;
    cudaMemcpy3D(&p_b);
  }
  d_c = d_b + 3 * imax * jmax * (d_pitch / sizeof(float));
  cudaMemcpy3DParms p_c = { 0 };
  for (size_t i = 0; i < 3; ++i) {
    p_c.srcPtr = make_cudaPitchedPtr(c[i], MKMAX * sizeof(float), MKMAX, MJMAX);
    p_c.dstPtr =
      make_cudaPitchedPtr(d_c + i * imax * jmax * (d_pitch / sizeof(float)),
                          d_pitch, kmax, jmax);
    p_c.extent = make_cudaExtent(kmax * sizeof(float), jmax, imax);
    p_c.kind = cudaMemcpyHostToDevice;
    cudaMemcpy3D(&p_c);
  }
  d_bnd = d_c + 3 * imax * jmax * (d_pitch / sizeof(float));
  cudaMemcpy3DParms p_bnd = { 0 };
  p_bnd.srcPtr = make_cudaPitchedPtr(bnd, MKMAX * sizeof(float), MKMAX, MJMAX);
  p_bnd.dstPtr = make_cudaPitchedPtr(d_bnd, d_pitch, kmax, jmax);
  p_bnd.extent = make_cudaExtent(kmax * sizeof(float), jmax, imax);
  p_bnd.kind = cudaMemcpyHostToDevice;
  cudaMemcpy3D(&p_bnd);
  d_wrk1 = d_bnd + imax * jmax * (d_pitch / sizeof(float));
  cudaMemcpy3DParms p_wrk1 = { 0 };
  p_wrk1.srcPtr =
    make_cudaPitchedPtr(wrk1, MKMAX * sizeof(float), MKMAX, MJMAX);
  p_wrk1.dstPtr = make_cudaPitchedPtr(d_wrk1, d_pitch, kmax, jmax);
  p_wrk1.extent = make_cudaExtent(kmax * sizeof(float), jmax, imax);
  p_wrk1.kind = cudaMemcpyHostToDevice;
  cudaMemcpy3D(&p_wrk1);
  cudaMalloc3DArray(&a_wrk2, &channelDesc,
                    make_cudaExtent(kmax, jmax, imax),
                    cudaArraySurfaceLoadStore);
  cudaMalloc(reinterpret_cast<void **>(&d_gosa), sizeof(float));
#ifdef BINDLESS
  cudaResourceDesc r_p;
  r_p.resType = cudaResourceTypeArray;
  r_p.res.array.array = a_p;
  cudaResourceDesc r_wrk2;
  r_wrk2.resType = cudaResourceTypeArray;
  r_wrk2.res.array.array = a_wrk2;
  cudaCreateSurfaceObject(&t_p, &r_p);
  cudaCreateSurfaceObject(&t_wrk2, &r_wrk2);
#endif
}

template<bool CALC_GOSA> static __global__ void
jacobi_kernel0(
#ifdef BINDLESS
               cudaSurfaceObject_t t_dst, cudaSurfaceObject_t t_src,
#endif
const float *__restrict__ a, size_t pitch, float omega, float *__restrict__ d_gosa)
{
  const size_t imax = MIMAX-1;
  const size_t jmax = MJMAX-1;
  const size_t kmax = MKMAX-1;
  const size_t width = pitch / sizeof(float);
  const float *b = a + 4 * imax * jmax * width; 
  const float *c = b + 3 * imax * jmax * width; 
  const float *bnd = c + 3 * imax * jmax * width; 
  const float *wrk1 = bnd + imax * jmax * width; 

  float gosa = 0.0f;

  const size_t i = blockIdx.z * blockDim.z + threadIdx.z;
  const size_t j = blockIdx.y * blockDim.y + threadIdx.y;
  const size_t k = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= 1 && i < imax - 1 &&
      j >= 1 && j < jmax - 1 &&
      k >= 1 && k < kmax - 1) {
    float s0 = a[((0 * imax + i) * jmax + j) * width + k] * surf3Dread<float>(t_src, (k  ) * sizeof(float), j  , i+1)
             + a[((1 * imax + i) * jmax + j) * width + k] * surf3Dread<float>(t_src, (k  ) * sizeof(float), j+1, i  )
             + a[((2 * imax + i) * jmax + j) * width + k] * surf3Dread<float>(t_src, (k+1) * sizeof(float), j  , i  )
             + b[((0 * imax + i) * jmax + j) * width + k] * ( surf3Dread<float>(t_src, (k  ) * sizeof(float), j+1, i+1) - surf3Dread<float>(t_src, (k  ) * sizeof(float), j-1, i+1)
                              - surf3Dread<float>(t_src, (k  ) * sizeof(float), j+1, i-1) + surf3Dread<float>(t_src, (k  ) * sizeof(float), j-1, i-1) )
             + b[((1 * imax + i) * jmax + j) * width + k] * ( surf3Dread<float>(t_src, (k+1) * sizeof(float), j+1, i  ) - surf3Dread<float>(t_src, (k+1) * sizeof(float), j-1, i  )
                               - surf3Dread<float>(t_src, (k-1) * sizeof(float), j+1, i  ) + surf3Dread<float>(t_src, (k-1) * sizeof(float), j-1, i  ) )
             + b[((2 * imax + i) * jmax + j) * width + k] * ( surf3Dread<float>(t_src, (k+1) * sizeof(float), j  , i+1) - surf3Dread<float>(t_src, (k+1) * sizeof(float), j  , i-1)
                               - surf3Dread<float>(t_src, (k-1) * sizeof(float), j  , i+1) + surf3Dread<float>(t_src, (k-1) * sizeof(float), j  , i-1) )
             + c[((0 * imax + i) * jmax + j) * width + k] * surf3Dread<float>(t_src, (k  ) * sizeof(float), j  , i-1)
             + c[((1 * imax + i) * jmax + j) * width + k] * surf3Dread<float>(t_src, (k  ) * sizeof(float), j-1, i  )
             + c[((2 * imax + i) * jmax + j) * width + k] * surf3Dread<float>(t_src, (k-1) * sizeof(float), j  , i  )
             + wrk1[(i * jmax + j) * width + k];

    float ss = ( s0 * a[((3 * imax + i) * jmax + j) * width + k] - surf3Dread<float>(t_src, (k) * sizeof(float), j, i) ) * bnd[(i * jmax + j) * width + k];

    if (CALC_GOSA) {
      gosa+= ss*ss;
      /* gosa= (gosa > ss*ss) ? a : b; */
    }

    surf3Dwrite(surf3Dread<float>(t_src, k * sizeof(float), j, i) + omega * ss, t_dst, k * sizeof(float), j, i);
  }

  if (CALC_GOSA) {
    gosa += __shfl_xor(gosa, 1);
    gosa += __shfl_xor(gosa, 2);
    gosa += __shfl_xor(gosa, 4);
    gosa += __shfl_xor(gosa, 8);
    gosa += __shfl_xor(gosa, 16);
    __shared__ float shared[32];
    unsigned int id = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
    if (id % 32u == 0) {
      shared[id / 32u] = gosa;
    }
    __syncthreads();
    if (id < blockDim.z * blockDim.y * blockDim.x / 32u) {
      gosa = shared[id];
    } else {
      gosa = 0.0f;
    }
    __syncthreads();
    gosa += __shfl_xor(gosa, 1);
    gosa += __shfl_xor(gosa, 2);
    gosa += __shfl_xor(gosa, 4);
    gosa += __shfl_xor(gosa, 8);
    gosa += __shfl_xor(gosa, 16);
    if (id == 0) {
      atomicAdd(d_gosa, gosa);
    }
  }
}

static __global__ void
jacobi_kernel1(
#ifdef BINDLESS
               cudaSurfaceObject_t t_dst, cudaSurfaceObject_t t_src
#endif
)
{
  const size_t imax = MIMAX-1;
  const size_t jmax = MJMAX-1;
  const size_t kmax = MKMAX-1;

  const size_t i = blockIdx.z * blockDim.z + threadIdx.z;
  const size_t j = blockIdx.y * blockDim.y + threadIdx.y;
  const size_t k = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= 1 && i < imax - 1 &&
      j >= 1 && j < jmax - 1 &&
      k >= 1 && k < kmax - 1) {
    surf3Dwrite(surf3Dread<float>(t_src, k * sizeof(float), j, i), t_dst, k * sizeof(float), j, i);
  }
}

float
jacobi(int nn)
{
  int i,j,k,n;
  float gosa, s0, ss;

  dim3 block(64, 4, 2);
  dim3 grid((kmax + block.x - 1) / block.x,
            (jmax + block.y - 1) / block.y,
            (imax + block.z - 1) / block.z);
  for(n=0 ; n<nn ; ++n){
#ifndef BINDLESS
    cudaBindSurfaceToArray(t_src, a_p);
    cudaBindSurfaceToArray(t_dst, a_wrk2);
#endif
    if (n + 1 == nn) {
      cudaMemset(d_gosa, 0, sizeof(float));
      jacobi_kernel0<true><<<grid, block>>>(
#ifdef BINDLESS
                                            t_wrk2, t_p,
#endif
                                            d_a, d_pitch, omega, d_gosa);
      cudaMemcpy(&gosa, d_gosa, sizeof(float), cudaMemcpyDeviceToHost);
    } else {
      jacobi_kernel0<false><<<grid, block>>>(
#ifdef BINDLESS
                                            t_wrk2, t_p,
#endif
                                             d_a, d_pitch, omega, 0);
    }

#ifndef BINDLESS
    cudaBindSurfaceToArray(t_src, a_wrk2);
    cudaBindSurfaceToArray(t_dst, a_p);
#endif
    jacobi_kernel1<<<grid, block>>>(
#ifdef BINDLESS
                                            t_p, t_wrk2
#endif
                                    );
  } /* end n loop */
  cudaDeviceSynchronize();

  return(gosa);
}

double
fflop(int mx,int my, int mz)
{
  return((double)(mz-2)*(double)(my-2)*(double)(mx-2)*34.0);
}

double
mflops(int nn,double cpu,double flop)
{
  return(flop/cpu*1.e-6*(double)nn);
}

#include <sys/time.h>
double
second()
{

  struct timeval tm;
  double t ;

  static int base_sec = 0,base_usec = 0;

  gettimeofday(&tm, NULL);
  
  if(base_sec == 0 && base_usec == 0)
    {
      base_sec = tm.tv_sec;
      base_usec = tm.tv_usec;
      t = 0.0;
  } else {
    t = (double) (tm.tv_sec-base_sec) + 
      ((double) (tm.tv_usec-base_usec))/1.0e6 ;
  }

  return t ;
}
