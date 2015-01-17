#include <stdint.h>
#include <unistd.h>
#include <stdio.h>
#include <assert.h>
#include <sys/time.h>
#include <time.h>
#include <stdlib.h>
#include <sys/mman.h>
static void HandleError( cudaError_t err, const char *file, int line ) {
    
    if (err != cudaSuccess) {
        
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ), file, line );
        exit( EXIT_FAILURE );
    
    }

}

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

#define HANDLE_NULL( a ) {if (a == NULL) { \
                            printf( "Host memory failed in %s at line %d\n", \
                                    __FILE__, __LINE__ ); \
                            exit( EXIT_FAILURE );}}

inline double diff_s(struct timeval start, struct timeval end)
{
   return ((double) (end.tv_usec - start.tv_usec) / 1000000 + (double) (end.tv_sec - start.tv_sec));
}

__global__ void kernel(uint64_t *in, uint64_t *out)
{
   int tid = threadIdx.x + blockIdx.x*blockDim.x;
   out[tid] = in[tid]+tid;
}
__global__ void nullKernel(int *memory)
{

}
void verify(uint64_t *in, uint64_t *out, int numBytes)
{
   int error = 0;
   for(int i =0; i<numBytes; i++)
      if(out[i]!=in[i]+i)
         error = 1;
   if(error)
      printf("ERROR in verification!\n");
   else
      printf("SUCCESS!\n");

}
void cpu_compute(uint64_t *in, uint64_t *out, int numBytes)
{
   for(int i =0; i<numBytes; i++)
      out[i]=in[i]+i;

}

int main( int argc, char *argv[] )
{
    uint64_t *in, *out, *in_d, *out_d;
    int ITERATIONS = 1;
    int numBytes = 1024*1024;
    struct timeval  tv1, tv2;
    int opt;
    int benchmarkType = 0;

    while ((opt = getopt(argc, argv, "m:b:i:")) != -1) {
       switch (opt) {
          case 'm':
             numBytes = atoi(optarg);
             //assert(numBytes%16 == 0 && numBytes<=1024);
             break;
          case 'b':
             benchmarkType = atoi(optarg);
             break;
          case 'i':
             ITERATIONS = atoi(optarg);
             break;

          default: /* '?' */
             break;
       }
    }


    int num_of_blocks=1;
    int num_of_threads_per_block=numBytes;
    if(numBytes>1024){
       num_of_blocks = 1024;
       num_of_threads_per_block = numBytes/1024;
    }

    HANDLE_ERROR(cudaFree(0));
    switch (benchmarkType) {
       case 0: {// default with data copy
                  in = (uint64_t *)malloc(sizeof(uint64_t)*numBytes);
                  out = (uint64_t *)malloc(sizeof(uint64_t)*numBytes);
                  assert(in);
                  assert(out);
                  HANDLE_ERROR( cudaMalloc( &in_d, sizeof(uint64_t)*numBytes) );
                  HANDLE_ERROR( cudaMalloc( &out_d, sizeof(uint64_t)*numBytes) );
                  for(int k=0;k< numBytes ;k++){
                     in[k]=1;
                  }

                  gettimeofday(&tv1, NULL);
                  for(int i=0; i<ITERATIONS; i++) {
                     cpu_compute(in, out, numBytes);
                     HANDLE_ERROR( cudaMemcpy(in_d,in, sizeof(uint64_t)*numBytes,cudaMemcpyDefault) );
                     kernel<<<num_of_blocks,num_of_threads_per_block>>>(in_d,out_d);
                     HANDLE_ERROR( cudaMemcpy(out,out_d, sizeof(uint64_t)*numBytes,cudaMemcpyDefault) );
                  }
                  verify(in,out,numBytes);
                  gettimeofday(&tv2, NULL);

                  HANDLE_ERROR( cudaGetLastError());
                  double elapsedTimeSeconds = diff_s(tv1,tv2);
                  printf("Default (including copy overhead) = %f ms\n",elapsedTimeSeconds*1e3/(float)ITERATIONS);

                  free(in);
                  free(out);
                  cudaFree(in_d);
                  cudaFree(out_d);
                  break;
               }
       case 1: {// cudaHostAlloc
                  HANDLE_ERROR( cudaHostAlloc( &in, sizeof(uint64_t)*numBytes,0) );
                  HANDLE_ERROR( cudaHostAlloc( &out, sizeof(uint64_t)*numBytes,0) );
                  for(int k=0;k< numBytes ;k++){
                     in[k]=1;
                  }

                  gettimeofday(&tv1, NULL);
                  for(int i=0; i<ITERATIONS; i++) {
                     cpu_compute(in, out, numBytes);
                     kernel<<<num_of_blocks,num_of_threads_per_block>>>(in,out);
                  //   HANDLE_ERROR(cudaDeviceSynchronize());
                  }
                  verify(in,out,numBytes);
                  gettimeofday(&tv2, NULL);

                  HANDLE_ERROR( cudaGetLastError());
                  double elapsedTimeSeconds = diff_s(tv1,tv2);
                  printf("cudaHostAlloc = %f ms\n",elapsedTimeSeconds*1e3/(float)ITERATIONS);

                  cudaFreeHost(in);
                  cudaFreeHost(out);
                  break;
               }
       case 2: {// cudaMallocManaged
                  HANDLE_ERROR( cudaMallocManaged( &in, sizeof(uint64_t)*numBytes) );
                  HANDLE_ERROR( cudaMallocManaged( &out, sizeof(uint64_t)*numBytes) );
                  for(int k=0;k< numBytes ;k++){
                     in[k]=1;
                  }
                  gettimeofday(&tv1, NULL);
                  for(int i=0; i<ITERATIONS; i++) {
                     cpu_compute(in, out, numBytes);
                     kernel<<<num_of_blocks,num_of_threads_per_block>>>(in,out);
                     HANDLE_ERROR(cudaDeviceSynchronize());
                  }
                  verify(in,out,numBytes);
                  gettimeofday(&tv2, NULL);

                  HANDLE_ERROR( cudaGetLastError());
                  double elapsedTimeSeconds = diff_s(tv1,tv2);
                  printf("cudaMallocManaged = %f ms\n",elapsedTimeSeconds*1e3/(float)ITERATIONS);

                  cudaFree(in);
                  cudaFree(out);
                  break;
               }
       case 3: {// ideal (discarding overhead of data copy)
                  in = (uint64_t *)malloc(sizeof(uint64_t)*numBytes);
                  out = (uint64_t *)malloc(sizeof(uint64_t)*numBytes);
                  assert(in);
                  assert(out);
                  for(int k=0;k< numBytes ;k++){
                     in[k]=1;
                  }

                  HANDLE_ERROR( cudaMalloc( &in_d, sizeof(uint64_t)*numBytes) );
                  HANDLE_ERROR( cudaMalloc( &out_d, sizeof(uint64_t)*numBytes) );
                  HANDLE_ERROR( cudaMemcpy(in_d,in, sizeof(uint64_t)*numBytes,cudaMemcpyDefault) );

                  gettimeofday(&tv1, NULL);
                  for(int i=0; i<ITERATIONS; i++) {
                     cpu_compute(in, out, numBytes);
                     kernel<<<num_of_blocks,num_of_threads_per_block>>>(in_d,out_d);
                  }
                  gettimeofday(&tv2, NULL);
                  double temp_elapsedTimeSeconds = diff_s(tv1,tv2);
                  HANDLE_ERROR( cudaMemcpy(out,out_d, sizeof(uint64_t)*numBytes,cudaMemcpyDefault) );

                  gettimeofday(&tv1, NULL);
                  verify(in,out,numBytes);
                  gettimeofday(&tv2, NULL);

                  HANDLE_ERROR( cudaGetLastError());
                  double elapsedTimeSeconds = temp_elapsedTimeSeconds + diff_s(tv1,tv2);
                  printf("Ideal (excluding copy overhead) = %f ms\n",elapsedTimeSeconds*1e3/(float)ITERATIONS);

                  free(in);
                  free(out);
                  cudaFree(in_d);
                  cudaFree(out_d);
                  break;
               }

    }

    cudaDeviceReset();
    return 0;
}
