#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <sched.h>
#include <sys/sysinfo.h>

#include <sys/time.h>
#include <time.h>
#include <unistd.h>

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

__device__ void busywait(int cycles)
{
   clock_t current_time;
   current_time = clock64();
   int until = current_time + cycles;
   while (until > current_time) {
      current_time = clock64();
   }
}

__global__ void copyKernel(float *output, float *input, int N) { 
   int tid = threadIdx.x + blockIdx.x*blockDim.x;
   if(tid < N)
      output[tid] = input[tid];

} 
__global__ void nullKernel(float *output, float *input, int N) { 

} 
__global__ void offsetCopy(float *odata, float* idata, int offset) { 
   int xid = blockIdx.x * blockDim.x + threadIdx.x + offset; 
   odata[xid] = idata[xid]; }




int main(int argc, char *argv[]) 
{
   struct timeval  tv1, tv2;

   int opt;

   int num_of_blocks = 1024;
	int num_of_threads_per_block = 1024;
   int memSize = 4*1024*1024;
   int benchmarkType = 0;
   int ITERATIONS = 10;
   int t = 512;
 
   while ((opt = getopt(argc, argv, "m:b:i:t:")) != -1) {
      switch (opt) {
         case 'm':
            memSize = atoi(optarg)*1024*1024;
            break;
         case 'b':
            benchmarkType = atoi(optarg);
            break;
         case 'i':
            ITERATIONS = atoi(optarg);
            break;
         case 't':
            t = atoi(optarg);
            break;
 
         default: /* '?' */
            break;
      }
   }
   num_of_blocks = memSize/t;
   num_of_threads_per_block = t;
   assert(num_of_blocks <= 2147483647);
   assert(num_of_threads_per_block <= 1024);
   int N = num_of_blocks * num_of_threads_per_block;
   HANDLE_ERROR(cudaDeviceReset());
   cudaFree(0); //set context so that overhead won't be later accounted

   float *cpuMemory,*inputcudamallocMemory,*outputcudamallocMemory;
   cpuMemory = (float *)malloc(memSize*sizeof(float)*32);
   assert(cpuMemory);
   for(int i = 0; i < memSize*32/sizeof(float); i++)
      cpuMemory[i] = (float)(i & 0xff);

   HANDLE_ERROR( cudaMalloc( (void**)& inputcudamallocMemory, sizeof(float)*memSize*32) );
   HANDLE_ERROR( cudaMalloc( (void**)& outputcudamallocMemory, sizeof(float)*memSize*32) );
   HANDLE_ERROR( cudaMemcpy(inputcudamallocMemory,cpuMemory, sizeof(float)*memSize*32,cudaMemcpyDefault) );

  
   switch (benchmarkType) {
      case 0: {//Device to Device memcpy test 

                 gettimeofday(&tv1, NULL);
                 for(int i = 0; i < ITERATIONS; i++) {
                    HANDLE_ERROR( cudaMemcpy(outputcudamallocMemory,inputcudamallocMemory, sizeof(float)*memSize,cudaMemcpyDefault) );
                 }
                 HANDLE_ERROR( cudaDeviceSynchronize());
                 gettimeofday(&tv2, NULL);
                 double elapsedTimeSeconds = diff_s(tv1,tv2);
                 printf("elapsedTime per iteration = %f\n",elapsedTimeSeconds/ITERATIONS);
                 //we multiply by two since the DeviceToDevice copy involves both reading and writing to device memory
                 float bandwidth = 2.0f * ((double)memSize/(1024*1024*1024))*ITERATIONS/elapsedTimeSeconds;
                 //float bandwidth =  2.0f * ((float)(1<<10) * memSize * (float)ITERATIONS) / (elapsedTimeSeconds *(1000.0) * (float)(1 << 20));

                 printf("DeviceToDevice Memcpy Bandwitdh = %f GB/s\n",bandwidth);
                 break;
              }
      case 1: {//custom kernel with cuda malloced memory
                 gettimeofday(&tv1, NULL);
                 for(int i = 0; i < ITERATIONS; i++) {
                    nullKernel<<<num_of_blocks,num_of_threads_per_block>>>(outputcudamallocMemory,inputcudamallocMemory,N);
                 }
                 HANDLE_ERROR( cudaDeviceSynchronize());
                 gettimeofday(&tv2, NULL);
                 HANDLE_ERROR( cudaGetLastError());
                 double nullElapsedTime = diff_s(tv1,tv2);


                 for (int offset=0;offset<=32;offset++)
                 {
                 gettimeofday(&tv1, NULL);
                 for(int i = 0; i < ITERATIONS; i++) {
                    offsetCopy<<<num_of_blocks,num_of_threads_per_block>>>(outputcudamallocMemory,inputcudamallocMemory,offset);
                 }
                 HANDLE_ERROR( cudaDeviceSynchronize());
                 gettimeofday(&tv2, NULL);
                 HANDLE_ERROR( cudaGetLastError());
                 double elapsedTimeSeconds = diff_s(tv1,tv2);
                 printf("elapsedTime per iteration = %f\n",elapsedTimeSeconds/ITERATIONS);
                 //we multiply by two since the DeviceToDevice copy involves both reading and writing to device memory
                 float bandwidth = 2.0f * ((double)memSize/(1024*1024*1024))*ITERATIONS/elapsedTimeSeconds;
                 float bandwidth_ex = 2.0f * ((double)memSize/(1024*1024*1024))*ITERATIONS/(elapsedTimeSeconds-nullElapsedTime);
                 //float bandwidth =  2.0f * ((float)(1<<10) * memSize * (float)ITERATIONS) / (elapsedTimeSeconds *(1000.0) * (float)(1 << 20));

                 printf("\noffset %d\n",offset);
                 printf("Custom kernel(cudaMalloc) memcpy Bandwitdh including kernel launch overhead = %f GB/s\n",bandwidth);
                 printf("Custom kernel(cudaMalloc) memcpy Bandwitdh excluding kernel launch overhead = %f GB/s\n",bandwidth_ex);
                 }
                 break;
              }

   }
   free(cpuMemory);
   cudaFree(inputcudamallocMemory);
   cudaFree(outputcudamallocMemory);

   return 0; 
}
