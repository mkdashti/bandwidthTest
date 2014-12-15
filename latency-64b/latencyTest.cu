#include <stdint.h>
#include <unistd.h>
#include <stdio.h>
#include <assert.h>
#include <sys/time.h>
#include <time.h>
#include <stdlib.h>
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

__global__ void readKernel(uint64_t *memory, uint64_t *memoryToRead)
{
   int tid = threadIdx.x + blockIdx.x*blockDim.x;
   //memory[tid]=memoryToRead[tid];
   //__shared__ uint64_t temp; 
   uint64_t temp = memoryToRead[tid];
   if(!temp)
      __syncthreads();
}
__global__ void writeKernel(uint64_t *memory)
{
   int tid = threadIdx.x + blockIdx.x*blockDim.x;
   memory[tid]=5;
}
__global__ void nullKernel(int *memory)
{

}
__global__ void initCudaMallocd(uint64_t *memory, int N)
{
   int tid =threadIdx.x;
   if(tid==0){
      for(int k=0;k< N ;k++)
         memory[k]=5;
   }
}

void verify(uint64_t* memory, int N)
{
   int error = 0;
   for(int i =0; i<N; i++){
      if(memory[i]!=5){
         error = 1;
         break;
      }
   }
   if(error)
      printf("error in verification\n");
   else
      printf("verified SUCCESS\n");
}
__global__ void verifyCudaMallocd(uint64_t* memory, int N)
{
   int tid=threadIdx.x;
   if(tid==0) {
      int error = 0;
      for(int i =0; i<N; i++){
         if(memory[i]!=5){
            error = 1;
            break;
         }
      }
      if(error)
         printf("error in verification\n");
      else
         printf("verified SUCCESS\n");
   }
}


   int
main( int argc, char *argv[] )
{
    uint64_t *hostAllocd, *cudaMallocd, *cpuMallocd;
    int ITERATIONS = 100000;
    int numBytes = 1;
    struct timeval  tv1, tv2;
    int opt;
    int read=0; //read benchmark? or write?
    int benchmarkType = 0;

      while ((opt = getopt(argc, argv, "m:b:i:r:")) != -1) {
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
         case 'r':
            read = atoi(optarg);
            break;
 
         default: /* '?' */
            break;
      }
   }


   cpuMallocd = (uint64_t *)malloc(sizeof(uint64_t)*numBytes);
   assert(cpuMallocd);
   HANDLE_ERROR( cudaHostAlloc( &hostAllocd, sizeof(uint64_t)*numBytes, 0 ) );
   for(int k=0;k< numBytes ;k++){
      cpuMallocd[k]=1;
      hostAllocd[k]=1;
   }

   HANDLE_ERROR( cudaMalloc( &cudaMallocd, sizeof(uint64_t)*numBytes) );
   HANDLE_ERROR( cudaMemcpy( cudaMallocd,hostAllocd, sizeof(uint64_t)*numBytes,cudaMemcpyDefault) );

   int num_of_blocks=1;
   int num_of_threads_per_block=numBytes;
   if(numBytes>1024){
      num_of_blocks = 16;
      num_of_threads_per_block = numBytes/16;
   }

   //HANDLE_ERROR(cudaDeviceReset());  //this causes kernel launch failure!! check with cuda-memcheck
   HANDLE_ERROR(cudaFree(0));
   switch (benchmarkType) {
      case 0: {//read/Write to hostAlloc'd data
                 if(read)
                 {

                    uint64_t *memoryToRead;
                    HANDLE_ERROR( cudaHostAlloc( &memoryToRead, sizeof(uint64_t)*numBytes, 0 ) );
                    for(int k=0;k< numBytes ;k++)
                       memoryToRead[k]=5;
                    gettimeofday(&tv1, NULL);
                    for(int i = 0; i < ITERATIONS; i++) {
                       readKernel<<<num_of_blocks,num_of_threads_per_block>>>(hostAllocd,memoryToRead);
                       HANDLE_ERROR( cudaDeviceSynchronize());
                    }
                    gettimeofday(&tv2, NULL);
                    cudaFreeHost(memoryToRead);
                    //verify(hostAllocd,numBytes);
                 }
                 else
                 {
                    gettimeofday(&tv1, NULL);
                    for(int i = 0; i < ITERATIONS; i++) {
                       writeKernel<<<num_of_blocks,num_of_threads_per_block>>>(hostAllocd);
                       HANDLE_ERROR( cudaDeviceSynchronize());
                    }
                    gettimeofday(&tv2, NULL);
                    verify(hostAllocd,numBytes);
                 }
                 HANDLE_ERROR( cudaGetLastError());
                 double elapsedTimeSeconds = diff_s(tv1,tv2);
                 printf("HostAlloc [%s] Latency including kernel launch overhead = %f us\n",(read==1)?"read":"write",elapsedTimeSeconds*1e6/(float)ITERATIONS);
                 break;
              }
   
      case 1: {//read/Write to cudaMalloc'd data
                 if(read)
                 {

                    uint64_t *memoryToRead;
                    HANDLE_ERROR( cudaMalloc( &memoryToRead, sizeof(uint64_t)*numBytes ) );
                    initCudaMallocd<<<1,1>>>(memoryToRead,numBytes);
                    HANDLE_ERROR( cudaDeviceSynchronize());
                    gettimeofday(&tv1, NULL);
                    for(int i = 0; i < ITERATIONS; i++) {
                       readKernel<<<num_of_blocks,num_of_threads_per_block>>>(cudaMallocd,memoryToRead);
                       HANDLE_ERROR( cudaDeviceSynchronize());
                    }
                    gettimeofday(&tv2, NULL);
                    cudaFree(memoryToRead);
                    //verifyCudaMallocd<<<1,1>>>(cudaMallocd,numBytes);
                    //HANDLE_ERROR( cudaDeviceSynchronize());
                 }
                 else
                 {
                    gettimeofday(&tv1, NULL);
                    for(int i = 0; i < ITERATIONS; i++) {
                       writeKernel<<<num_of_blocks,num_of_threads_per_block>>>(cudaMallocd);
                       HANDLE_ERROR( cudaDeviceSynchronize());
                    }
                    gettimeofday(&tv2, NULL);
                    //verifyCudaMallocd<<<1,1>>>(cudaMallocd,numBytes);
                    //HANDLE_ERROR( cudaDeviceSynchronize());
                 }
                 HANDLE_ERROR( cudaGetLastError());
                 double elapsedTimeSeconds = diff_s(tv1,tv2);
                 printf("CudaMalloc [%s] Latency including kernel launch overhead = %f us\n",(read==1)?"read":"write",elapsedTimeSeconds*1e6/(float)ITERATIONS);
                 break;
              }

      case 2:
              {
                 gettimeofday(&tv1, NULL);
                 for(int i = 0; i < ITERATIONS; i++) {
                    nullKernel<<<num_of_blocks,num_of_threads_per_block>>>(0);
                    HANDLE_ERROR( cudaDeviceSynchronize());
                 }
                 gettimeofday(&tv2, NULL);
                 HANDLE_ERROR( cudaGetLastError());
                 double elapsedTimeSeconds = diff_s(tv1,tv2);
                 printf("null kernel launch overhead = %f us\n",elapsedTimeSeconds*1e6/(float)ITERATIONS);
                 break;
              
              }
      case 3: {//read/Write to cpu mallocd data
                 uint64_t *memory_to_access = (uint64_t *)malloc(sizeof(uint64_t)*numBytes );
                 assert(memory_to_access);
                 if(read)
                 {
                    for(int k=0;k< numBytes ;k++)
                       memory_to_access[k]=5;

                    uint64_t fake;
                    gettimeofday(&tv1, NULL);
                    for(int i=0; i<ITERATIONS; i++) {
                       for (int j = 0; j < (numBytes); j += 8) {
                          fake += memory_to_access[j];
                          fake += memory_to_access[j + 1];
                          fake += memory_to_access[j + 2];
                          fake += memory_to_access[j + 3];
                          fake += memory_to_access[j + 4];
                          fake += memory_to_access[j + 5];
                          fake += memory_to_access[j + 6];
                          fake += memory_to_access[j + 7];
                       }
                    }
                    gettimeofday(&tv2, NULL);
                 }
                 else
                 {
                    uint64_t fake=5;
                    gettimeofday(&tv1, NULL);
                    for(int i=0; i<ITERATIONS; i++) {
                       for (int j = 0; j < (numBytes); j += 8) {
                          memory_to_access[j] = fake;
                          memory_to_access[j + 1] = fake;
                          memory_to_access[j + 2] = fake;
                          memory_to_access[j + 3] = fake;
                          memory_to_access[j + 4] = fake;
                          memory_to_access[j + 5] = fake;
                          memory_to_access[j + 6] = fake;
                          memory_to_access[j + 7] = fake;
                       }
                    }
                    gettimeofday(&tv2, NULL);
                 }
                 double elapsedTimeSeconds = diff_s(tv1,tv2);
                 printf("cpu malloc [%s] Latency = %f us\n",(read==1)?"read":"write",elapsedTimeSeconds*1e6/(float)ITERATIONS);
                 free(memory_to_access);
                 break;
              }
      case 4: {//read/Write to cpu but hostsllocd data
                 uint64_t *memory_to_access;
                 HANDLE_ERROR(cudaHostAlloc(&memory_to_access,sizeof(uint64_t)*numBytes,0));
                 if(read)
                 {
                    for(int k=0;k< numBytes ;k++)
                       memory_to_access[k]=5;

                    uint64_t fake;
                    gettimeofday(&tv1, NULL);
                    for(int i=0; i<ITERATIONS; i++) {
                       for (int j = 0; j < (numBytes); j += 8) {
                          fake += memory_to_access[j];
                          fake += memory_to_access[j + 1];
                          fake += memory_to_access[j + 2];
                          fake += memory_to_access[j + 3];
                          fake += memory_to_access[j + 4];
                          fake += memory_to_access[j + 5];
                          fake += memory_to_access[j + 6];
                          fake += memory_to_access[j + 7];
                       }
                    }
                    gettimeofday(&tv2, NULL);
                 }
                 else
                 {
                    uint64_t fake=5;
                    gettimeofday(&tv1, NULL);
                    for(int i=0; i<ITERATIONS; i++) {
                       for (int j = 0; j < (numBytes); j += 8) {
                          memory_to_access[j] = fake;
                          memory_to_access[j + 1] = fake;
                          memory_to_access[j + 2] = fake;
                          memory_to_access[j + 3] = fake;
                          memory_to_access[j + 4] = fake;
                          memory_to_access[j + 5] = fake;
                          memory_to_access[j + 6] = fake;
                          memory_to_access[j + 7] = fake;
                       }
                    }
                    gettimeofday(&tv2, NULL);
                 }
                 double elapsedTimeSeconds = diff_s(tv1,tv2);
                 printf("cpu malloc [%s] Latency = %f us\n",(read==1)?"read":"write",elapsedTimeSeconds*1e6/(float)ITERATIONS);
                 cudaFreeHost(memory_to_access);
                 break;
              }
       
   }

   free(cpuMallocd);
   cudaFree(cudaMallocd);
   cudaFreeHost(hostAllocd);
   cudaDeviceReset();
   return 0;
}