#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <sched.h>
#include <sys/sysinfo.h>
#include <pthread.h>
#include <sched.h>
#include <sys/syscall.h> 

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

__global__ void copyKernel(unsigned char *output, unsigned char *input, int N) { 
   int tid = threadIdx.x + blockIdx.x*blockDim.x;
   if(tid < N)
      output[tid] = input[tid];

} 
__global__ void nullKernel(unsigned char *output, unsigned char *input, int N) { 

} 


static pid_t gettid(void) {                                                                      
   return syscall(__NR_gettid);
}

void set_affinity(int tid, int core_id) {
   cpu_set_t mask;
   CPU_ZERO(&mask);
   CPU_SET(core_id, &mask);

   int r = sched_setaffinity(tid, sizeof(mask), &mask);                                                                  
   if (r < 0) {
      fprintf(stderr, "couldn't set affinity for %d\n", core_id);
      exit(1);
   }
}


typedef struct {                                                                                                        
   int id;
   unsigned char *input;
   unsigned char *output;
   int memSize;
} parm;

void *work( void *arg ) {
   parm *p=(parm *)arg;
   int tid = gettid();                                                                                                   
   set_affinity(tid, (p->id)%get_nprocs());

   memcpy(p->output,p->input,p->memSize);
   return 0;
}
void launch_cpu_threads(int nthreads, unsigned char **out, unsigned char **in, int memSize)
{
   pthread_t *threads;
   pthread_attr_t attr;
   parm *p;
   int j;
   
   threads=(pthread_t *)malloc(nthreads * sizeof(pthread_t));
   if(threads == NULL) {
      printf("ERROR malloc failed to create CPU threads\n");
      exit(1);
   }
   pthread_attr_init(&attr);
   p=(parm *)malloc(nthreads * sizeof(parm));

   for (j=0; j<nthreads; j++)
   {
      p[j].id=j;
      p[j].input=in[j];
      p[j].output=out[j];
      p[j].memSize=memSize;
      if(pthread_create(&threads[j], &attr, work, (void *)(p+j))!=0)
      {
         printf("ERROR creating threads\n");
         exit(1);
      }
   }

   for (j=0; j<nthreads; j++)
   {
      if(pthread_join(threads[j],NULL)!=0) {
         printf("ERROR in joing threads\n");
         exit(1);
      }
   }

   pthread_attr_destroy(&attr);
   free(p);
}


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

   unsigned char *cpuMemory,*inputhostallocMemory,*outputhostallocMemory,*inputcudamallocMemory,*outputcudamallocMemory,
                 *inputmanagedMemory, *outputmanagedMemory;
   cpuMemory = (unsigned char *)malloc(memSize);
   assert(cpuMemory);
   for(int i = 0; i < memSize/sizeof(unsigned char); i++)
      cpuMemory[i] = (unsigned char)(i & 0xff);

   HANDLE_ERROR( cudaHostAlloc( (void**)& inputhostallocMemory, sizeof(unsigned char)*memSize, cudaHostAllocDefault) );
   HANDLE_ERROR( cudaHostAlloc( (void**)& outputhostallocMemory, sizeof(unsigned char)*memSize, cudaHostAllocDefault) );
   HANDLE_ERROR( cudaMalloc( (void**)& inputcudamallocMemory, sizeof(unsigned char)*memSize) );
   HANDLE_ERROR( cudaMalloc( (void**)& outputcudamallocMemory, sizeof(unsigned char)*memSize) );
   HANDLE_ERROR( cudaMemcpy(inputcudamallocMemory,cpuMemory, sizeof(unsigned char)*memSize,cudaMemcpyDefault) );
   HANDLE_ERROR( cudaMemcpy(inputhostallocMemory,cpuMemory, sizeof(unsigned char)*memSize,cudaMemcpyDefault) );

  
   switch (benchmarkType) {
      case 0: {//Device to Device memcpy test 

                 gettimeofday(&tv1, NULL);
                 for(int i = 0; i < ITERATIONS; i++) {
                    HANDLE_ERROR( cudaMemcpy(outputcudamallocMemory,inputcudamallocMemory, sizeof(unsigned char)*memSize,cudaMemcpyDefault) );
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

                 gettimeofday(&tv1, NULL);
                 for(int i = 0; i < ITERATIONS; i++) {
                    copyKernel<<<num_of_blocks,num_of_threads_per_block>>>(outputcudamallocMemory,inputcudamallocMemory,N);
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

                 printf("Custom kernel(cudaMalloc) memcpy Bandwitdh including kernel launch overhead = %f GB/s\n",bandwidth);
                 printf("Custom kernel(cudaMalloc) memcpy Bandwitdh excluding kernel launch overhead = %f GB/s\n",bandwidth_ex);
                 break;
              }
      case 2: {//Custom kernel with host allocated memory
                 gettimeofday(&tv1, NULL);
                 for(int i = 0; i < ITERATIONS; i++) {
                    nullKernel<<<num_of_blocks,num_of_threads_per_block>>>(outputhostallocMemory,inputhostallocMemory,N);
                 }
                 HANDLE_ERROR( cudaDeviceSynchronize());
                 gettimeofday(&tv2, NULL);
                 HANDLE_ERROR( cudaGetLastError());
                 double nullElapsedTime = diff_s(tv1,tv2);

 
                 gettimeofday(&tv1, NULL);
                 for(int i = 0; i < ITERATIONS; i++) {
                    copyKernel<<<num_of_blocks,num_of_threads_per_block>>>(outputhostallocMemory,inputhostallocMemory,N);
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

                 printf("Custom kernel(cudaHostAlloc) memcpy Bandwitdh = %f GB/s\n",bandwidth);
                 printf("Custom kernel(cudaHostAlloc) memcpy Bandwitdh excluding kernel launch overhead = %f GB/s\n",bandwidth_ex);
                 break;
              }
      case 3: {//host allocated memory copy test

                 gettimeofday(&tv1, NULL);
                 for(int i = 0; i < ITERATIONS; i++) {
                    HANDLE_ERROR( cudaMemcpyAsync(outputhostallocMemory,inputhostallocMemory, sizeof(unsigned char)*memSize,cudaMemcpyDefault) );
                 }
                 HANDLE_ERROR( cudaDeviceSynchronize());
                 gettimeofday(&tv2, NULL);
                 HANDLE_ERROR( cudaGetLastError());
                 double elapsedTimeSeconds = diff_s(tv1,tv2);
                 printf("elapsedTime per iteration = %f\n",elapsedTimeSeconds/ITERATIONS);
                 //we multiply by two since the DeviceToDevice copy involves both reading and writing to device memory
                 float bandwidth = 2.0f * ((double)memSize/(1024*1024*1024))*ITERATIONS/elapsedTimeSeconds;
                 //float bandwidth =  2.0f * ((float)(1<<10) * memSize * (float)ITERATIONS) / (elapsedTimeSeconds *(1000.0) * (float)(1 << 20));

                 printf("Device to Device cudaHostAlloc memcpy Bandwitdh = %f GB/s\n",bandwidth);
                 break;
              }

      case 4: {//managed memory copy test
                 HANDLE_ERROR( cudaMallocManaged( (void**)& inputmanagedMemory, sizeof(unsigned char)*memSize) );
                 HANDLE_ERROR( cudaMallocManaged( (void**)& outputmanagedMemory, sizeof(unsigned char)*memSize) );
                 HANDLE_ERROR( cudaMemcpy(inputmanagedMemory,cpuMemory, sizeof(unsigned char)*memSize,cudaMemcpyDefault) );
                 gettimeofday(&tv1, NULL);
                 for(int i = 0; i < ITERATIONS; i++) {
                    nullKernel<<<num_of_blocks,num_of_threads_per_block>>>(outputmanagedMemory,inputmanagedMemory,N);
                 }
                 HANDLE_ERROR( cudaDeviceSynchronize());
                 gettimeofday(&tv2, NULL);
                 HANDLE_ERROR( cudaGetLastError());
                 double nullElapsedTime = diff_s(tv1,tv2);

 

                 gettimeofday(&tv1, NULL);
                 for(int i = 0; i < ITERATIONS; i++) {
                    copyKernel<<<num_of_blocks,num_of_threads_per_block>>>(outputmanagedMemory,inputmanagedMemory,N);
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

                 printf("Custom kernel (managed memory)  Bandwitdh = %f GB/s\n",bandwidth);
                 printf("Custom kernel (managed memory)  Bandwitdh excluding kernel launch overhead = %f GB/s\n",bandwidth_ex);
                 cudaFree(inputmanagedMemory);
                 cudaFree(outputmanagedMemory);

                 break;
              }
      case 5: {//Custom kernel with host allocated to malloc copy
                 gettimeofday(&tv1, NULL);
                 for(int i = 0; i < ITERATIONS; i++) {
                    nullKernel<<<num_of_blocks,num_of_threads_per_block>>>(outputcudamallocMemory,inputhostallocMemory,N);
                 }
                 HANDLE_ERROR( cudaDeviceSynchronize());
                 gettimeofday(&tv2, NULL);
                 HANDLE_ERROR( cudaGetLastError());
                 double nullElapsedTime = diff_s(tv1,tv2);

 
                 gettimeofday(&tv1, NULL);
                 for(int i = 0; i < ITERATIONS; i++) {
                    copyKernel<<<num_of_blocks,num_of_threads_per_block>>>(outputcudamallocMemory,inputhostallocMemory,N);
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

                 printf("Custom kernel(cudaHostAlloc to cudaMalloc) memcpy Bandwitdh = %f GB/s\n",bandwidth);
                 printf("Custom kernel(cudaHostAlloc to cudaMalloc) memcpy Bandwitdh excluding kernel launch overhead = %f GB/s\n",bandwidth_ex);
                 break;
              }
      case 6: {//Cpu malloc to malloc
 
                 unsigned char *mallocdOut,*mallocdIn;
                 mallocdOut = (unsigned char *)malloc(sizeof(unsigned char)*memSize);
                 mallocdIn = (unsigned char *)malloc(sizeof(unsigned char)*memSize);
                 for(int i=0; i<memSize/sizeof(unsigned char); i++)
                    mallocdIn[i]=5;
                 if(!mallocdOut || !mallocdIn) {
                    printf("ERROR in malloc\n");
                    return -1;
                 }
                 for(int i=0; i<memSize/sizeof(unsigned char); i++)
                    mallocdIn[i]=5;
                
                 gettimeofday(&tv1, NULL);
                 for(int i = 0; i < ITERATIONS; i++) {
                    memcpy(mallocdOut,mallocdIn,sizeof(unsigned char)*memSize);
                 }
                 gettimeofday(&tv2, NULL);
                 double elapsedTimeSeconds = diff_s(tv1,tv2);
                 printf("elapsedTime per iteration = %f\n",elapsedTimeSeconds/ITERATIONS);
                 //we multiply by two since the DeviceToDevice copy involves both reading and writing to device memory
                 float bandwidth =  ((double)memSize/(1024*1024*1024))*ITERATIONS/elapsedTimeSeconds;

                 printf("cpu malloc to malloc memcpy Bandwitdh = %f GB/s\n",bandwidth);
                 free(mallocdOut);
                 free(mallocdIn);
                 break;
              }
       case 7: {//Cpu multithreaded malloc to malloc

                 const int nthreads=4; 
                 unsigned char *mallocdOut[nthreads],*mallocdIn[nthreads];
                 for(int i=0; i<nthreads; i++) {
                    mallocdOut[i] = (unsigned char *)malloc(sizeof(unsigned char)*memSize);
                    mallocdIn[i] = (unsigned char *)malloc(sizeof(unsigned char)*memSize);
                 }
                  if(!mallocdOut || !mallocdIn) {
                    printf("ERROR in malloc\n");
                    return -1;
                 }
                for(int i=0; i<memSize/sizeof(unsigned char); i++){
                    mallocdIn[0][i]=5;
                    mallocdIn[1][i]=5;
                    mallocdIn[2][i]=5;
                    mallocdIn[3][i]=5;
                }
                 gettimeofday(&tv1, NULL);
                 for(int i = 0; i < ITERATIONS; i++) {
                    launch_cpu_threads(nthreads,mallocdOut,mallocdIn,memSize);
                 }
                 gettimeofday(&tv2, NULL);
                 double elapsedTimeSeconds = diff_s(tv1,tv2);
                 printf("elapsedTime per iteration = %f\n",elapsedTimeSeconds/ITERATIONS);
                 //we multiply by two since the DeviceToDevice copy involves both reading and writing to device memory
                 float bandwidth = 4 * ((double)memSize/(1024*1024*1024))*ITERATIONS/elapsedTimeSeconds;
                 printf("cpu malloc to malloc memcpy Bandwitdh = %f GB/s\n",bandwidth);
                 for(int i=0; i<nthreads; i++){
                    free(mallocdOut[i]);
                    free(mallocdIn[i]);
                 }
                 break;
              }
 





   }
   free(cpuMemory);
   cudaFreeHost(inputhostallocMemory);
   cudaFreeHost(outputhostallocMemory);
   cudaFree(inputcudamallocMemory);
   cudaFree(outputcudamallocMemory);
   return 0; 
}
