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

typedef struct blob {
   uint64_t data;
}theblob;

__global__ void kernel(theblob *in, theblob *out, int threads)
{
   int tid = threadIdx.x + blockIdx.x*blockDim.x;
   //printf("OUTSIDE hostalloc tid=%d\n",tid);
   //in = (uint64_t *)malloc(sizeof(uint64_t));
   //out = (uint64_t *)malloc(sizeof(uint64_t));
   if(tid < threads){
     // printf("hostalloc tid=%d\n",tid);
      out[tid].data = in[tid].data+tid;
   }
}
__global__ void kernel_d(theblob *in, theblob *out, int threads)
{
   int tid = threadIdx.x + blockIdx.x*blockDim.x;
   //printf("OUTSIDE managed tid=%d\n",tid);
   //in = (uint64_t *)malloc(sizeof(uint64_t));
   //out = (uint64_t *)malloc(sizeof(uint64_t));
   if(tid < threads){
     // printf("managed tid=%d\n",tid);
      out[tid].data = in[tid].data+tid;
   }
}


__global__ void nullKernel(void)
{

}
void verify(theblob *in, theblob *out, int numBytes)
{
   int error = 0;
   for(int i =0; i<numBytes; i++)
      if(out[i].data!=in[i].data+i)
         error = 1;
   if(error)
      printf("ERROR in verification!\n");
   else
      printf("SUCCESS!\n");

}
void cpu_compute(theblob *in, theblob *out, int numBytes)
{
   for(int i =0; i<numBytes; i++)
      out[i].data=in[i].data+i;

}
int main( int argc, char *argv[] )
{
    theblob *in, *out, *in_d, *out_d;
    int opt;
    int iterations = 1;
    int blocks = 1;
    int threads = 1;

    while ((opt = getopt(argc, argv, "b:t:i:")) != -1) {
       switch (opt) {
          case 'b':
             blocks = atoi(optarg);
             //assert(numBytes%16 == 0 && numBytes<=1024);
             break;
          case 'i':
             iterations = atoi(optarg);
             break;
          case 't':
             threads = atoi(optarg);
             break;
          default: /* '?' */
             break;
       }
    }


    HANDLE_ERROR(cudaFree(0));
    //cudaHostAlloc(&in,numBytes*sizeof(uint64_t),0);
    //cudaHostAlloc(&out,numBytes*sizeof(uint64_t),0);

    //printf("hostalloc %p\n",in);
    //printf("hostalloc %p\n",out);
    
    //cudaMallocManaged(&in_d,numBytes*sizeof(uint64_t));
    //cudaMallocManaged(&out_d,numBytes*sizeof(uint64_t));

    //printf("managed %p\n",in_d);
    //printf("managed %p\n",out_d);


    //printf("Press enter to continue...\n");
    //getchar();

    cudaHostAlloc(&in,blocks*threads*sizeof(theblob),0);
    cudaHostAlloc(&out,blocks*threads*sizeof(theblob),0);
   
    //printf("Press enter to continue...\n");
    //getchar();

    cudaMallocManaged(&in_d,blocks*threads*sizeof(theblob));
    cudaMallocManaged(&out_d,blocks*threads*sizeof(theblob));

    //printf("Press enter to continue...\n");
    //getchar();

    for(int i = 0; i<iterations; i++) {
       kernel<<<blocks,threads>>>(in,out,blocks*threads);
       cudaDeviceSynchronize();
       kernel_d<<<blocks,threads>>>(in_d,out_d,blocks*threads);
       cudaDeviceSynchronize();
    }

    cpu_compute(in,out,blocks*threads);

    //printf("Press enter to continue...\n");
    //getchar();
    cudaFreeHost(in);
    cudaFreeHost(out);
    
    //printf("Press enter to continue...\n");
    //getchar();
    cudaFree(in_d);
    cudaFree(out_d);

    //printf("Press enter to continue...\n");
    //getchar();
   
    return 0;
}
