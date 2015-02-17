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
   //in = (uint64_t *)malloc(sizeof(uint64_t));
   //out = (uint64_t *)malloc(sizeof(uint64_t));
   out[tid] = in[tid]+tid;
}
__global__ void nullKernel(void)
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
    int numBytes = 1;
    int opt;
    int benchmarkType = 0;

    while ((opt = getopt(argc, argv, "m:b:")) != -1) {
       switch (opt) {
          case 'm':
             numBytes = atoi(optarg);
             //assert(numBytes%16 == 0 && numBytes<=1024);
             break;
          case 'b':
             benchmarkType = atoi(optarg);
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
    //cudaHostAlloc(&in,numBytes*sizeof(uint64_t),0);
    //cudaHostAlloc(&out,numBytes*sizeof(uint64_t),0);

    //printf("hostalloc %p\n",in);
    //printf("hostalloc %p\n",out);
    
    //cudaMallocManaged(&in_d,numBytes*sizeof(uint64_t));
    //cudaMallocManaged(&out_d,numBytes*sizeof(uint64_t));

    //printf("managed %p\n",in_d);
    //printf("managed %p\n",out_d);


    printf("Press enter to continue...\n");
    getchar();

    cudaHostAlloc(&in,numBytes*sizeof(uint64_t),0);
    cudaHostAlloc(&out,numBytes*sizeof(uint64_t),0);
   
    printf("Press enter to continue...\n");
    getchar();

    cudaMallocManaged(&in_d,numBytes*sizeof(uint64_t));
    cudaMallocManaged(&out_d,numBytes*sizeof(uint64_t));

    printf("Press enter to continue...\n");
    getchar();

    kernel<<<num_of_blocks,num_of_threads_per_block>>>(in,out);

    printf("Press enter to continue...\n");
    getchar();

    cudaDeviceSynchronize();
    cpu_compute(in,out,numBytes);

    printf("Press enter to continue...\n");
    getchar();
    cudaFreeHost(in);
    cudaFreeHost(out);
    
    printf("Press enter to continue...\n");
    getchar();
    cudaFree(in_d);
    cudaFree(out_d);

    printf("Press enter to continue...\n");
    getchar();
   
    return 0;
}
