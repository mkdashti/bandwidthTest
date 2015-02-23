#include <stdint.h>
#include <unistd.h>
#include <stdio.h>
#include <assert.h>
#include <sys/time.h>
#include <time.h>
#include <stdlib.h>
#include <sys/mman.h>

#include <sys/syscall.h>
#define gpu_hook(x) syscall(380,x)

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
   char pad[24];
}theblob;

__global__ void kernel(theblob *in, theblob *out, int threads)
{
   //int tid = threadIdx.x + blockIdx.x*blockDim.x;
   //if(tid < threads){
      //printf("hostalloc out addr =%p, in addr =%p\n",&(out[tid].data), &(in[tid].data));
      //printf("hostalloc out addr =%p, in addr =%p\n",&(out[0].data), &(in[0].data));
      //out[tid].data = in[tid].data+tid;
      //int temp = in[tid].data+tid;
      //if(temp == 999999)
      //   out[tid].data = 5;
   out[0].data += 5;

   //}
}
__global__ void kernel_d(theblob *in, theblob *out, int threads)
{
   //int tid = threadIdx.x + blockIdx.x*blockDim.x;
   //if(tid < threads){
      //printf("managed out addr =%p, in addr =%p\n",&(out[tid].data), &(in[tid].data));
      //printf("managed  out addr =%p, in addr =%p\n",&(out[0].data), &(in[0].data));
      //out[tid].data = in[tid].data+tid;
      //int temp = in[tid].data+tid;
      //if(temp == 999999)
      //   out[tid].data = 5;
   out[0].data += 5;
   //}
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

    gpu_hook(1);

    //printf("done with init...\n");
    //getchar();


    
    cudaHostAlloc(&in,blocks*threads*sizeof(theblob),0);
    cudaHostAlloc(&out,blocks*threads*sizeof(theblob),0);


    //printf("done with init and memory allocations...\n");
    //getchar();


    cudaMallocManaged(&in_d,blocks*threads*sizeof(theblob));
    cudaMallocManaged(&out_d,blocks*threads*sizeof(theblob));

   
    //printf("done with init and memory allocations...\n");
    //getchar();

    for(int i = 0; i<iterations; i++) {
       
       kernel<<<blocks,threads>>>(in,out,blocks*threads);
       cudaDeviceSynchronize();

      // printf("done with hostalloc kernel...\n");
      // getchar();


       kernel_d<<<blocks,threads>>>(in_d,out_d,blocks*threads);
       cudaDeviceSynchronize();

      // printf("done with managed kernel...\n");
      // getchar();


    }
    
   // printf("Press enter to continue...\n");
   // getchar();


    cpu_compute(in,out,blocks*threads);

    cudaFreeHost(in);
    cudaFreeHost(out);
    
    cudaFree(in_d);
    cudaFree(out_d);

    cudaDeviceReset();
    return 0;
}
