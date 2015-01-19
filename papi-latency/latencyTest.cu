#include <stdint.h>
#include <unistd.h>
#include <stdio.h>
#include <assert.h>
#include <sys/time.h>
#include <time.h>
#include <stdlib.h>
#include <sys/mman.h>

#include <papi.h>

#define NUM_EVENTS 8
#define PAPI


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
    int read=1; //read benchmark? or write?
    int benchmarkType = 0;
    int locked = 0; //mlock data?
    int dryRun = 0; //dry run to measure noise TLB misses/...etc

    while ((opt = getopt(argc, argv, "m:b:i:r:ld")) != -1) {
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
          case 'l':
             locked = 1;
             break;
          case 'd':
             dryRun = 1;
             break;



          default: /* '?' */
             break;
       }
    }


    int num_of_blocks=1;
    int num_of_threads_per_block=numBytes;
    if(numBytes>1024){
       num_of_blocks = 16;
       num_of_threads_per_block = numBytes/16;
    }

    if(benchmarkType == 0 || benchmarkType == 1)
       HANDLE_ERROR(cudaFree(0));

#ifdef PAPI
	int retval, i;
	int EventSet = PAPI_NULL;
	long long values[NUM_EVENTS];
   char *EventName[] = {"PERF_COUNT_HW_CPU_CYCLES",/*"PERF_COUNT_HW_INSTRUCTIONS",*/"INST_RETIRED", "L1I_TLB_REFILL","L1D_TLB_REFILL","L2D_CACHE_REFILL","L1D_INVALIDATE","PERF_COUNT_SW_PAGE_FAULTS_MIN","DATA_MEM_ACCESS" };
	int events[NUM_EVENTS];
	int eventCount = 0;
	
	/* PAPI Initialization */
	retval = PAPI_library_init( PAPI_VER_CURRENT );
	if( retval != PAPI_VER_CURRENT )
		fprintf( stderr, "PAPI_library_init failed\n" );
	
	printf( "PAPI_VERSION     : %4d %6d %7d\n",
			PAPI_VERSION_MAJOR( PAPI_VERSION ),
			PAPI_VERSION_MINOR( PAPI_VERSION ),
			PAPI_VERSION_REVISION( PAPI_VERSION ) );
	
	/* convert PAPI native events to PAPI code */
	for( i = 0; i < NUM_EVENTS; i++ ){
		retval = PAPI_event_name_to_code( EventName[i], &events[i] );
		if( retval != PAPI_OK ) {
			fprintf( stderr, "PAPI_event_name_to_code failed\n" );
			continue;
		}
		eventCount++;
		printf( "Name %s --- Code: %#x\n", EventName[i], events[i] );
	}

	/* if we did not find any valid events, just report test failed. */
	if (eventCount == 0) {
		printf( "Test FAILED: no valid events found.\n");
		return 1;
	}
	
	retval = PAPI_create_eventset( &EventSet );
	if( retval != PAPI_OK )
		fprintf( stderr, "PAPI_create_eventset failed\n" );
	
	retval = PAPI_add_events( EventSet, events, eventCount );
	if( retval != PAPI_OK )
		fprintf( stderr, "PAPI_add_events failed\n" );

#endif
	

    switch (benchmarkType) {
       case 0: {//read/Write to hostAlloc'd data
                  HANDLE_ERROR( cudaHostAlloc( &hostAllocd, sizeof(uint64_t)*numBytes, 0 ) );
                  for(int k=0;k< numBytes ;k++){
                     hostAllocd[k]=1;
                  }
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
                 
                 cudaFreeHost(hostAllocd);
                 break;
              }
   
      case 1: {//read/Write to cudaMalloc'd data
                 cpuMallocd = (uint64_t *)malloc(sizeof(uint64_t)*numBytes);
                 assert(cpuMallocd);
                 for(int k=0;k< numBytes ;k++){
                    cpuMallocd[k]=1;
                 }

                 HANDLE_ERROR( cudaMalloc( &cudaMallocd, sizeof(uint64_t)*numBytes) );
                 HANDLE_ERROR( cudaMemcpy( cudaMallocd,cpuMallocd, sizeof(uint64_t)*numBytes,cudaMemcpyDefault) );
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
                 free(cpuMallocd);
                 cudaFree(cudaMallocd);
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
                 if(!dryRun) { 
                    if(locked)
                       mlock(memory_to_access,sizeof(uint64_t)*numBytes);
                    assert(memory_to_access);
                    if(read)
                    {
                       for(int k=0;k< numBytes ;k++)
                          memory_to_access[k]=5;

                       uint64_t fake;
                       if(numBytes<8) {
                          gettimeofday(&tv1, NULL);
#ifdef PAPI
                          retval = PAPI_start( EventSet );
                          if( retval != PAPI_OK )
                             fprintf( stderr, "PAPI_start failed\n" );
#endif
 
                          for(int i=0; i<ITERATIONS; i++) {
                             for (int j = 0; j < (numBytes); j++) {
                                fake += memory_to_access[j];
                             }
                          }
#ifdef PAPI
                          retval = PAPI_stop( EventSet, values );
                          if( retval != PAPI_OK )
                             fprintf( stderr, "PAPI_stop failed\n" );
#endif
 
                          gettimeofday(&tv2, NULL);
                       }
                       else {
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
                    }
                    else
                    {
                       uint64_t fake=5;
                       if(numBytes<8) {
                          gettimeofday(&tv1, NULL);
                          for(int i=0; i<ITERATIONS; i++) {
                             for (int j = 0; j < (numBytes); j++) {
                                memory_to_access[j] = fake;
                             }
                          }
                          gettimeofday(&tv2, NULL);
                       }
                       else {
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
                    }
                    double elapsedTimeSeconds = diff_s(tv1,tv2);
                    printf("cpu malloc [%s] Latency = %f us\n",(read==1)?"read":"write",elapsedTimeSeconds*1e6/(float)ITERATIONS);
                 }
          //       printf("Press enter to continue...\n");
          //       getchar();
                 free(memory_to_access);
                 break;
              }
      case 4: {//read/Write to cpu but hostsllocd data
                 uint64_t *memory_to_access;
                 HANDLE_ERROR(cudaHostAlloc(&memory_to_access,sizeof(uint64_t)*numBytes,0));
                 if(!dryRun) {
                    if(read)
                    {
                       for(int k=0;k< numBytes ;k++)
                          memory_to_access[k]=5;

                       uint64_t fake;
                       if(numBytes<8) {
                          gettimeofday(&tv1, NULL);
#ifdef PAPI
                          retval = PAPI_start( EventSet );
                          if( retval != PAPI_OK )
                             fprintf( stderr, "PAPI_start failed\n" );
#endif
                          for(int i=0; i<ITERATIONS; i++) {
                             for (int j = 0; j < (numBytes); j++) {
                                fake += memory_to_access[j];
                             }
                          }
#ifdef PAPI
                          retval = PAPI_stop( EventSet, values );
                          if( retval != PAPI_OK )
                             fprintf( stderr, "PAPI_stop failed\n" );
#endif
                          gettimeofday(&tv2, NULL);
                       }
                       else {
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
                    }
                    else
                    {
                       uint64_t fake=5;
                       if(numBytes<8) {
                          gettimeofday(&tv1, NULL);
                          for(int i=0; i<ITERATIONS; i++) {
                             for (int j = 0; j < (numBytes); j++) {
                                memory_to_access[j] = fake;
                             }
                          }
                          gettimeofday(&tv2, NULL);
                       }

                       else {
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
                    }
                    double elapsedTimeSeconds = diff_s(tv1,tv2);
                    printf("cpu hostAlloc [%s] Latency = %f us\n",(read==1)?"read":"write",elapsedTimeSeconds*1e6/(float)ITERATIONS);
                 }
            //     printf("Press enter to continue...\n");
            //     getchar();
                 cudaFreeHost(memory_to_access);
                 break;
              }
      case 5: {//read/Write to cpu but mallocManaged data
                 uint64_t *memory_to_access;
                 HANDLE_ERROR(cudaMallocManaged(&memory_to_access,sizeof(uint64_t)*numBytes));
                 if(!dryRun) {
                    if(read)
                    {
                       for(int k=0;k< numBytes ;k++)
                          memory_to_access[k]=5;

                       uint64_t fake;
                       if(numBytes<8) {
                          gettimeofday(&tv1, NULL);
#ifdef PAPI
                          retval = PAPI_start( EventSet );
                          if( retval != PAPI_OK )
                             fprintf( stderr, "PAPI_start failed\n" );
#endif
                          for(int i=0; i<ITERATIONS; i++) {
                             for (int j = 0; j < (numBytes); j++) {
                                fake += memory_to_access[j];
                             }
                          }
#ifdef PAPI
                          retval = PAPI_stop( EventSet, values );
                          if( retval != PAPI_OK )
                             fprintf( stderr, "PAPI_stop failed\n" );
#endif
                          gettimeofday(&tv2, NULL);
                       }
                       else {
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
                    }
                    else
                    {
                       uint64_t fake=5;
                       if(numBytes<8) {
                          gettimeofday(&tv1, NULL);
                          for(int i=0; i<ITERATIONS; i++) {
                             for (int j = 0; j < (numBytes); j++) {
                                memory_to_access[j] = fake;
                             }
                          }
                          gettimeofday(&tv2, NULL);
                       }

                       else {
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
                    }
                    double elapsedTimeSeconds = diff_s(tv1,tv2);
                    printf("cpu mallocManaged [%s] Latency = %f us\n",(read==1)?"read":"write",elapsedTimeSeconds*1e6/(float)ITERATIONS);
                 }
            //     printf("Press enter to continue...\n");
            //     getchar();
                 cudaFree(memory_to_access);
                 break;
              }


    }

    if(benchmarkType == 0 || benchmarkType == 1)
       cudaDeviceReset();

#ifdef PAPI
	retval = PAPI_cleanup_eventset(EventSet);
	if( retval != PAPI_OK )
		fprintf(stderr, "PAPI_cleanup_eventset failed\n");

	retval = PAPI_destroy_eventset(&EventSet);
	if (retval != PAPI_OK)
		fprintf(stderr, "PAPI_destroy_eventset failed\n");

	PAPI_shutdown();

	for( i = 0; i < eventCount; i++ )
		printf( "%12lld \t\t --> %s \n", values[i], EventName[i] );
#endif


    return 0;
}
