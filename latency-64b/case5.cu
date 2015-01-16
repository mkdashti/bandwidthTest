#include <stdint.h>
#include <unistd.h>
#include <stdio.h>
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


   int
main( int argc, char *argv[] )
{
   int ITERATIONS = 1;
   //int numBytes = 131072;
   int numBytes = 131072*2;

   uint64_t *memory_to_access;
   //HANDLE_ERROR(cudaHostAlloc(&memory_to_access,sizeof(uint64_t)*numBytes,0));
   HANDLE_ERROR(cudaMallocManaged(&memory_to_access,sizeof(uint64_t)*numBytes));
   for(int k=0;k< numBytes ;k++)
      memory_to_access[k]=5;

   //printf("address = %p\n",memory_to_access);
   //printf("Press enter to continue...\n");
   //getchar();
  
   
   uint64_t fake=0;
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
   //printf("Press enter to continue...\n");
   //getchar();
    
   //cudaFreeHost(memory_to_access);
   cudaFree(memory_to_access);

   return 0;
}
