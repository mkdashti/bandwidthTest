#include <stdint.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>

   int
main( int argc, char *argv[] )
{
   int ITERATIONS = 100000;
   int numBytes = 1024;

   uint64_t *memory_to_access = (uint64_t *)malloc(sizeof(uint64_t)*numBytes );
   for(int k=0;k< numBytes ;k++)
      memory_to_access[k]=5;

   printf("Press enter to continue...\n");
   getchar();
             
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
   printf("Press enter to continue...\n");
   getchar();
    
   free(memory_to_access);

   return 0;
}
