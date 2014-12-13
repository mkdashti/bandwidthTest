#include <stdint.h>
#include <unistd.h>
#include <stdio.h>
#include <assert.h>
#include <sys/time.h>
#include <time.h>
#include <stdlib.h>

inline double diff_s(struct timeval start, struct timeval end)
{
   return ((double) (end.tv_usec - start.tv_usec) / 1000000 + (double) (end.tv_sec - start.tv_sec));
}


int main(int argc, char *argv[])
{
   int numEntries=1, ITERATIONS=1000000;
   struct timeval  tv1, tv2;
   int opt;
   int read=0; //read benchmark? or write?

   printf("size of uint64_t = %d\n",sizeof(uint64_t));
   while ((opt = getopt(argc, argv, "m:i:r:")) != -1) {
      switch (opt) {
         case 'm':
            numEntries = atoi(optarg);
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

   uint64_t *memory_to_access = (uint64_t *)malloc(sizeof(uint64_t)*numEntries );
   assert(memory_to_access);
   int i,j;
   int entrySize = sizeof(uint64_t);
   if(read)
   {
      for(int k=0;k< numEntries;k++)
         memory_to_access[k]=5;

      uint64_t fake;
      gettimeofday(&tv1, NULL);
      for(i=0; i<ITERATIONS; i++) {
         for (j = 0; j < (numEntries); j += 8) {
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
      for(i=0; i<ITERATIONS; i++) {
         for (j = 0; j < (numEntries); j += 8) {
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
   //free(memory_to_access);
   return 0;
}
