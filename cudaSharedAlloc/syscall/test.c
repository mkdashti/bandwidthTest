#define _GNU_SOURCE
#include <unistd.h>
#include <sys/syscall.h>
#include <stdio.h>

#define gpu_hook() syscall(380)

int main (int argc, char* argv[])
{
   int j;
   printf("invocing kernel function\n");
   //j=syscall(380);
   j=gpu_hook();
   printf("invoked. Return is %d. Bye.\n", j);

   return 0;
}
