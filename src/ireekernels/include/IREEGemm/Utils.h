#ifndef IREE_GEMM_UTILS
#define IREE_GEMM_UTILS

#include <stdio.h>

void print_progress(int current, int total, const char* message) {
  printf("\r\033[K");
  float percentage = ((float)current / total) * 100;
  printf("\r[%d/%d] (%.1f%%) %s", current, total, percentage, message);
  fflush(stdout);
}

#endif