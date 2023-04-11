#include "utility.cuh"

int get_ncore_pSM(cudaDeviceProp& devProp) {
  int cores = 0;
  switch (devProp.major) {
    case 2:  // Fermi
      if (devProp.minor == 1)
        cores = 48;
      else
        cores = 32;
      break;
    case 3:  // Kepler
      cores = 192;
      break;
    case 5:  // Maxwell
      cores = 128;
      break;
    case 6:  // Pascal
      if ((devProp.minor == 1) || (devProp.minor == 2))
        cores = 128;
      else if (devProp.minor == 0)
        cores = 64;
      else
        printf("Unknown device type\n");
      break;
    case 7:  // Volta and Turing
      if ((devProp.minor == 0) || (devProp.minor == 5))
        cores = 64;
      else
        printf("Unknown device type\n");
      break;
    case 8:  // Ampere
      if (devProp.minor == 0)
        cores = 64;
      else if (devProp.minor == 6)
        cores = 128;
      else if (devProp.minor == 9)
        cores = 128;  // ada lovelace
      else
        printf("Unknown device type\n");
      break;
    case 9:  // Hopper
      if (devProp.minor == 0)
        cores = 128;
      else
        printf("Unknown device type\n");
      break;
    default:
      printf("Unknown device type\n");
      break;
  }
  return cores;
}

double current_seconds(void) {
  struct timespec ts;
  if (clock_gettime(CLOCK_MONOTONIC_RAW, &ts) != 0) {
    printf("Error getting time.\n");
    exit(1);
  }
  return ((double)ts.tv_sec) + (((double)ts.tv_nsec) / 1e9);
}