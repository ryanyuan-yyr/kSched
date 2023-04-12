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

/**
 * TODO
 *
 * GPU Current Temp
 * GPU Shutdown Temp
 * GPU Slowdown Temp
 * GPU Max Operating Temp
 */
static void get_gpu_temp(double (*res)[4]) {
  FILE* nv_smi = popen("nvidia-smi -q -d TEMPERATURE", "r");
  constexpr unsigned BUFF_SZ = 128;
  char buff[BUFF_SZ];
  for (size_t i = 0; i < 10; i++) {
    if (!fgets(buff, BUFF_SZ, nv_smi)) {
      pclose(nv_smi);
      throw std::logic_error(
          "get_gpu_temp unexpected result from nvidia-smi: reading FILE* "
          "failed");
    }
  }
  const char* prefixes[] = {"        GPU Current Temp                  : ",
                            "        GPU Shutdown Temp                 : ",
                            "        GPU Slowdown Temp                 : ",
                            "        GPU Max Operating Temp            : "};
  const char* suffix = " C\n";
  for (size_t i = 0; i < 4; i++) {
    if (!fgets(buff, BUFF_SZ, nv_smi)) {
      pclose(nv_smi);
      throw std::logic_error(
          "get_gpu_temp unexpected result from nvidia-smi: reading FILE* "
          "failed");
    }
    std::string line{buff};
    if (strncmp(prefixes[i], line.c_str(), strlen(prefixes[i])) != 0) {
      pclose(nv_smi);
      printf("%lu: %s\n", i, line.c_str());
      throw std::logic_error(
          "get_gpu_temp unexpected result from nvidia-smi prefix");
    }
    if (strncmp(suffix, line.c_str() + line.size() - strlen(suffix),
                strlen(suffix)) != 0) {
      pclose(nv_smi);
      printf("%lu: %s\n", i, line.c_str());
      throw std::logic_error(
          "get_gpu_temp unexpected result from nvidia-smi suffix");
    }
    (*res)[i] = std::atof(
        line.substr(strlen(prefixes[i]),
                    line.size() - strlen(prefixes[i]) - strlen(suffix))
            .c_str());
  }
  pclose(nv_smi);
}

bool gpu_temp_in_range(unsigned safe_region) {
  // // Initialize NVML
  // CHECK_NVML(nvmlInit());

  // // Get the device count
  // unsigned int deviceCount;
  // CHECK_NVML(nvmlDeviceGetCount(&deviceCount));

  // // Get the temperature of each device
  // for (unsigned int i = 0; i < deviceCount; i++) {
  //   nvmlDevice_t device;
  //   CHECK_NVML(nvmlDeviceGetHandleByIndex(i, &device));

  //   unsigned int temperature;
  //   CHECK_NVML(
  //       nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU,
  //       &temperature));

  //   unsigned int slow_down_temp;
  //   CHECK_NVML(nvmlDeviceGetTemperatureThreshold(
  //       device, NVML_TEMPERATURE_THRESHOLD_SLOWDOWN, &slow_down_temp));
  //   if (temperature >= slow_down_temp + safe_region) {
  //     nvmlShutdown();
  //     return false;
  //   }

  //   unsigned int max_temp;
  //   CHECK_NVML(nvmlDeviceGetTemperatureThreshold(
  //       device, NVML_TEMPERATURE_THRESHOLD_GPU_MAX, &max_temp));
  //   if (temperature >= max_temp + safe_region) {
  //     nvmlShutdown();
  //     return false;
  //   }

  //   unsigned int shutdown_temp;
  //   CHECK_NVML(nvmlDeviceGetTemperatureThreshold(
  //       device, NVML_TEMPERATURE_THRESHOLD_SHUTDOWN, &shutdown_temp));
  //   if (temperature >= shutdown_temp + safe_region) {
  //     nvmlShutdown();
  //     return false;
  //   }
  // }

  // // Shutdown NVML
  // nvmlShutdown();
  constexpr unsigned TEMP_CLASS_NUM = 4;
  double temps[TEMP_CLASS_NUM];
  get_gpu_temp(&temps);
  for (size_t i = 1; i < TEMP_CLASS_NUM; i++) {
    if (temps[0] >= temps[i] + safe_region) return false;
  }

  return true;
}

void wait_gpu_cooling(useconds_t interval, unsigned safe_region) {
  if (gpu_temp_in_range(safe_region)) {
    return;
  }
  auto start = current_seconds();
  printf("Waiting GPU cooling down...\n");
  usleep(interval);
  while (true) {
    if (gpu_temp_in_range(safe_region)) break;
    usleep(interval);
  }
  auto end = current_seconds();
  printf("Waiting GPU cooling down: %lfs\n", end - start);
}