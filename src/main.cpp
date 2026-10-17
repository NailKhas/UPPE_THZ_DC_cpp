#include <omp.h>

#include <cstdio>
#include <exception>
#include <stdexcept>
#include <string>

#include "blas.hpp"
#include "simulation.hpp"

namespace {

void usage() {
  std::printf(
      "Usage: uppe_thz_dc [options]\n"
      "  --z-max <cm>      stop once z reaches this distance (default: run until killed, as the Fortran code)\n"
      "  --threads <n>     number of OpenMP threads (default: OMP_NUM_THREADS or all cores)\n"
      "  --data-dir <dir>  directory with N2.dat and O2.dat (default: current directory)\n"
      "  --out-dir <dir>   directory for res/, SpSIC.dat and Incident.dat (default: current directory)\n");
}

}  // namespace

int main(int argc, char** argv) {
  try {
    uppe::RunConfig cfg;
    for (int a = 1; a < argc; ++a) {
      const std::string opt = argv[a];
      auto value = [&]() -> std::string {
        if (a + 1 >= argc) throw std::runtime_error("missing value for " + opt);
        return argv[++a];
      };
      if (opt == "--z-max") {
        cfg.z_max_cm = std::stod(value());
      } else if (opt == "--threads") {
        const int n = std::stoi(value());
        if (n < 1) throw std::runtime_error("--threads must be positive");
        omp_set_num_threads(n);
      } else if (opt == "--data-dir") {
        cfg.data_dir = value();
      } else if (opt == "--out-dir") {
        cfg.out_dir = value();
      } else if (opt == "-h" || opt == "--help") {
        usage();
        return 0;
      } else {
        throw std::runtime_error("unknown option " + opt);
      }
    }
    uppe::blas_init();
    uppe::run_simulation(cfg);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
  return 0;
}
