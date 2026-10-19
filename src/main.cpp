#include <omp.h>

#include <csignal>
#include <cstdio>
#include <exception>
#include <stdexcept>
#include <string>

#include "blas.hpp"
#include "simulation.hpp"

namespace {

// Exit code of a run that stopped early and left a checkpoint to continue from (EX_TEMPFAIL).
constexpr int kExitContinue = 75;

volatile std::sig_atomic_t g_stop = 0;

// The first signal asks for a stop after the current step; a second one kills the process.
extern "C" void on_stop_signal(int sig) {
  g_stop = 1;
  std::signal(sig, SIG_DFL);
}

void usage() {
  std::printf(
      "Usage: uppe_thz_dc [options]\n"
      "  --z-max <cm>          stop once z reaches this distance (default: run until killed, as the Fortran code)\n"
      "  --threads <n>         number of OpenMP threads (default: OMP_NUM_THREADS or all cores)\n"
      "  --data-dir <dir>      directory with N2.dat and O2.dat (default: current directory)\n"
      "  --out-dir <dir>       directory for res/, SpSIC.dat and Incident.dat (default: current directory)\n"
      "  --checkpoint <file>   on an early stop (time limit, SIGINT/SIGTERM, --max-steps) save the state here\n"
      "  --resume <file>       continue from a checkpoint; res/Imax.dat, Nemax.dat, Energy.dat are appended to\n"
      "  --wall-limit-min <m>  stop before the run exceeds m minutes of wall time\n"
      "  --min-free-gb <g>     stop once less than g GB (1e9 bytes) are free on the disk of --out-dir\n"
      "  --max-steps <n>       stop after n z steps in total\n"
      "  --snap-dz-cm <cm>     write snapshots every <cm> (default: 0.25 cm, 0.1 cm between 19 and 20.2 cm)\n"
      "  --linear              switch the nonlinear response off (linear reference run)\n"
      "  --dphi-max <rad>      shorten dz so that the plasma phase per step at the carrier stays below <rad>\n"
      "Exit code: 0 when --z-max is reached, %d when the run stopped early, 1 on error.\n",
      kExitContinue);
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
      } else if (opt == "--checkpoint") {
        cfg.checkpoint_path = value();
      } else if (opt == "--resume") {
        cfg.resume_path = value();
      } else if (opt == "--wall-limit-min") {
        const double m = std::stod(value());
        if (!(m > 0.0)) throw std::runtime_error("--wall-limit-min must be positive");
        cfg.wall_limit_s = m * 60.0;
      } else if (opt == "--min-free-gb") {
        const double g = std::stod(value());
        if (!(g > 0.0)) throw std::runtime_error("--min-free-gb must be positive");
        cfg.min_free_bytes = g * 1e9;
      } else if (opt == "--max-steps") {
        cfg.max_steps = std::stoll(value());
      } else if (opt == "--snap-dz-cm") {
        const double s = std::stod(value());
        if (!(s > 0.0)) throw std::runtime_error("--snap-dz-cm must be positive");
        cfg.snap_dz_cm = s;
      } else if (opt == "--linear") {
        cfg.linear = true;
      } else if (opt == "--dphi-max") {
        const double p = std::stod(value());
        if (!(p > 0.0)) throw std::runtime_error("--dphi-max must be positive");
        cfg.dphi_max = p;
      } else if (opt == "-h" || opt == "--help") {
        usage();
        return 0;
      } else {
        throw std::runtime_error("unknown option " + opt);
      }
    }
    std::signal(SIGINT, on_stop_signal);
    std::signal(SIGTERM, on_stop_signal);
    cfg.stop_flag = &g_stop;
    uppe::blas_init();
    if (uppe::run_simulation(cfg) == uppe::RunStatus::Stopped) return kExitContinue;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
  return 0;
}
