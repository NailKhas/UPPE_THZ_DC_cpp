#include <omp.h>

#include <csignal>
#include <cstdio>
#include <exception>
#include <stdexcept>
#include <string>
#include <vector>

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
      "  --resume <file>       continue from a checkpoint; res/Imax.dat, Nemax.dat, Energy.dat, Den.dat are appended to\n"
      "  --wall-limit-min <m>  stop before the run exceeds m minutes of wall time\n"
      "  --min-free-gb <g>     stop once less than g GB (1e9 bytes) are free on the disk of --out-dir\n"
      "  --max-steps <n>       stop after n z steps in total\n"
      "  --kw-z <a,b,...>      write res/kw_THz only at these z (cm, ascending, >= 71), once each, on the first step\n"
      "                        with z >= the point (default: with every THz-stage snapshot)\n"
      "  --kw-format bin|text  res/kw_THz format: bin = Z<pos>.kwb, float32 |E|^2 (default), text = Fortran-like Z<pos>.dat\n"
      "Exit code: 0 when --z-max is reached, %d when the run stopped early, 1 on error.\n",
      kExitContinue);
}

// Comma-separated numbers, e.g. "77,80,85".
std::vector<double> parse_list(const std::string& s, const std::string& opt) {
  std::vector<double> v;
  std::size_t begin = 0;
  while (true) {
    const std::size_t end = s.find(',', begin);
    const std::string tok = s.substr(begin, end == std::string::npos ? std::string::npos : end - begin);
    std::size_t used = 0;
    double x = 0.0;
    try {
      x = std::stod(tok, &used);
    } catch (const std::exception&) {
      used = 0;
    }
    if (tok.empty() || used != tok.size()) throw std::runtime_error(opt + ": bad number '" + tok + "'");
    v.push_back(x);
    if (end == std::string::npos) break;
    begin = end + 1;
  }
  return v;
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
      } else if (opt == "--kw-z") {
        cfg.kw_z_cm = parse_list(value(), opt);
      } else if (opt == "--kw-format") {
        const std::string f = value();
        if (f == "bin")
          cfg.kw_format = uppe::KwFormat::Binary;
        else if (f == "text")
          cfg.kw_format = uppe::KwFormat::Text;
        else
          throw std::runtime_error("--kw-format must be bin or text");
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
