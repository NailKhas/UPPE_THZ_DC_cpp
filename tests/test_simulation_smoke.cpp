// Runs two z steps of the full program on a reduced grid and checks that the outputs are sane.
// Usage: test_simulation_smoke <data dir with N2.dat and O2.dat>

#include <cmath>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "check.hpp"
#include "simulation.hpp"

using namespace uppe;
namespace fs = std::filesystem;

namespace {

// Rows of whitespace-separated numbers; "nan"/"inf" tokens are parsed as such.
std::vector<std::vector<double>> read_table(const fs::path& p) {
  std::vector<std::vector<double>> rows;
  std::ifstream f(p);
  std::string line;
  while (std::getline(f, line)) {
    std::istringstream ss(line);
    std::vector<double> row;
    std::string tok;
    while (ss >> tok) row.push_back(std::strtod(tok.c_str(), nullptr));
    if (!row.empty()) rows.push_back(row);
  }
  return rows;
}

bool all_finite(const std::vector<std::vector<double>>& t) {
  for (const auto& r : t)
    for (double v : r)
      if (!std::isfinite(v)) return false;
  return true;
}

int count_files(const fs::path& dir) {
  int n = 0;
  if (!fs::is_directory(dir)) return 0;
  for (const auto& e : fs::directory_iterator(dir))
    if (e.is_regular_file()) ++n;
  return n;
}

}  // namespace

int main(int argc, char** argv) {
  test::init();

  RunConfig cfg;
  cfg.sizes.nt_all = 131072;
  cfg.sizes.nR_first = 200;
  cfg.sizes.nR_opt = 20;
  cfg.sizes.nR_THz = 60;
  cfg.max_steps = 2;
  cfg.data_dir = argc > 1 ? fs::path(argv[1]) : fs::path("data");
  cfg.out_dir = fs::temp_directory_path() / "uppe_test_smoke";
  fs::remove_all(cfg.out_dir);
  fs::create_directories(cfg.out_dir);

  try {
    run_simulation(cfg);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "run_simulation failed: %s\n", e.what());
    return 1;
  }

  const fs::path res = cfg.out_dir / "res";

  const auto imax = read_table(res / "Imax.dat");
  CHECK(imax.size() == 2);
  CHECK(all_finite(imax));
  for (const auto& r : imax) {
    CHECK(r.size() == 3);
    if (r.size() == 3) CHECK(r[1] > 0.0 && r[2] > 0.0 && r[2] <= r[1]);
  }
  if (imax.size() == 2) CHECK(imax[1][0] > imax[0][0]);  // z increases

  const auto energy = read_table(res / "Energy.dat");
  CHECK(energy.size() == 2);
  CHECK(all_finite(energy));
  for (const auto& r : energy) CHECK(r.size() == 2 && r[1] > 0.0);
  // Two short steps in weak nonlinearity cannot change the energy much.
  if (energy.size() == 2) CHECK(std::fabs(energy[1][1] / energy[0][1] - 1.0) < 0.1);

  const auto nemax = read_table(res / "Nemax.dat");
  CHECK(nemax.size() == 2);
  CHECK(all_finite(nemax));
  for (const auto& r : nemax) CHECK(r.size() == 2 && r[1] >= 0.0 && r[1] <= 1.0);

  for (const char* d : {"IS", "Fl", "kw_THz", "Or", "Ne", "IO"}) CHECK(count_files(res / d) >= 1);
  // kw_THz may contain nan (asin of an argument > 1, as in the Fortran); the others must be finite.
  for (const char* d : {"IS", "Fl", "Or", "Ne", "IO"}) {
    if (!fs::is_directory(res / d)) continue;
    for (const auto& e : fs::directory_iterator(res / d)) {
      const auto t = read_table(e.path());
      CHECK(!t.empty());
      CHECK(all_finite(t));
    }
  }

  for (const char* f : {"parameters.dat", "rO1.dat", "rO1_um.dat", "krO1.dat", "r1T.dat", "r1T_um.dat", "krT.dat",
                        "wO2.dat", "wrO2.dat", "krOmax2.dat", "w_kzO2.dat", "wT2.dat", "wrT2.dat", "krTmax2.dat",
                        "w_kzT2.dat", "wA_coll2.dat", "Den.dat"})
    CHECK(fs::exists(res / f));
  CHECK(fs::exists(cfg.out_dir / "SpSIC.dat"));
  CHECK(fs::exists(cfg.out_dir / "Incident.dat"));

  fs::remove_all(cfg.out_dir);
  return test::finish("test_simulation_smoke");
}
