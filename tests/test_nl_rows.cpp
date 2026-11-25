// The nonlinear response on the first R_nl rows (--nl-eps) against all rows, through both stages.
// Usage: test_nl_rows <data dir with N2.dat and O2.dat>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <system_error>
#include <vector>

#include "check.hpp"
#include "kw_output.hpp"
#include "simulation.hpp"

using namespace uppe;
namespace fs = std::filesystem;

namespace {

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

// Largest relative difference between two tables, entry by entry (1 if the shapes differ).
double table_diff(const std::vector<std::vector<double>>& a, const std::vector<std::vector<double>>& b) {
  if (a.size() != b.size() || a.empty()) return 1.0;
  double d = 0.0;
  for (std::size_t k = 0; k < a.size(); ++k) {
    if (a[k].size() != b[k].size()) return 1.0;
    for (std::size_t j = 0; j < a[k].size(); ++j) {
      const double s = std::max(std::fabs(a[k][j]), std::fabs(b[k][j]));
      if (s > 0.0) d = std::max(d, std::fabs(a[k][j] - b[k][j]) / s);
    }
  }
  return d;
}

// Relative L2 difference ||a - b|| / ||a|| (1 if the sizes differ).
double l2_diff(const std::vector<float>& a, const std::vector<float>& b) {
  if (a.size() != b.size() || a.empty()) return 1.0;
  double num = 0.0, den = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    num += std::pow(static_cast<double>(a[i]) - b[i], 2);
    den += std::pow(static_cast<double>(a[i]), 2);
  }
  return den > 0.0 ? std::sqrt(num / den) : 1.0;
}

struct Result {
  std::vector<std::vector<std::vector<double>>> tables;  // Imax, Energy, Nemax, Den
  std::vector<float> kw;                                 // |E_THz(w,kr)|^2 at 71.1 cm
};

Result run(const fs::path& data, const fs::path& out, double eps) {
  RunConfig cfg;
  cfg.sizes.nt_all = 131072;
  cfg.sizes.nR_first = 200;
  // Enough rows for the field to fall off inside the grid; pad = 0 lets the cut reach them.
  cfg.sizes.nR_opt = 100;
  cfg.sizes.nR_THz = 120;
  cfg.nl_eps = eps;
  cfg.nl_pad = 0;
  cfg.z_max_cm = c::opt_stage_end + 0.15;
  cfg.kw_z_cm = {c::opt_stage_end + 0.1};
  cfg.kw_format = KwFormat::Binary;
  cfg.data_dir = data;
  cfg.out_dir = out;
  std::error_code ec;
  fs::remove_all(out, ec);
  fs::create_directories(out);

  Result r;
  CHECK(run_simulation(cfg) == RunStatus::Finished);
  const fs::path res = out / "res";
  for (const char* f : {"Imax.dat", "Energy.dat", "Nemax.dat", "Den.dat"}) r.tables.push_back(read_table(res / f));
  for (const auto& e : fs::directory_iterator(res / "kw_THz")) r.kw = read_kw_binary(e.path()).e2;
  // A file still held by another process (an antivirus scan) must not abort the test.
  fs::remove_all(out, ec);
  return r;
}

double diff(const Result& a, const Result& b, double& kw) {
  double d = 0.0;
  for (std::size_t k = 0; k < a.tables.size(); ++k) d = std::max(d, table_diff(a.tables[k], b.tables[k]));
  kw = l2_diff(a.kw, b.kw);
  return d;
}

}  // namespace

int main(int argc, char** argv) {
  test::init();
  const fs::path data = argc > 1 ? fs::path(argv[1]) : fs::path("data");
  const fs::path dir = fs::temp_directory_path() / "uppe_test_nl_rows";

  // On this grid the field stays above 1e-9 of Imax up to the edge, so the default eps = 1e-12 keeps all rows;
  // 1e-6 drops up to a third of them, 1e-3 (capped by the ionization level) up to two thirds.
  try {
    const Result all = run(data, dir, 0.0);
    CHECK(!all.kw.empty());
    double kw = 0.0;
    const double d6 = diff(all, run(data, dir, 1e-6), kw);
    std::printf(" nl_eps 1e-6: scalars %.3g, kw_THz %.3g\n", d6, kw);
    CHECK(d6 < 1e-12);
    CHECK(kw < 1e-7);
    const double d3 = diff(all, run(data, dir, 1e-3), kw);
    std::printf(" nl_eps 1e-3: scalars %.3g, kw_THz %.3g\n", d3, kw);
    CHECK(d3 < 1e-9);
    CHECK(kw > 0.0 && kw < 1e-7);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "run_simulation failed: %s\n", e.what());
    return 1;
  }
  return test::finish("test_nl_rows");
}
