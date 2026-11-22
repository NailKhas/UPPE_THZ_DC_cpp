// Runs the full program on a reduced grid from zsic through the optical stage into the THz stage
// and checks that the outputs are sane.
// Usage: test_simulation_smoke <data dir with N2.dat and O2.dat>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iterator>
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

// z_pos of the snapshot files Z<z_pos>.dat in dir.
std::vector<int> snapshot_positions(const fs::path& dir) {
  std::vector<int> v;
  if (!fs::is_directory(dir)) return v;
  for (const auto& e : fs::directory_iterator(dir)) {
    const std::string name = e.path().filename().string();
    if (name.size() > 5 && name[0] == 'Z') v.push_back(std::atoi(name.c_str() + 1));
  }
  return v;
}

RunConfig smoke_config(const fs::path& data, const fs::path& out) {
  RunConfig cfg;
  cfg.sizes.nt_all = 131072;
  cfg.sizes.nR_first = 200;
  cfg.sizes.nR_opt = 20;
  cfg.sizes.nR_THz = 60;
  // Past the end of the optical stage (71 cm) and the first THz-stage snapshot.
  cfg.z_max_cm = c::opt_stage_end + 0.2;
  cfg.max_steps = 100000;
  cfg.data_dir = data;
  cfg.out_dir = out;
  fs::remove_all(cfg.out_dir);
  fs::create_directories(cfg.out_dir);
  return cfg;
}

}  // namespace

int main(int argc, char** argv) {
  test::init();
  const fs::path data = argc > 1 ? fs::path(argv[1]) : fs::path("data");

  // kw_THz at its own points in the binary format. 71.0 cm falls on the first THz-stage step.
  RunConfig bin = smoke_config(data, fs::temp_directory_path() / "uppe_test_smoke_kwb");
  bin.kw_z_cm = {c::opt_stage_end, c::opt_stage_end + 0.1};
  bin.kw_format = KwFormat::Binary;
  // kw_THz as text with every THz-stage snapshot, as the Fortran code does.
  RunConfig cfg = smoke_config(data, fs::temp_directory_path() / "uppe_test_smoke");
  cfg.kw_format = KwFormat::Text;

  try {
    CHECK(run_simulation(bin) == RunStatus::Finished);
    CHECK(run_simulation(cfg) == RunStatus::Finished);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "run_simulation failed: %s\n", e.what());
    return 1;
  }

  {
    const fs::path res = bin.out_dir / "res";
    const int nKw = count_files(res / "kw_THz");
    CHECK(nKw >= 1 && nKw <= 2);
    for (int zp : snapshot_positions(res / "kw_THz")) CHECK(zp >= 100 * c::opt_stage_end);
    CHECK(count_files(res / "IS") == static_cast<int>(read_table(res / "snapshots.txt").size()));
    const TimeFreq tf = make_time_freq(bin.sizes.nt_all);
    const std::uintmax_t nr = bin.sizes.nR_THz - 1, nf = tf.nqT - 1;
    for (const auto& e : fs::directory_iterator(res / "kw_THz")) {
      CHECK(e.path().extension() == ".kwb");
      CHECK(fs::file_size(e.path()) == 48 + 8 * (2 * nf + nr) + 4 * nr * nf);
      try {
        const KwData d = read_kw_binary(e.path());
        CHECK(d.nr == static_cast<int>(nr) && d.nf == static_cast<int>(nf));
        CHECK(d.z_cm >= c::opt_stage_end && d.z_cm < bin.z_max_cm);
        CHECK(static_cast<int>(d.z_cm * 1e2) == std::atoi(e.path().filename().string().c_str() + 1));
        bool finite = true;
        for (float v : d.e2) finite = finite && std::isfinite(v) && v >= 0.0f;
        CHECK(finite);
      } catch (const std::exception& ex) {
        std::fprintf(stderr, "read_kw_binary failed: %s\n", ex.what());
        CHECK(false);
      }
    }
    std::ifstream params(res / "parameters.dat");
    const std::string text((std::istreambuf_iterator<char>(params)), std::istreambuf_iterator<char>());
    CHECK(text.find("kw_THz:") != std::string::npos && text.find("71,71.1 cm (bin, .kwb)") != std::string::npos);
    fs::remove_all(bin.out_dir);
  }

  const fs::path res = cfg.out_dir / "res";

  const auto imax = read_table(res / "Imax.dat");
  CHECK(imax.size() >= 3);
  CHECK(all_finite(imax));
  for (const auto& r : imax) {
    CHECK(r.size() == 3);
    if (r.size() == 3) CHECK(r[1] > 0.0 && r[2] > 0.0 && r[2] <= r[1]);
  }
  for (std::size_t k = 1; k < imax.size(); ++k) CHECK(imax[k][0] > imax[k - 1][0]);  // z increases
  if (!imax.empty()) CHECK_CLOSE(imax[0][0], c::zsic, 1e-9);
  // Steps on both sides of the stage boundary.
  int optSteps = 0, thzSteps = 0;
  for (const auto& r : imax) (r[0] < c::opt_stage_end ? optSteps : thzSteps)++;
  CHECK(optSteps > 0 && thzSteps > 0);

  const auto energy = read_table(res / "Energy.dat");
  CHECK(energy.size() == imax.size());
  CHECK(all_finite(energy));
  for (const auto& r : energy) CHECK(r.size() == 2 && r[1] > 0.0);
  // Two short steps in weak nonlinearity cannot change the energy much.
  if (energy.size() >= 2) CHECK(std::fabs(energy[1][1] / energy[0][1] - 1.0) < 0.1);

  const auto nemax = read_table(res / "Nemax.dat");
  CHECK(nemax.size() == imax.size());
  CHECK(all_finite(nemax));
  for (const auto& r : nemax) CHECK(r.size() == 2 && r[1] >= 0.0 && r[1] <= 1.0);

  const auto den = read_table(res / "Den.dat");
  CHECK(den.size() == nemax.size());
  CHECK(all_finite(den));
  for (const auto& r : den) CHECK(r.size() == 2 && r[1] >= 0.0);
  for (std::size_t k = 0; k < den.size() && k < nemax.size(); ++k)
    if (den[k].size() == 2 && nemax[k].size() == 2) CHECK(den[k][0] == nemax[k][0]);

  for (const char* d : {"IS", "Fl", "kw_THz", "Or", "Ne", "IO"}) CHECK(count_files(res / d) >= 1);
  // Snapshots: zsic (the first step, z_flag = 56), then 61 and 66 cm in the optical stage, from 71 cm every
  // 0.1 cm in the THz stage; kw_THz only in the THz stage.
  const auto snaps = read_table(res / "snapshots.txt");
  CHECK(snaps.size() >= 4);
  if (!snaps.empty()) CHECK(std::fabs(snaps[0][0] - 100 * c::zsic) <= 1.0);
  CHECK(count_files(res / "IS") == static_cast<int>(snaps.size()));
  int optSnaps = 0;
  for (const auto& r : snaps) optSnaps += r[0] < 100 * c::opt_stage_end;
  CHECK(optSnaps == 3);
  for (int zp : snapshot_positions(res / "kw_THz")) CHECK(zp >= 100 * c::opt_stage_end);
  CHECK(count_files(res / "kw_THz") == static_cast<int>(snaps.size()) - optSnaps);
  if (!snaps.empty()) {
    // THz rows of IS and IO in the optical stage are zeros.
    char name[32];
    std::snprintf(name, sizeof name, "Z%04d.dat", static_cast<int>(snaps[0][0]));
    const auto is = read_table(res / "IS" / name);
    const auto io = read_table(res / "IO" / name);
    CHECK(!is.empty() && is[0].size() == 2 && is[0][1] == 0.0);
    CHECK(!io.empty() && io[0].size() == 2 && io[0][1] == 0.0);
    CHECK(is.size() == io.size());
  }
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
