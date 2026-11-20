// A run stopped at checkpoints and resumed, once in the optical stage and once in the THz stage, must give
// byte-identical output to an uninterrupted run.
// Usage: test_checkpoint <data dir with N2.dat and O2.dat>

#include <cstdio>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <set>
#include <string>

#include "check.hpp"
#include "checkpoint.hpp"
#include "simulation.hpp"

using namespace uppe;
namespace fs = std::filesystem;

namespace {

// A little past the end of the optical stage; dz stays below 0.1 cm there, so the THz stage gets several steps.
constexpr double kZmax = c::opt_stage_end + 0.3;

std::string read_file(const fs::path& p) {
  std::ifstream f(p, std::ios::binary);
  return std::string(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
}

// Relative paths of all regular files under dir.
std::set<std::string> list_files(const fs::path& dir) {
  std::set<std::string> names;
  for (const auto& e : fs::recursive_directory_iterator(dir))
    if (e.is_regular_file()) names.insert(fs::relative(e.path(), dir).generic_string());
  return names;
}

int count_lines(const fs::path& p) {
  const std::string s = read_file(p);
  int n = 0;
  for (char ch : s) n += ch == '\n';
  return n;
}

RunConfig base_config(const fs::path& data, const fs::path& out) {
  RunConfig cfg;
  cfg.sizes.nt_all = 131072;
  cfg.sizes.nR_first = 200;
  cfg.sizes.nR_opt = 20;
  cfg.sizes.nR_THz = 60;
  cfg.data_dir = data;
  cfg.out_dir = out;
  fs::remove_all(out);
  fs::create_directories(out);
  return cfg;
}

bool throws_on_read(const fs::path& file, const Sizes& sizes) {
  const TimeFreq tf = make_time_freq(sizes.nt_all);
  AlignedArray<cplx> E_O(static_cast<std::size_t>(sizes.nR_opt - 1) * tf.nqO);
  AlignedArray<cplx> E_T(static_cast<std::size_t>(sizes.nR_THz - 1) * tf.nqT);
  try {
    read_checkpoint(file, sizes, E_O, E_T);
  } catch (const std::exception&) {
    return true;
  }
  return false;
}

int stage_of(const fs::path& file, const Sizes& sizes) {
  const TimeFreq tf = make_time_freq(sizes.nt_all);
  AlignedArray<cplx> E_O(static_cast<std::size_t>(sizes.nR_opt - 1) * tf.nqO);
  AlignedArray<cplx> E_T(static_cast<std::size_t>(sizes.nR_THz - 1) * tf.nqT);
  return read_checkpoint(file, sizes, E_O, E_T).stage;
}

// Number of z steps that started in the optical stage (the z column of Imax.dat).
long long optical_steps(const fs::path& imax) {
  std::ifstream f(imax);
  long long n = 0;
  double z = 0.0, a = 0.0, b = 0.0;
  while (f >> z >> a >> b) n += z < c::opt_stage_end;
  return n;
}

}  // namespace

int main(int argc, char** argv) {
  test::init();
  const fs::path data = argc > 1 ? fs::path(argv[1]) : fs::path("data");
  const fs::path root = fs::temp_directory_path() / "uppe_test_checkpoint";
  const fs::path dirR = root / "continuous";
  const fs::path dirB = root / "resumed";
  const fs::path dirA = root / "stopped";
  const fs::path ckA = root / "a.ckpt";
  const fs::path ckB1 = root / "b1.ckpt";
  const fs::path ckB2 = root / "b2.ckpt";
  const Sizes sizes = base_config(data, root / "scratch").sizes;

  try {
    // Uninterrupted run to kZmax.
    RunConfig r = base_config(data, dirR);
    r.z_max_cm = kZmax;
    CHECK(run_simulation(r) == RunStatus::Finished);
    const long long nOpt = optical_steps(dirR / "res" / "Imax.dat");
    CHECK(nOpt >= 2);
    CHECK(count_lines(dirR / "res" / "Imax.dat") >= nOpt + 2);

    // Stop one step before the end of the optical stage, then one step into the THz stage, then finish.
    RunConfig b1 = base_config(data, dirB);
    b1.max_steps = nOpt - 1;
    b1.checkpoint_path = ckB1;
    CHECK(run_simulation(b1) == RunStatus::Stopped);
    CHECK(stage_of(ckB1, sizes) == 0);
    const int snapsBefore = count_lines(dirB / "res" / "snapshots.txt");

    RunConfig b2 = b1;
    b2.max_steps = nOpt + 1;
    b2.resume_path = ckB1;
    b2.checkpoint_path = ckB2;
    CHECK(run_simulation(b2) == RunStatus::Stopped);
    CHECK(stage_of(ckB2, sizes) == 1);
    // The 71 cm snapshot is written after the resume, so the snapshot schedule is checked too.
    CHECK(count_lines(dirB / "res" / "snapshots.txt") > snapsBefore);

    RunConfig b3 = b2;
    b3.max_steps = -1;
    b3.z_max_cm = kZmax;
    b3.resume_path = ckB2;
    b3.checkpoint_path.clear();
    CHECK(run_simulation(b3) == RunStatus::Finished);

    // An uninterrupted run stopped at the same step as b2 writes the same checkpoint.
    RunConfig a = base_config(data, dirA);
    a.max_steps = nOpt + 1;
    a.checkpoint_path = ckA;
    CHECK(run_simulation(a) == RunStatus::Stopped);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "run_simulation failed: %s\n", e.what());
    return 1;
  }

  const auto filesR = list_files(dirR);
  const auto filesB = list_files(dirB);
  CHECK(filesR == filesB);
  for (const auto& name : filesR) {
    if (!filesB.count(name)) continue;
    const bool same = read_file(dirR / name) == read_file(dirB / name);
    if (!same) std::fprintf(stderr, "  differs: %s\n", name.c_str());
    CHECK(same);
  }
  CHECK(read_file(ckA) == read_file(ckB2));
  CHECK(!fs::exists(fs::path(ckB2.string() + ".tmp")));

  // A damaged or foreign checkpoint is rejected.
  CHECK(!throws_on_read(ckA, sizes));
  {
    const std::string full = read_file(ckA);
    std::string flipped = full;
    flipped[full.size() / 2] ^= 1;
    // Header: magic[8], version, nt_all, nR_first, nR_opt, nR_THz (int32 each), then the int32 DC profile id.
    std::string otherProfile = full;
    otherProfile[28] ^= 1;
    std::ofstream(root / "short.ckpt", std::ios::binary)
        .write(full.data(), static_cast<std::streamsize>(full.size() / 2));
    std::ofstream(root / "flipped.ckpt", std::ios::binary)
        .write(flipped.data(), static_cast<std::streamsize>(flipped.size()));
    std::ofstream(root / "profile.ckpt", std::ios::binary)
        .write(otherProfile.data(), static_cast<std::streamsize>(otherProfile.size()));
  }
  CHECK(throws_on_read(root / "short.ckpt", sizes));
  CHECK(throws_on_read(root / "flipped.ckpt", sizes));
  CHECK(throws_on_read(root / "profile.ckpt", sizes));
  Sizes other = sizes;
  other.nR_THz = 64;
  CHECK(throws_on_read(ckA, other));

  fs::remove_all(root);
  return test::finish("test_checkpoint");
}
