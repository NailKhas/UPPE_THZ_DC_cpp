// A run stopped at a checkpoint and resumed must give byte-identical output to an uninterrupted run.
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

// On this grid dz is about 0.024 cm, so the 18.25 cm snapshot falls between kSplit and kTotal.
constexpr long long kSplit = 6;
constexpr long long kTotal = 14;

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

}  // namespace

int main(int argc, char** argv) {
  test::init();
  const fs::path data = argc > 1 ? fs::path(argv[1]) : fs::path("data");
  const fs::path root = fs::temp_directory_path() / "uppe_test_checkpoint";
  const fs::path dirA = root / "continuous";
  const fs::path dirB = root / "resumed";
  const fs::path ckA = root / "a.ckpt";
  const fs::path ckB = root / "b.ckpt";

  try {
    RunConfig a = base_config(data, dirA);
    a.max_steps = kTotal;
    a.checkpoint_path = ckA;
    CHECK(run_simulation(a) == RunStatus::Stopped);

    RunConfig b1 = base_config(data, dirB);
    b1.max_steps = kSplit;
    b1.checkpoint_path = ckB;
    CHECK(run_simulation(b1) == RunStatus::Stopped);
    CHECK(fs::exists(ckB));
    const int snapsBefore = count_lines(dirB / "res" / "snapshots.txt");

    RunConfig b2 = b1;
    b2.max_steps = kTotal;
    b2.resume_path = ckB;
    CHECK(run_simulation(b2) == RunStatus::Stopped);
    // At least one snapshot must be written after the resume, so the snapshot schedule is checked too.
    CHECK(count_lines(dirB / "res" / "snapshots.txt") > snapsBefore);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "run_simulation failed: %s\n", e.what());
    return 1;
  }

  const auto filesA = list_files(dirA);
  const auto filesB = list_files(dirB);
  CHECK(filesA == filesB);
  for (const auto& name : filesA) {
    if (!filesB.count(name)) continue;
    const bool same = read_file(dirA / name) == read_file(dirB / name);
    if (!same) std::fprintf(stderr, "  differs: %s\n", name.c_str());
    CHECK(same);
  }
  CHECK(read_file(ckA) == read_file(ckB));
  CHECK(!fs::exists(fs::path(ckB.string() + ".tmp")));

  // A damaged or foreign checkpoint is rejected.
  const Sizes sizes = base_config(data, root / "scratch").sizes;
  CHECK(!throws_on_read(ckA, sizes));
  {
    const std::string full = read_file(ckA);
    std::string flipped = full;
    flipped[full.size() / 2] ^= 1;
    std::ofstream(root / "short.ckpt", std::ios::binary)
        .write(full.data(), static_cast<std::streamsize>(full.size() / 2));
    std::ofstream(root / "flipped.ckpt", std::ios::binary)
        .write(flipped.data(), static_cast<std::streamsize>(flipped.size()));
  }
  CHECK(throws_on_read(root / "short.ckpt", sizes));
  CHECK(throws_on_read(root / "flipped.ckpt", sizes));
  Sizes other = sizes;
  other.nR_THz = 64;
  CHECK(throws_on_read(ckA, other));

  fs::remove_all(root);
  return test::finish("test_checkpoint");
}
