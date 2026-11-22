// kw_THz writers: the binary file reads back exactly, the text file has the Fortran-like rows.

#include <cmath>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "check.hpp"
#include "grids.hpp"
#include "kw_output.hpp"
#include "output.hpp"

using namespace uppe;
namespace fs = std::filesystem;

namespace {

std::string read_file(const fs::path& p) {
  std::ifstream f(p, std::ios::binary);
  return std::string(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
}

}  // namespace

int main() {
  test::init();
  const TimeFreq tf = make_time_freq(4096);
  const int nR = 30;
  const int nrT = nR - 1;
  const int nqT = tf.nqT;
  const std::vector<double> roots = bessel_j0_roots(nR + 5);
  const RadialGrid gT = make_radial(roots, nR, 10, RadialKind::THz);
  const Band bandT = make_band(tf.ntT, tf.dw_THz, tf.wmin_THz);
  CHECK(gT.size() == nrT && bandT.nq() == nqT);

  test::Rng rng;
  std::vector<cplx> E(static_cast<std::size_t>(nrT) * nqT);
  for (auto& v : E) v = rng.cuniform(1e3);

  const fs::path dir = fs::temp_directory_path() / "uppe_test_kw_output";
  fs::remove_all(dir);
  fs::create_directories(dir);

  try {
    const double z_cm = 77.0123;
    const fs::path bin = dir / "Z7701.kwb";
    write_kw_binary(bin, E.data(), nrT, nqT, gT, bandT, z_cm);
    CHECK(fs::exists(bin));
    CHECK(!fs::exists(fs::path(bin.string() + ".tmp")));
    const int nf = nqT - 1;
    CHECK(fs::file_size(bin) == 48u + 8u * (2u * nf + nrT) + 4u * static_cast<std::uintmax_t>(nrT) * nf);

    const KwData d = read_kw_binary(bin);
    CHECK(d.nr == nrT && d.nf == nf);
    CHECK(d.z_cm == z_cm && d.t0 == c::t0 && d.a0 == c::a0 && d.cL == c::cL);
    bool axes = d.f_THz.size() == static_cast<std::size_t>(nf) && d.k_cm.size() == static_cast<std::size_t>(nf) &&
                d.kr_cm.size() == static_cast<std::size_t>(nrT);
    for (int q = 1; axes && q < nqT; ++q)
      axes = d.f_THz[q - 1] == static_cast<double>(bandT.fr[nqT + q]) &&
             d.k_cm[q - 1] == bandT.freqs[nqT + q] / (c::cL * c::t0);
    for (int i = 0; axes && i < nrT; ++i) axes = d.kr_cm[i] == gT.kr[i] / c::a0;
    CHECK(axes);
    bool e2 = d.e2.size() == static_cast<std::size_t>(nrT) * nf;
    for (int i = 0; e2 && i < nrT; ++i)
      for (int q = 1; e2 && q < nqT; ++q)
        e2 = d.e2[static_cast<std::size_t>(i) * nf + q - 1] ==
             static_cast<float>(abs2(E[static_cast<std::size_t>(i) * nqT + q]));
    CHECK(e2);
    // The text angle asin(kr/freqs*cL*t0/a0) is asin(kr_cm/k_cm) of the binary axes.
    if (axes) {
      const double s = gT.kr[3] / bandT.freqs[nqT + 100] * c::cL * c::t0 / c::a0;
      CHECK_CLOSE(d.kr_cm[3] / d.k_cm[99], s, 1e-14 * std::fabs(s));
    }

    // Text: the rows of the Fortran kw_THz file, radial nodes outer, frequencies inner.
    const fs::path txt = dir / "Z7701.dat";
    write_kw_text(txt, E.data(), nrT, nqT, gT, bandT);
    CHECK(!fs::exists(fs::path(txt.string() + ".tmp")));
    LineBuffer expected;
    for (int i = 0; i < nrT; ++i)
      for (int q = 1; q < nqT; ++q) {
        const int itt = nqT + q;
        const float angle = static_cast<float>(std::asin(gT.kr[i] / bandT.freqs[itt] * c::cL * c::t0 / c::a0));
        expected.f(bandT.fr[itt]).f(angle).d(abs2(E[static_cast<std::size_t>(i) * nqT + q])).nl();
      }
    CHECK(read_file(txt) == expected.str());

    // A damaged or truncated file is rejected.
    const std::string full = read_file(bin);
    std::ofstream(dir / "short.kwb", std::ios::binary).write(full.data(), static_cast<std::streamsize>(full.size() - 1));
    std::string magic = full;
    magic[0] = 'X';
    std::ofstream(dir / "magic.kwb", std::ios::binary).write(magic.data(), static_cast<std::streamsize>(magic.size()));
    std::string longer = full + "x";
    std::ofstream(dir / "long.kwb", std::ios::binary).write(longer.data(), static_cast<std::streamsize>(longer.size()));
    for (const char* name : {"short.kwb", "magic.kwb", "long.kwb"}) {
      bool threw = false;
      try {
        read_kw_binary(dir / name);
      } catch (const std::exception&) {
        threw = true;
      }
      if (!threw) std::fprintf(stderr, "  accepted: %s\n", name);
      CHECK(threw);
    }

    // The schedule hash tells lists and formats apart.
    const std::vector<double> a = {77, 80, 85}, b = {77, 80, 85.5};
    CHECK(kw_schedule_hash(a, KwFormat::Binary) == kw_schedule_hash(a, KwFormat::Binary));
    CHECK(kw_schedule_hash(a, KwFormat::Binary) != kw_schedule_hash(b, KwFormat::Binary));
    CHECK(kw_schedule_hash(a, KwFormat::Binary) != kw_schedule_hash(a, KwFormat::Text));
    CHECK(kw_schedule_hash({}, KwFormat::Binary) != kw_schedule_hash({}, KwFormat::Text));
  } catch (const std::exception& e) {
    std::fprintf(stderr, "kw_output failed: %s\n", e.what());
    return 1;
  }

  fs::remove_all(dir);
  return test::finish("test_kw_output");
}
