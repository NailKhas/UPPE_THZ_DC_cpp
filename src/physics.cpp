#include "physics.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>

#include "common.hpp"
#include "output.hpp"

namespace uppe {

Physics make_physics() {
  Physics p;
  // Integral of exp(-2 (r/a00)^16) over the plane is pi*a00^2*S; the Fortran I0 omits S and puts
  // 0.864*WW0 into the pulse.
  const double S = std::tgamma(9. / 8) / std::pow(2., 1. / 8);
  p.I0 = c::WW0 / sq(c::a00) / c::t0 / std::pow(c::pi, 1.5) / S;
  p.DC_Field_in_kV = static_cast<double>(4.f / 0.4f);
  p.DC_Field = p.DC_Field_in_kV * std::sqrt(static_cast<double>(31000 / 5.15f / 5.15f) / p.I0);
  p.RKer = c::n2 * p.I0 * 4. / 3;
  p.RDru = 2 * c::pi * sq(c::el) * c::N0 * sq(c::t0) / c::me;
  p.ROhm = c::cL * static_cast<double>(1.6e-19f) * c::N0 / 4. / p.I0;
  p.P = c::WW0 / (std::sqrt(c::pi) * c::t0);
  p.Pcr = 3.77 * sq(c::lmbd0) / (8 * c::pi * c::n2);
  p.P_Pcr = p.P / p.Pcr;
  // Marburger's fit holds only above the critical power.
  p.zself = p.P_Pcr > 1.0 ? 0.367 * c::k0 * sq(c::a00) / std::sqrt(sq(std::sqrt(p.P_Pcr) - 0.852) - 0.0219)
                          : std::numeric_limits<double>::quiet_NaN();
  return p;
}

namespace {

std::vector<double> read_table(const std::filesystem::path& file) {
  std::ifstream in(file);
  if (!in) throw std::runtime_error("cannot open " + path_string(file));
  std::vector<double> v(c::ion_table_size);
  std::string tok;
  for (int i = 0; i < c::ion_table_size; ++i) {
    if (!(in >> tok)) throw std::runtime_error(path_string(file) + ": expected " +
                                               std::to_string(c::ion_table_size) + " values, got " +
                                               std::to_string(i));
    char* end = nullptr;
    v[i] = std::strtod(tok.c_str(), &end);
    if (end == tok.c_str() || *end != '\0')
      throw std::runtime_error(path_string(file) + ": bad number '" + tok + "' at line " + std::to_string(i + 1));
  }
  return v;
}

}  // namespace

void load_ionization_tables(Physics& ph, const std::filesystem::path& dir) {
  ph.P_N2 = read_table(dir / "N2.dat");
  ph.P_O2 = read_table(dir / "O2.dat");
}

void write_params(const Physics& ph, double ng, double d2kdw2, int nthreads, const Sizes& sizes,
                  const TimeFreq& tf, const std::filesystem::path& outDir) {
  TextWriter f(outDir / "res" / "parameters.dat");
  auto file_line = [&](const char* s) { f.text(s).nl(); };
  auto both = [&](const char* label, double v, const char* unit) {
    std::printf(" %s %.15g %s\n", label, v, unit);
    f.text(label).d(v).text(unit).nl();
  };
  auto file_value = [&](const char* label, double v, const char* unit) { f.text(label).d(v).text(unit).nl(); };

  std::printf(" X-DOT by filamentators. FME for optical\n");
  file_line("X-DOT by filamentators. FME for optical");
  file_line("First harmonic pulse:");
  both("lmbd0   = ", c::lmbd0 * 1e7, " nm");
  both("t0      = ", c::t0 * 1e15, " fs");
  both("a00     = ", c::a00, " cm");
  both("W0      = ", c::WW0 * 1e3, " mJ");
  both("Focus   = ", c::focus, " cm");
  both("Prelens = ", c::prelens, " cm");
  both("zsic   = ", c::zsic, " cm");
  both("DC field = ", ph.DC_Field_in_kV, " kV/cm");
  file_line(" ");
  file_line("All is normalized to:");
  both("I0      = ", ph.I0 * 1e-9, " GW/cm^2");
  both("a0      = ", c::a0, " cm");
  file_value("k0      = ", c::k0, " cm^(-1)");
  both("Ldifr   = ", c::k0 * c::a0 * c::a0, " cm");
  both("Rmax    = ", c::Rmax * c::a0, " cm");
  file_line(" ");
  file_line("Model parameters:");
  both("ng      = ", ng, "");
  both("Ldisp   = ", c::cL * c::t0 / d2kdw2, " cm");
  file_value("n2      = ", c::n2, " cm^2/W");
  both("P       = ", ph.P * 1e-9, " GW");
  both("Pcr     = ", ph.Pcr * 1e-9, " GW");
  if (std::isfinite(ph.zself)) {
    both("zself   = ", ph.zself, " cm");
  } else {
    std::printf(" zself   = n/a (P < Pcr)\n");
    file_line("zself   = n/a (P < Pcr)");
  }
  both("P/Pcr   = ", ph.P_Pcr, "");
  file_line("Paralleled:");
  const float memory = 1.f * sizes.nR_THz * tf.ntT * 16 + 1.f * sizes.nR_opt * tf.ntO * 16;
  std::printf(" Paralleling by %d\n", nthreads);
  std::printf(" Memory: %.9g\n", static_cast<double>(memory));
  f.text("Paralleling by ").i(nthreads).nl();
  f.text("Memory:").f(memory).nl();
}

}  // namespace uppe
