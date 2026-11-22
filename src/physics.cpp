#include "physics.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>

#include "common.hpp"
#include "output.hpp"

namespace uppe {

Physics make_physics() {
  Physics p;
  p.I0 = c::WW0 / sq(c::a00) / c::t0 / std::pow(c::pi, 1.5);
  p.DC_norm = std::sqrt(static_cast<double>(31000 / 5.15f / 5.15f) / p.I0);
  p.RKer = c::n2 * p.I0 * 4. / 3;
  p.RDru = 2 * c::pi * sq(c::el) * c::N0 * sq(c::t0) / c::me;
  p.ROhm = c::cL * static_cast<double>(1.6e-19f) * c::N0 / 4. / p.I0;
  p.P_Pcr = p.I0 * sq(c::a0) * c::pi * 8 * c::pi * c::n2 / (3.77f * sq(c::lmbd0));
  p.zself = 0.367f * c::k0 * c::a0 * c::a0 / std::sqrt(sq(std::sqrt(p.P_Pcr) - 0.852f) - 0.0219f);
  return p;
}

double dc_field_kV(double z_cm) {
  if constexpr (c::dc_profile == 0) {
    (void)z_cm;
    return static_cast<double>(4.f / 0.4f);
  } else if constexpr (c::dc_profile == 1) {
    const double d = z_cm - c::dc_z0;
    return c::dc_amp * (1.0 / (1.0 + std::exp(-c::dc_rate * d)) - 0.5) *
           std::exp(-std::pow(std::abs(d / c::dc_width), c::dc_order));
  } else if constexpr (c::dc_profile == 2) {
    const double d = z_cm - c::dc_sg_z0;
    return c::dc_sg_amp * std::exp(-std::pow(std::abs(d / c::dc_width), c::dc_sg_order));
  } else if constexpr (c::dc_profile == 3) {
    return c::dc_sin_amp * std::sin(z_cm * 2.0 * c::dc_sin_pi * c::dc_sin_cycles / c::dc_sin_span);
  } else {
    return (z_cm < c::dc_win_begin || z_cm >= c::dc_win_end) ? 0.0 : static_cast<double>(4.f / 0.4f);
  }
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
                  const TimeFreq& tf, const std::filesystem::path& outDir, const std::vector<double>& kw_z_cm,
                  KwFormat kw_format) {
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
  both("zsic   = ", c::zsic, " cm");
  if constexpr (c::dc_profile == 0) {
    both("DC field = ", dc_field_kV(c::zsic), " kV/cm");
  } else {
    char profile[160];
    if constexpr (c::dc_profile == 1)
      std::snprintf(profile, sizeof profile, "DC field = %g*(1/(1+exp(-%g*(z-%g)))-1/2)*exp(-((z-%g)/%g)^%g) kV/cm",
                    c::dc_amp, c::dc_rate, c::dc_z0, c::dc_z0, c::dc_width, c::dc_order);
    else if constexpr (c::dc_profile == 2)
      std::snprintf(profile, sizeof profile, "DC field = %g*exp(-((z-%g)/%g)^%g) kV/cm", c::dc_sg_amp, c::dc_sg_z0,
                    c::dc_width, c::dc_sg_order);
    else if constexpr (c::dc_profile == 4)
      std::snprintf(profile, sizeof profile, "DC field = %g kV/cm for %g <= z < %g cm, 0 elsewhere",
                    static_cast<double>(4.f / 0.4f), c::dc_win_begin, c::dc_win_end);
    else
      std::snprintf(profile, sizeof profile, "DC field = %g*sin(z*2*pi*%g/%g) kV/cm", c::dc_sin_amp,
                    c::dc_sin_cycles, c::dc_sin_span);
    std::printf(" %s\n", profile);
    file_line(profile);
  }
  file_line(" ");
  file_line("All is normalized to:");
  both("I0      = ", ph.I0 * 1e-9, " GW/cm^2");
  both("a0      = ", c::a0, " cm");
  file_value("k0      = ", c::k0, " cm^(-1)");
  both("Ldifr   = ", c::k0 * c::a0 * c::a0, " cm");
  file_line(" ");
  file_line("Model parameters:");
  both("ng      = ", ng, "");
  both("Ldisp   = ", c::cL * c::t0 / d2kdw2, " cm");
  file_value("n2      = ", c::n2, " cm^2/W");
  both("zself   = ", ph.zself, " cm");
  both("P/Pcr   = ", ph.P_Pcr, "");
  file_line("Paralleled:");
  const float memory = 1.f * sizes.nR_THz * tf.ntT * 16 + 1.f * sizes.nR_opt * tf.ntO * 16;
  std::printf(" Paralleling by %d\n", nthreads);
  std::printf(" Memory: %.9g\n", static_cast<double>(memory));
  f.text("Paralleling by ").i(nthreads).nl();
  f.text("Memory:").f(memory).nl();

  std::string kw = kw_z_cm.empty() ? std::string("every THz snapshot") : std::string();
  for (std::size_t k = 0; k < kw_z_cm.size(); ++k) {
    char v[32];
    std::snprintf(v, sizeof v, "%s%g", k ? "," : "", kw_z_cm[k]);
    kw += v;
    if (k + 1 == kw_z_cm.size()) kw += " cm";
  }
  kw += kw_format == KwFormat::Binary ? " (bin, .kwb)" : " (text, .dat)";
  std::printf(" kw_THz: %s\n", kw.c_str());
  f.text("kw_THz: ").text(kw.c_str()).nl();
}

}  // namespace uppe
