#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "common.hpp"
#include "grids.hpp"

namespace uppe {

// res/kw_THz/Z<pos>.kwb (Binary) or the Fortran-like text Z<pos>.dat (Text).
enum class KwFormat { Binary = 0, Text = 1 };

// Binary kw_THz file, little-endian:
//   char[8] "UPPEKW01"; int32 nr, nf; double z_cm, t0, a0, cL;
//   double f_THz[nf]; double k_cm[nf]; double kr_cm[nr]; float E2[nr][nf]
// f_THz = frT, k_cm = freqs/(cL*t0), kr_cm = kr/a0, so sin(angle) = kr_cm/k_cm as in the text file;
// E2[i][q-1] = |E_THz(q, kr_i)|^2 for q = 1..nqT-1.
struct KwData {
  int nr = 0, nf = 0;
  double z_cm = 0, t0 = 0, a0 = 0, cL = 0;
  std::vector<double> f_THz, k_cm, kr_cm;
  std::vector<float> e2;
};

const char* kw_extension(KwFormat format);

// FNV-1a over the target list and the format; stored in checkpoints so a resume keeps the kw schedule.
std::uint64_t kw_schedule_hash(const std::vector<double>& z_cm, KwFormat format);

// Both writers write <file>.tmp and rename it, so a file with the final name is complete.
// E_T is E_THz(w,kr) with rows of nqT bins for the nrT points of gT.
void write_kw_binary(const std::filesystem::path& file, const cplx* E_T, int nrT, int nqT, const RadialGrid& gT,
                     const Band& bandT, double z_cm);
void write_kw_text(const std::filesystem::path& file, const cplx* E_T, int nrT, int nqT, const RadialGrid& gT,
                   const Band& bandT);

KwData read_kw_binary(const std::filesystem::path& file);

}  // namespace uppe
