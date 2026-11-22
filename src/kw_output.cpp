#include "kw_output.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>

#include "constants.hpp"
#include "output.hpp"

namespace uppe {

namespace {

namespace fs = std::filesystem;

constexpr char kKwMagic[8] = {'U', 'P', 'P', 'E', 'K', 'W', '0', '1'};

struct KwHeader {
  char magic[8];
  std::int32_t nr, nf;
  double z_cm, t0, a0, cL;
};
static_assert(sizeof(KwHeader) == 48, "KwHeader must have no padding");

bool little_endian() {
  const std::uint32_t one = 1;
  unsigned char b;
  std::memcpy(&b, &one, 1);
  return b == 1;
}

class RawFile {
 public:
  RawFile(const fs::path& p, bool write) : name_(path_string(p)) {
#if defined(_WIN32)
    fp_ = _wfopen(p.c_str(), write ? L"wb" : L"rb");
#else
    fp_ = std::fopen(p.c_str(), write ? "wb" : "rb");
#endif
    if (!fp_) throw std::runtime_error(std::string("cannot open ") + (write ? "for writing: " : "for reading: ") + name_);
  }
  ~RawFile() {
    if (fp_) std::fclose(fp_);
  }
  RawFile(const RawFile&) = delete;
  RawFile& operator=(const RawFile&) = delete;

  void write(const void* p, std::size_t n) {
    if (n && std::fwrite(p, 1, n, fp_) != n) throw std::runtime_error("write failed: " + name_);
  }
  void read(void* p, std::size_t n) {
    if (n && std::fread(p, 1, n, fp_) != n) throw std::runtime_error("kw_THz file is truncated: " + name_);
  }
  bool at_end() { return std::fgetc(fp_) == EOF; }
  void close() {
    std::FILE* fp = fp_;
    fp_ = nullptr;
    if (std::fclose(fp) != 0) throw std::runtime_error("close failed: " + name_);
  }

 private:
  std::FILE* fp_ = nullptr;
  std::string name_;
};

fs::path tmp_path(const fs::path& file) {
  fs::path tmp = file;
  tmp += ".tmp";
  return tmp;
}

}  // namespace

const char* kw_extension(KwFormat format) { return format == KwFormat::Binary ? ".kwb" : ".dat"; }

std::uint64_t kw_schedule_hash(const std::vector<double>& z_cm, KwFormat format) {
  constexpr std::uint64_t prime = 0x100000001b3ULL;
  std::uint64_t h = 0xcbf29ce484222325ULL;
  h = (h ^ static_cast<std::uint64_t>(format)) * prime;
  h = (h ^ static_cast<std::uint64_t>(z_cm.size())) * prime;
  for (double z : z_cm) {
    std::uint64_t w;
    std::memcpy(&w, &z, sizeof w);
    h = (h ^ w) * prime;
  }
  return h;
}

void write_kw_binary(const fs::path& file, const cplx* E_T, int nrT, int nqT, const RadialGrid& gT, const Band& bandT,
                     double z_cm) {
  if (!little_endian()) throw std::runtime_error("binary kw_THz output assumes a little-endian machine");
  if (gT.size() != nrT || bandT.nq() != nqT) throw std::invalid_argument("write_kw_binary: grid sizes do not match");
  const int nf = nqT - 1;

  KwHeader h{};
  std::memcpy(h.magic, kKwMagic, sizeof kKwMagic);
  h.nr = nrT;
  h.nf = nf;
  h.z_cm = z_cm;
  h.t0 = c::t0;
  h.a0 = c::a0;
  h.cL = c::cL;

  std::vector<double> f(nf), k(nf), kr(nrT);
  for (int q = 1; q < nqT; ++q) {
    f[q - 1] = static_cast<double>(bandT.fr[nqT + q]);
    k[q - 1] = bandT.freqs[nqT + q] / (c::cL * c::t0);
  }
  for (int i = 0; i < nrT; ++i) kr[i] = gT.kr[i] / c::a0;

  const fs::path tmp = tmp_path(file);
  {
    RawFile out(tmp, true);
    out.write(&h, sizeof h);
    out.write(f.data(), sizeof(double) * f.size());
    out.write(k.data(), sizeof(double) * k.size());
    out.write(kr.data(), sizeof(double) * kr.size());

    const int chunk = 64;
    std::vector<float> block(static_cast<std::size_t>(chunk) * nf);
    for (int i0 = 0; i0 < nrT; i0 += chunk) {
      const int i1 = std::min(nrT, i0 + chunk);
#pragma omp parallel for schedule(static)
      for (int i = i0; i < i1; ++i) {
        const cplx* row = E_T + static_cast<std::size_t>(i) * nqT;
        float* dst = block.data() + static_cast<std::size_t>(i - i0) * nf;
        for (int q = 1; q < nqT; ++q) dst[q - 1] = static_cast<float>(abs2(row[q]));
      }
      out.write(block.data(), sizeof(float) * static_cast<std::size_t>(i1 - i0) * nf);
    }
    out.close();
  }
  fs::rename(tmp, file);
}

void write_kw_text(const fs::path& file, const cplx* E_T, int nrT, int nqT, const RadialGrid& gT, const Band& bandT) {
  const fs::path tmp = tmp_path(file);
  {
    TextWriter w(tmp);
    write_rows_parallel(w, nrT, [&](int i, LineBuffer& b) {
      const cplx* row = E_T + static_cast<std::size_t>(i) * nqT;
      const double kr = gT.kr[i];
      for (int q = 1; q < nqT; ++q) {
        const int itt = nqT + q;
        const float angle = static_cast<float>(std::asin(kr / bandT.freqs[itt] * c::cL * c::t0 / c::a0));
        b.f(bandT.fr[itt]).f(angle).d(abs2(row[q])).nl();
      }
    });
    w.close();
  }
  fs::rename(tmp, file);
}

KwData read_kw_binary(const fs::path& file) {
  const std::string name = path_string(file);
  RawFile in(file, false);
  KwHeader h{};
  in.read(&h, sizeof h);
  if (std::memcmp(h.magic, kKwMagic, sizeof kKwMagic) != 0) throw std::runtime_error("not a kw_THz file: " + name);
  if (h.nr < 1 || h.nf < 1) throw std::runtime_error("bad kw_THz header: " + name);
  KwData d;
  d.nr = h.nr;
  d.nf = h.nf;
  d.z_cm = h.z_cm;
  d.t0 = h.t0;
  d.a0 = h.a0;
  d.cL = h.cL;
  d.f_THz.resize(d.nf);
  d.k_cm.resize(d.nf);
  d.kr_cm.resize(d.nr);
  d.e2.resize(static_cast<std::size_t>(d.nr) * d.nf);
  in.read(d.f_THz.data(), sizeof(double) * d.f_THz.size());
  in.read(d.k_cm.data(), sizeof(double) * d.k_cm.size());
  in.read(d.kr_cm.data(), sizeof(double) * d.kr_cm.size());
  in.read(d.e2.data(), sizeof(float) * d.e2.size());
  if (!in.at_end()) throw std::runtime_error("kw_THz file has trailing data: " + name);
  return d;
}

}  // namespace uppe
