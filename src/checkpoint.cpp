#include "checkpoint.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>

#include "output.hpp"

namespace uppe {

namespace {

namespace fs = std::filesystem;

constexpr char kMagic[8] = {'U', 'P', 'P', 'E', 'C', 'K', 'P', 'T'};
constexpr std::int32_t kVersion = 4;
constexpr std::uint64_t kHashSeed = 0xcbf29ce484222325ULL;

struct Header {
  char magic[8];
  std::int32_t version;
  std::int32_t nt_all, nR_first, nR_opt, nR_THz;
  std::int32_t reserved;
  std::int64_t iz, clamped_total;
  double z, z_flag, Imax;
  std::uint64_t nO, nT;
  double focus, zsic;
  double prelens, a0, Rmax, WW0;
};

// FNV-1a over 64-bit words: only detects truncated or damaged files.
std::uint64_t hash_bytes(const void* p, std::size_t n, std::uint64_t h) {
  constexpr std::uint64_t prime = 0x100000001b3ULL;
  const unsigned char* b = static_cast<const unsigned char*>(p);
  std::size_t i = 0;
  for (; i + 8 <= n; i += 8) {
    std::uint64_t w;
    std::memcpy(&w, b + i, 8);
    h = (h ^ w) * prime;
  }
  for (; i < n; ++i) h = (h ^ b[i]) * prime;
  return h;
}

class BinaryFile {
 public:
  BinaryFile(const fs::path& p, bool write) : name_(path_string(p)) {
#if defined(_WIN32)
    fp_ = _wfopen(p.c_str(), write ? L"wb" : L"rb");
#else
    fp_ = std::fopen(p.c_str(), write ? "wb" : "rb");
#endif
    if (!fp_)
      throw std::runtime_error(std::string("cannot open checkpoint for ") + (write ? "writing: " : "reading: ") + name_);
  }
  ~BinaryFile() {
    if (fp_) std::fclose(fp_);
  }
  BinaryFile(const BinaryFile&) = delete;
  BinaryFile& operator=(const BinaryFile&) = delete;

  void write(const void* p, std::size_t n) {
    if (n && std::fwrite(p, 1, n, fp_) != n) throw std::runtime_error("checkpoint write failed: " + name_);
  }
  void read(void* p, std::size_t n) {
    if (n && std::fread(p, 1, n, fp_) != n) throw std::runtime_error("checkpoint is truncated: " + name_);
  }
  void close() {
    std::FILE* fp = fp_;
    fp_ = nullptr;
    if (std::fclose(fp) != 0) throw std::runtime_error("checkpoint close failed: " + name_);
  }

 private:
  std::FILE* fp_ = nullptr;
  std::string name_;
};

}  // namespace

void write_checkpoint(const fs::path& file, const Sizes& sizes, const LoopState& state, const AlignedArray<cplx>& E_O,
                      const AlignedArray<cplx>& E_T) {
  Header h{};
  std::memcpy(h.magic, kMagic, sizeof kMagic);
  h.version = kVersion;
  h.nt_all = sizes.nt_all;
  h.nR_first = sizes.nR_first;
  h.nR_opt = sizes.nR_opt;
  h.nR_THz = sizes.nR_THz;
  h.iz = state.iz;
  h.clamped_total = state.clamped_total;
  h.z = state.z;
  h.z_flag = state.z_flag;
  h.Imax = state.Imax;
  h.nO = E_O.size();
  h.nT = E_T.size();
  h.focus = c::focus;
  h.zsic = c::zsic;
  h.prelens = c::prelens;
  h.a0 = c::a0;
  h.Rmax = c::Rmax;
  h.WW0 = c::WW0;

  const std::size_t bytesO = E_O.size() * sizeof(cplx);
  const std::size_t bytesT = E_T.size() * sizeof(cplx);
  std::uint64_t sum = hash_bytes(&h, sizeof h, kHashSeed);
  sum = hash_bytes(E_O.data(), bytesO, sum);
  sum = hash_bytes(E_T.data(), bytesT, sum);

  fs::path tmp = file;
  tmp += ".tmp";
  {
    BinaryFile f(tmp, true);
    f.write(&h, sizeof h);
    f.write(E_O.data(), bytesO);
    f.write(E_T.data(), bytesT);
    f.write(&sum, sizeof sum);
    f.close();
  }
  fs::rename(tmp, file);
}

LoopState read_checkpoint(const fs::path& file, const Sizes& sizes, AlignedArray<cplx>& E_O, AlignedArray<cplx>& E_T) {
  const std::string name = path_string(file);
  BinaryFile f(file, false);
  Header h{};
  f.read(&h, sizeof h);
  if (std::memcmp(h.magic, kMagic, sizeof kMagic) != 0) throw std::runtime_error("not a checkpoint file: " + name);
  if (h.version != kVersion) throw std::runtime_error("unsupported checkpoint version in " + name);
  if (h.nt_all != sizes.nt_all || h.nR_first != sizes.nR_first || h.nR_opt != sizes.nR_opt ||
      h.nR_THz != sizes.nR_THz || h.nO != E_O.size() || h.nT != E_T.size())
    throw std::runtime_error("checkpoint " + name + " was written for other grid sizes");
  if (h.focus != c::focus || h.zsic != c::zsic)
    throw std::runtime_error("checkpoint " + name + " was written for focus " + std::to_string(h.focus) +
                             " cm, zsic " + std::to_string(h.zsic) + " cm; this build has focus " +
                             std::to_string(c::focus) + " cm, zsic " + std::to_string(c::zsic) + " cm");
  if (h.prelens != c::prelens || h.a0 != c::a0 || h.Rmax != c::Rmax || h.WW0 != c::WW0)
    throw std::runtime_error("checkpoint " + name + " was written for prelens " + std::to_string(h.prelens) +
                             " cm, a0 " + std::to_string(h.a0) + " cm, Rmax " + std::to_string(h.Rmax) + ", W0 " +
                             std::to_string(h.WW0 * 1e3) + " mJ; this build has prelens " +
                             std::to_string(c::prelens) + " cm, a0 " + std::to_string(c::a0) + " cm, Rmax " +
                             std::to_string(c::Rmax) + ", W0 " + std::to_string(c::WW0 * 1e3) + " mJ");

  const std::size_t bytesO = E_O.size() * sizeof(cplx);
  const std::size_t bytesT = E_T.size() * sizeof(cplx);
  f.read(E_O.data(), bytesO);
  f.read(E_T.data(), bytesT);
  std::uint64_t stored = 0;
  f.read(&stored, sizeof stored);
  f.close();

  std::uint64_t sum = hash_bytes(&h, sizeof h, kHashSeed);
  sum = hash_bytes(E_O.data(), bytesO, sum);
  sum = hash_bytes(E_T.data(), bytesT, sum);
  if (sum != stored) throw std::runtime_error("checkpoint " + name + " is damaged (checksum mismatch)");

  LoopState s;
  s.iz = h.iz;
  s.clamped_total = h.clamped_total;
  s.z = h.z;
  s.z_flag = h.z_flag;
  s.Imax = h.Imax;
  return s;
}

}  // namespace uppe
