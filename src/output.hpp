#pragma once

#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

#include "common.hpp"

namespace uppe {

std::string path_string(const std::filesystem::path& p);

// One record per line, every item preceded by a blank, like Fortran list-directed output.
// Numbers are printed with the shortest representation that round-trips exactly.
class LineBuffer {
 public:
  LineBuffer& text(const char* s);
  LineBuffer& i(long long v);
  LineBuffer& d(double v);
  LineBuffer& f(float v);
  LineBuffer& nl() {
    s_.push_back('\n');
    return *this;
  }
  void clear() { s_.clear(); }
  const std::string& str() const { return s_; }
  std::size_t size() const { return s_.size(); }

 private:
  std::string s_;
};

class TextWriter {
 public:
  TextWriter() = default;
  explicit TextWriter(const std::filesystem::path& file, bool append = false);
  ~TextWriter();
  TextWriter(TextWriter&& o) noexcept;
  TextWriter& operator=(TextWriter&& o) noexcept;
  TextWriter(const TextWriter&) = delete;
  TextWriter& operator=(const TextWriter&) = delete;

  TextWriter& text(const char* s) {
    buf_.text(s);
    return *this;
  }
  TextWriter& i(long long v) {
    buf_.i(v);
    return *this;
  }
  TextWriter& d(double v) {
    buf_.d(v);
    return *this;
  }
  TextWriter& f(float v) {
    buf_.f(v);
    return *this;
  }
  TextWriter& nl() {
    buf_.nl();
    if (buf_.size() >= kFlushBytes) drain();
    return *this;
  }

  void write(const std::string& s);
  void flush();  // pushes everything to the OS, so a killed run keeps its data
  void close();
  bool is_open() const { return fp_ != nullptr; }

 private:
  void drain();
  static constexpr std::size_t kFlushBytes = std::size_t(1) << 20;
  std::FILE* fp_ = nullptr;
  std::string name_;
  LineBuffer buf_;
};

// Formats rows in parallel (fn(row, LineBuffer&) must be thread-safe) and writes them in order.
template <class RowFn>
void write_rows_parallel(TextWriter& w, int nrows, RowFn&& fn) {
  const int chunk = 64;
  std::vector<LineBuffer> bufs(chunk);
  for (int r0 = 0; r0 < nrows; r0 += chunk) {
    const int r1 = std::min(nrows, r0 + chunk);
#pragma omp parallel for schedule(dynamic, 1)
    for (int r = r0; r < r1; ++r) {
      bufs[r - r0].clear();
      fn(r, bufs[r - r0]);
    }
    for (int r = r0; r < r1; ++r) w.write(bufs[r - r0].str());
  }
}

}  // namespace uppe
