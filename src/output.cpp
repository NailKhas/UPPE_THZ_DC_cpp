#include "output.hpp"

#include <charconv>
#include <stdexcept>
#include <utility>

#if defined(__cpp_lib_to_chars) && __cpp_lib_to_chars >= 201611L
#define UPPE_FP_TO_CHARS 1
#else
#define UPPE_FP_TO_CHARS 0
#endif

namespace uppe {

std::string path_string(const std::filesystem::path& p) { return p.u8string(); }

LineBuffer& LineBuffer::text(const char* s) {
  s_.push_back(' ');
  s_.append(s);
  return *this;
}

LineBuffer& LineBuffer::i(long long v) {
  char buf[32];
  buf[0] = ' ';
  const auto r = std::to_chars(buf + 1, buf + sizeof buf, v);
  s_.append(buf, r.ptr);
  return *this;
}

LineBuffer& LineBuffer::d(double v) {
  char buf[64];
  buf[0] = ' ';
#if UPPE_FP_TO_CHARS
  const auto r = std::to_chars(buf + 1, buf + sizeof buf, v);
  s_.append(buf, r.ptr);
#else
  const int n = std::snprintf(buf + 1, sizeof buf - 1, "%.17g", v);
  s_.append(buf, buf + 1 + n);
#endif
  return *this;
}

LineBuffer& LineBuffer::f(float v) {
  char buf[48];
  buf[0] = ' ';
#if UPPE_FP_TO_CHARS
  const auto r = std::to_chars(buf + 1, buf + sizeof buf, v);
  s_.append(buf, r.ptr);
#else
  const int n = std::snprintf(buf + 1, sizeof buf - 1, "%.9g", static_cast<double>(v));
  s_.append(buf, buf + 1 + n);
#endif
  return *this;
}

TextWriter::TextWriter(const std::filesystem::path& file, bool append) : name_(path_string(file)) {
#if defined(_WIN32)
  fp_ = _wfopen(file.c_str(), append ? L"ab" : L"wb");
#else
  fp_ = std::fopen(file.c_str(), append ? "ab" : "wb");
#endif
  if (!fp_) throw std::runtime_error("cannot open file for writing: " + name_);
}

TextWriter::~TextWriter() {
  try {
    close();
  } catch (...) {
  }
}

TextWriter::TextWriter(TextWriter&& o) noexcept
    : fp_(o.fp_), name_(std::move(o.name_)), buf_(std::move(o.buf_)) {
  o.fp_ = nullptr;
}

TextWriter& TextWriter::operator=(TextWriter&& o) noexcept {
  if (this != &o) {
    try {
      close();
    } catch (...) {
    }
    fp_ = o.fp_;
    name_ = std::move(o.name_);
    buf_ = std::move(o.buf_);
    o.fp_ = nullptr;
  }
  return *this;
}

void TextWriter::drain() {
  if (!fp_ || buf_.size() == 0) return;
  const std::string& s = buf_.str();
  if (std::fwrite(s.data(), 1, s.size(), fp_) != s.size()) throw std::runtime_error("write failed: " + name_);
  buf_.clear();
}

void TextWriter::write(const std::string& s) {
  if (!fp_) throw std::runtime_error("write to a closed file: " + name_);
  drain();
  if (!s.empty() && std::fwrite(s.data(), 1, s.size(), fp_) != s.size())
    throw std::runtime_error("write failed: " + name_);
}

void TextWriter::flush() {
  drain();
  if (fp_ && std::fflush(fp_) != 0) throw std::runtime_error("flush failed: " + name_);
}

void TextWriter::close() {
  if (!fp_) return;
  std::FILE* fp = fp_;
  try {
    drain();
  } catch (...) {
    std::fclose(fp);
    fp_ = nullptr;
    throw;
  }
  fp_ = nullptr;
  if (std::fclose(fp) != 0) throw std::runtime_error("close failed: " + name_);
}

}  // namespace uppe
