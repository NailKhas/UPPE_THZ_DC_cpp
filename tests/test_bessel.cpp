#include <cmath>
#include <vector>

#include "bessel.hpp"
#include "check.hpp"
#include "constants.hpp"
#include "grids.hpp"

using namespace uppe;

// Root search of PrepareHankel: scan with step Rmax/16000, Newton from the left grid point.
static std::vector<double> fortran_scan_roots(int count) {
  const double hh = c::Rmax / 16000.0;
  std::vector<double> roots;
  double rrr = 0.0;
  while (static_cast<int>(roots.size()) < count) {
    rrr += hh;
    if (bessel_j0(rrr) * bessel_j0(rrr + hh) < 0.0) {
      double r1 = rrr;
      double r2 = r1 - bessel_j0(r1) / (-bessel_j1(r1));
      while (std::fabs(r2 - r1) > c::err) {
        r1 = r2;
        r2 = r1 - bessel_j0(r1) / (-bessel_j1(r1));
      }
      roots.push_back((r1 + r2) / 2.0);
    }
  }
  return roots;
}

int main() {
  test::init();

  CHECK_CLOSE(bessel_j0(1.0), 0.765197686557966551, 1e-15);
  CHECK_CLOSE(bessel_j1(1.0), 0.440050585744933516, 1e-15);
  CHECK_CLOSE(bessel_j0(0.0), 1.0, 0.0);
  CHECK_CLOSE(bessel_j1(0.0), 0.0, 0.0);

  const int count = 16000;
  const std::vector<double> roots = bessel_j0_roots(count);
  CHECK(static_cast<int>(roots.size()) == count);

  const double known[] = {2.404825557695773, 5.520078110286311, 8.653727912911012, 11.79153443901428,
                          14.93091770848779};
  for (int k = 0; k < 5; ++k) CHECK_CLOSE(roots[k], known[k], 1e-12);
  CHECK_CLOSE(roots[9], 30.634606468431975, 1e-11);

  double maxJ0 = 0.0, minGap = 1e300, maxGap = 0.0;
  for (int n = 0; n < count; ++n) {
    maxJ0 = std::max(maxJ0, std::fabs(bessel_j0(roots[n])));
    if (n > 0) {
      minGap = std::min(minGap, roots[n] - roots[n - 1]);
      maxGap = std::max(maxGap, roots[n] - roots[n - 1]);
    }
  }
  CHECK(maxJ0 < 1e-12);
  CHECK(minGap > 3.1);
  CHECK(maxGap < 3.2);

  const int nscan = 300;
  const std::vector<double> scan = fortran_scan_roots(nscan);
  double maxDiff = 0.0;
  for (int n = 0; n < nscan; ++n) maxDiff = std::max(maxDiff, std::fabs(scan[n] - roots[n]));
  CHECK(maxDiff < 1e-11);

  return test::finish("test_bessel");
}
