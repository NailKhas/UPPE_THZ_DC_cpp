#include <algorithm>
#include <cmath>

#include "check.hpp"
#include "physics.hpp"

using namespace uppe;

namespace {

struct Point {
  double z_cm, kV;
};

// f(z) = 20*(1/(1+exp(-5(z-20.373))) - 1/2)*exp(-((z-20.373)/3)^10), evaluated in Python (double precision).
constexpr Point kWindow[] = {
    {16.0, -1.5522583513722288e-18}, {17.0, -0.39631581296959195}, {18.0, -9.085528759845214},
    {19.0, -9.97512558872788},       {20.373, 0.0},                {20.6, 5.135209012364624},
    {22.0, 9.972164655195016},       {23.5, 2.200737566157395},    {24.5, 2.8718331879674543e-10},
    {25.0, 8.315802409840995e-33},
};

// f(z) = 10*exp(-((z-20.5)/3)^8), evaluated in Python (double precision).
constexpr Point kSuperGauss[] = {
    {16.0, 7.404699497933558e-11}, {17.0, 0.3231532598302924}, {18.0, 7.9249582660242766},
    {19.0, 9.961013694701176},     {20.5, 10.0},               {22.0, 9.961013694701176},
    {23.0, 7.9249582660242766},    {24.0, 0.3231532598302924}, {25.0, 7.404699497933558e-11},
};

}  // namespace

int main() {
  test::init();
  const Physics ph = make_physics();

  if constexpr (c::dc_profile == 0) {
    for (double z : {0.0, 18.0, 20.5, 24.0}) {
      CHECK(dc_field_kV(z) == static_cast<double>(4.f / 0.4f));
      CHECK(dc_field(ph, z) == static_cast<double>(4.f / 0.4f) * ph.DC_norm);
    }
  } else if constexpr (c::dc_profile == 2) {
    for (const Point& p : kSuperGauss)
      CHECK_CLOSE(dc_field_kV(p.z_cm), p.kV, 1e-13 * std::max(1.0, std::fabs(p.kV)));
    CHECK(dc_field_kV(c::dc_sg_z0) == 10.0);
    // Even about z0, positive everywhere, at most 10 kV/cm.
    for (double d = 0.01; d < 5.0; d += 0.037) {
      CHECK_CLOSE(dc_field_kV(c::dc_sg_z0 + d), dc_field_kV(c::dc_sg_z0 - d), 1e-13);
      CHECK(dc_field_kV(c::dc_sg_z0 + d) >= 0.0 && dc_field_kV(c::dc_sg_z0 + d) <= 10.0);
    }
    CHECK(dc_field_kV(0.0) == 0.0);
    CHECK(dc_field_kV(40.0) == 0.0);
    for (const Point& p : kSuperGauss) CHECK(dc_field(ph, p.z_cm) == dc_field_kV(p.z_cm) * ph.DC_norm);
  } else {
    for (const Point& p : kWindow) CHECK_CLOSE(dc_field_kV(p.z_cm), p.kV, 1e-13 * std::max(1.0, std::fabs(p.kV)));
    CHECK(dc_field_kV(c::dc_z0) == 0.0);
    // Odd about z0: the field changes sign there.
    for (double d = 0.01; d < 5.0; d += 0.037) CHECK_CLOSE(dc_field_kV(c::dc_z0 + d), -dc_field_kV(c::dc_z0 - d), 1e-13);
    // Plateaus of about -+10 kV/cm inside the window, nothing far outside it.
    for (double d = -1.6; d < -1.05; d += 0.1) CHECK(dc_field_kV(c::dc_z0 + d) < -9.9 && dc_field_kV(c::dc_z0 + d) > -10.0);
    for (double d = 1.1; d < 1.65; d += 0.1) CHECK(dc_field_kV(c::dc_z0 + d) > 9.9 && dc_field_kV(c::dc_z0 + d) < 10.0);
    CHECK(std::fabs(dc_field_kV(0.0)) == 0.0);
    CHECK(std::fabs(dc_field_kV(40.0)) == 0.0);
    for (const Point& p : kWindow) CHECK(dc_field(ph, p.z_cm) == dc_field_kV(p.z_cm) * ph.DC_norm);
  }
  // 10 kV/cm in field units, as DC_Field = DC_Field_in_kV*sqrt(31000/5.15/5.15/I0) of the Fortran code.
  CHECK(10.0 * ph.DC_norm == static_cast<double>(4.f / 0.4f) * std::sqrt(static_cast<double>(31000 / 5.15f / 5.15f) / ph.I0));
  return test::finish("test_dc_profile");
}
