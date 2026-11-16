#include <algorithm>
#include <cmath>

#include "check.hpp"
#include "physics.hpp"

using namespace uppe;

namespace {

struct Point {
  double z_cm, kV;
};

// f(z) = 20*(1/(1+exp(-5(z-20.5))) - 1/2)*exp(-((z-20.5)/3)^10), evaluated in Python (double precision).
constexpr Point kWindow[] = {
    {16.0, -9.044649387858707e-25}, {17.0, -0.09357059294733021}, {18.0, -8.508554394746366},
    {19.0, -9.979194360269044},     {20.5, 0.0},                  {20.6, 2.449186624037119},
    {22.0, 9.979194360269044},      {23.5, 3.678792161011617},    {24.5, 1.9405163666159681e-07},
    {25.0, 9.044649387858707e-25},
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
  } else {
    for (const Point& p : kWindow) CHECK_CLOSE(dc_field_kV(p.z_cm), p.kV, 1e-13 * std::max(1.0, std::fabs(p.kV)));
    CHECK(dc_field_kV(c::dc_z0) == 0.0);
    // Odd about z0: the field changes sign there.
    for (double d = 0.01; d < 5.0; d += 0.037) CHECK_CLOSE(dc_field_kV(c::dc_z0 + d), -dc_field_kV(c::dc_z0 - d), 1e-13);
    // Plateaus of about -+10 kV/cm inside the window, nothing far outside it.
    for (double z = 18.9; z < 19.45; z += 0.1) CHECK(dc_field_kV(z) < -9.9 && dc_field_kV(z) > -10.0);
    for (double z = 21.6; z < 22.15; z += 0.1) CHECK(dc_field_kV(z) > 9.9 && dc_field_kV(z) < 10.0);
    CHECK(std::fabs(dc_field_kV(0.0)) == 0.0);
    CHECK(std::fabs(dc_field_kV(40.0)) == 0.0);
    for (const Point& p : kWindow) CHECK(dc_field(ph, p.z_cm) == dc_field_kV(p.z_cm) * ph.DC_norm);
  }
  // 10 kV/cm in field units, as DC_Field = DC_Field_in_kV*sqrt(31000/5.15/5.15/I0) of the Fortran code.
  CHECK(10.0 * ph.DC_norm == static_cast<double>(4.f / 0.4f) * std::sqrt(static_cast<double>(31000 / 5.15f / 5.15f) / ph.I0));
  return test::finish("test_dc_profile");
}
