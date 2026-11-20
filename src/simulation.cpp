#include "simulation.hpp"

#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "checkpoint.hpp"
#include "common.hpp"
#include "fft.hpp"
#include "first_step.hpp"
#include "grids.hpp"
#include "hankel.hpp"
#include "nonlinear.hpp"
#include "output.hpp"
#include "physics.hpp"
#include "propagation.hpp"

namespace uppe {

namespace {

namespace fs = std::filesystem;

void validate(const Sizes& s, const TimeFreq& tf) {
  if (s.nt_all < 256 || s.nt_all % 64 != 0) throw std::invalid_argument("nt_all must be a multiple of 64 (>= 256)");
  if (s.nR_opt < 3) throw std::invalid_argument("nR_opt must be >= 3");
  if (s.nR_first < s.nR_opt) throw std::invalid_argument("nR_first must be >= nR_opt (E = Ebuf(:,1:nR_opt-1))");
  if (s.nR_THz < s.nR_opt) throw std::invalid_argument("nR_THz must be >= nR_opt (JefThread fills k_THz(:,1:nR_opt-1))");
  if (tf.opt_lin_start < 0 || tf.opt_lin_start >= tf.nqO) throw std::invalid_argument("optical band is too narrow");
}

void create_output_dirs(const fs::path& out) {
  for (const char* d : {"IS", "Fl", "kw_THz", "Or", "Ne", "IO"}) fs::create_directories(out / "res" / d);
}

struct Snapshot {
  TextWriter is, fl, kw_thz, orr, ne, io;

  void close() {
    is.close();
    fl.close();
    kw_thz.close();
    orr.close();
    ne.close();
    io.close();
  }
};

// The optical stage writes no kw_THz file (withKw = false).
std::unique_ptr<Snapshot> open_snapshot(const fs::path& out, int z_pos, bool withKw) {
  char name[32];
  std::snprintf(name, sizeof name, "Z%04d.dat", z_pos);
  const fs::path res = out / "res";
  auto s = std::make_unique<Snapshot>();
  s->is = TextWriter(res / "IS" / name);
  s->fl = TextWriter(res / "Fl" / name);
  if (withKw) s->kw_thz = TextWriter(res / "kw_THz" / name);
  s->orr = TextWriter(res / "Or" / name);
  s->ne = TextWriter(res / "Ne" / name);
  s->io = TextWriter(res / "IO" / name);
  return s;
}

// FurThread with Dir = -1: Et(t,r) from the (w,r) spectrum.
void spectra_to_time(const cplx* W, int nr, int nq, double* Et, int nt, FftPool& pool) {
#pragma omp parallel for schedule(static)
  for (int i = 0; i < nr; ++i) {
    RealFFT& f = *pool[thread_id()];
    centered_backward(f, W + static_cast<std::size_t>(i) * nq, nq);
    std::memcpy(Et + static_cast<std::size_t>(i) * nt, f.time(), sizeof(double) * static_cast<std::size_t>(nt));
  }
}

// acc[q] = sum_i |A(q,i)|^2 * r1(i) * dr1(i), summed over i in increasing order.
void radial_power(const cplx* A, int nr, int nq, const RadialGrid& g, std::vector<double>& acc) {
  acc.assign(nq, 0.0);
  const int chunk = 256;
  const int nchunks = (nq + chunk - 1) / chunk;
#pragma omp parallel for schedule(static)
  for (int ci = 0; ci < nchunks; ++ci) {
    const int q0 = ci * chunk;
    const int q1 = std::min(nq, q0 + chunk);
    for (int i = 0; i < nr; ++i) {
      const cplx* row = A + static_cast<std::size_t>(i) * nq;
      const double r = g.r1[i];
      const double dr = g.dr1[i];
      for (int q = q0; q < q1; ++q) acc[q] += abs2(row[q]) * r * dr;
    }
  }
}

// Per radial point: sum over t of Et^2 and max over t of Et^2.
void row_stats(const double* Et, int nr, int nt, std::vector<double>& sumsq, std::vector<double>& maxsq) {
#pragma omp parallel for schedule(static)
  for (int i = 0; i < nr; ++i) {
    const double* row = Et + static_cast<std::size_t>(i) * nt;
    double s = 0.0;
    double m = 0.0;
    for (int t = 0; t < nt; ++t) {
      const double v = row[t] * row[t];
      s += v;
      m = std::max(m, v);
    }
    sumsq[i] = s;
    maxsq[i] = m;
  }
}

void add_to(cplx* E, const cplx* K, std::size_t n) {
  const long long nn = static_cast<long long>(n);
  const int chunk = 1 << 16;
  const int nchunks = static_cast<int>((nn + chunk - 1) / chunk);
#pragma omp parallel for schedule(static)
  for (int ci = 0; ci < nchunks; ++ci) {
    const std::size_t b = static_cast<std::size_t>(ci) * chunk;
    const std::size_t e = std::min(n, b + chunk);
    for (std::size_t j = b; j < e; ++j) E[j] += K[j];
  }
}

}  // namespace

RunStatus run_simulation(const RunConfig& cfg) {
  const double wall_start = omp_get_wtime();
  const Sizes& sz = cfg.sizes;
  const TimeFreq tf = make_time_freq(sz.nt_all);
  validate(sz, tf);
  const int nthreads = max_threads();
  const fs::path& out = cfg.out_dir;
  create_output_dirs(out);
  // A resumed run continues the scalar files and does not rewrite what the first run already wrote.
  const bool resume = !cfg.resume_path.empty();

  // InitConsts
  Physics ph = make_physics();
  load_ionization_tables(ph, cfg.data_dir);
  // PARALLEL.f90 calls write_params before PrepareKZ_opt, so ng = 0 and Ldisp = Infinity are reported.
  if (!resume) write_params(ph, 0.0, 0.0, nthreads, sz, tf, out);

  const int ntO = tf.ntO;
  const int nqO = tf.nqO;
  const int nqT = tf.nqT;
  const int nrO = sz.nR_opt - 1;
  const int nrT = sz.nR_THz - 1;
  const std::size_t nO = static_cast<std::size_t>(nrO) * nqO;
  const std::size_t nT = static_cast<std::size_t>(nrT) * nqT;

  // PrepareHankel (roots) and the frequency part of PrepareKZ_opt
  const std::vector<double> roots = bessel_j0_roots(std::max(sz.nR_first, sz.nR_THz));
  const Band bandO = make_band(ntO, tf.dw, tf.wmin);
  const Dispersion disp = compute_dispersion(bandO, tf);

  TextWriter fImax(out / "res" / "Imax.dat", resume);
  TextWriter fNemax(out / "res" / "Nemax.dat", resume);
  TextWriter fEnergy(out / "res" / "Energy.dat", resume);
  TextWriter fDen(out / "res" / "Den.dat", resume);
  TextWriter fSnaps(out / "res" / "snapshots.txt", resume);  // z_pos of every completed snapshot

  const double k0a0a0 = c::k0 * c::a0 * c::a0;
  // The optical stage runs while z < z_opt_end (do while (z.lt.71.0/(k0*a0*a0)) in PARALLEL.f90).
  const double z_opt_end = c::opt_stage_end / k0a0a0;

  AlignedArray<cplx> E_O(nO);   // E(w,kr)
  AlignedArray<cplx> E_T(nT);   // E_THz(w,kr)
  double dz = c::zsic / c::Ldifr;
  double z = 0.0;
  double z_flag = 0.0;
  double Imax = 0.0;
  long long iz = 0;
  long long clamped_total = 0;
  if (resume) {
    const LoopState st = read_checkpoint(cfg.resume_path, sz, E_O, E_T);
    iz = st.iz;
    clamped_total = st.clamped_total;
    z = st.z;
    z_flag = st.z_flag;
    Imax = st.Imax;
    if (st.stage != (z < z_opt_end ? 0 : 1))
      throw std::runtime_error("checkpoint " + path_string(cfg.resume_path) + ": stage " + std::to_string(st.stage) +
                               " does not match z = " + std::to_string(z * c::Ldifr) + " cm");
    std::printf(" --- Resumed from %s at step %lld, z = %.15g cm (%s stage) ---\n",
                path_string(cfg.resume_path).c_str(), iz, z * c::Ldifr, st.stage == 0 ? "optical" : "THz");
    std::fflush(stdout);
  } else {
    std::printf(" --- Incoming ---\n");
    std::fflush(stdout);
    {
      const RadialGrid g1 = make_radial(roots, sz.nR_first, sz.nR_opt, RadialKind::First);
      const HankelMatrix H1 = make_hankel(roots, sz.nR_first);
      first_step(tf, bandO, g1, H1, disp.vg, dz, nrO, E_O.data(), out);
    }
    z = z + dz;
    z_flag = c::snap_start / k0a0a0;
    Imax = sq(c::a00 / c::a0);
  }

  // Prepare_R_opt and the second PrepareKZ_opt
  const RadialGrid gO = make_radial(roots, sz.nR_opt, sz.nR_opt, RadialKind::Optical);
  if (!resume) write_radial_files(out, gO, "rO1_um.dat", "rO1.dat", "krO1.dat");
  const HankelMatrix HO = make_hankel(roots, sz.nR_opt);
  const Propagator PO = make_propagator(bandO, gO, disp.vg, c::omega_max);
  if (!resume) {
    write_band_files(out, bandO, PO, BandKind::Optical);
    write_collision_file(out, tf);
  }

  AlignedArray<cplx> W_O(nO);   // Et before FurThread: E(w,r)
  AlignedArray<cplx> K_O(nO);   // k(w,kr)
  AlignedArray<cplx> Kr_O(nO);  // k(w,r) from JefThread

  if (!resume) {
    // E = Ebuf(:,1:nR-1); call ParallelThis(HnkThread,E)
    hankel_spectral(HO, HankelInput{E_O.data(), nullptr, 0.0, nrO}, W_O.data(), nrO, nqO, nullptr, nullptr);
    std::swap(E_O, W_O);
    std::printf(" E(w,kr) is obtained on new grid\n");
  }

  // PrepareKZ
  const RadialGrid gT = make_radial(roots, sz.nR_THz, sz.nR_opt, RadialKind::THz);
  if (!resume) write_radial_files(out, gT, "r1T_um.dat", "r1T.dat", "krT.dat");
  const HankelMatrix HT = make_hankel(roots, sz.nR_THz);
  const Band bandT = make_band(tf.ntT, tf.dw_THz, tf.wmin_THz);
  const Propagator PT = make_propagator(bandT, gT, disp.vg, c::omega_max_THz);
  if (!resume) write_band_files(out, bandT, PT, BandKind::THz);

  AlignedArray<cplx> K_T(nT);   // k_THz(w,kr)
  AlignedArray<cplx> B_T(nT);   // Ebuf(w,r); rows >= nR_opt-1 are valid only on snapshot steps
  AlignedArray<cplx> Kr_T(static_cast<std::size_t>(nrO) * nqT);  // k_THz(w,r) from JefThread
  AlignedArray<double> Et_O(static_cast<std::size_t>(nrO) * ntO);  // Et(t,r)
  std::vector<double> neMax(nrO, 0.0), Fl(nrO, 0.0), sumsq(nrO, 0.0), maxsq(nrO, 0.0), intSp;
  FftPool fftO = make_fft_pool(ntO, nthreads);
  NonlinearWorkspace nlws(tf, nthreads);
  NonlinearOptWorkspace nlwsO(tf, nthreads);

  NonlinearInputs nl;
  nl.tf = &tf;
  nl.ph = &ph;
  nl.thz_r = B_T.data();
  nl.et = Et_O.data();
  nl.nr = nrO;
  nl.k_opt = Kr_O.data();
  nl.k_thz = Kr_T.data();
  nl.ne_max = neMax.data();

  auto count_clamped = [&](long long cnt) {
    if (cnt > 0 && clamped_total == 0)
      std::printf(" WARNING: I0*|E|^2 >= 1e15 W/cm^2 is beyond the N2.dat/O2.dat tables; "
                  "the last table entry is used (the Fortran code reads past the array)\n");
    clamped_total += cnt;
  };

  // JefThread_THz; z_stage (in units of Ldifr) is where this RK stage samples the external field.
  auto run_nonlinear = [&](double step, double z_stage) {
    nl.dz = step;
    nl.dc = dc_field(ph, z_stage * c::Ldifr);
    count_clamped(nonlinear_response(nl, nlws));
  };

  // JefThread: the optical stage has no DC field (DC_Field = 0d0 before the loop in PARALLEL.f90).
  auto run_nonlinear_opt = [&](double step) {
    nl.dz = step;
    nl.dc = 0.0;
    count_clamped(nonlinear_response_opt(nl, nlwsO));
  };

  // HnkThread(k), KdivKZ_opt
  auto to_kr_and_divide_opt = [&]() {
    hankel_spectral(HO, HankelInput{Kr_O.data(), nullptr, 0.0, nrO}, K_O.data(), nrO, nqO, nullptr, nullptr);
    kdivkz(K_O.data(), PO, gO, bandO);
  };

  // HnkThread(k), HnkThread(k_THz), KdivKZ
  auto to_kr_and_divide = [&]() {
    hankel_spectral(HO, HankelInput{Kr_O.data(), nullptr, 0.0, nrO}, K_O.data(), nrO, nqO, nullptr, nullptr);
    hankel_spectral(HT, HankelInput{Kr_T.data(), nullptr, 0.0, nrO}, K_T.data(), nrT, nqT, nullptr, PT.cut.data());
    kdivkz(K_T.data(), PT, gT, bandT);
    kdivkz(K_O.data(), PO, gO, bandO);
  };

  // Et = E; HnkThread(E, Et); FurThread(Et, Dir=-1), with the optical rows of IS and IO, Fl, Or,
  // Imax and Energy. Both stages do the same.
  auto optical_diagnostics = [&](Snapshot* snap) {
    hankel_spectral(HO, HankelInput{E_O.data(), nullptr, 0.0, nrO}, W_O.data(), nrO, nqO, nullptr, nullptr);
    if (snap) {
      radial_power(W_O.data(), nrO, nqO, gO, intSp);
      for (int p = tf.opt_lin_start; p < nqO; ++p) {
        const int itt = nqO + p;
        snap->is.f(bandO.fr[itt]).d(2.0 * intSp[p]).nl();
        snap->io.f(bandO.fr[itt]).d(W_O[p].real()).d(W_O[p].imag()).nl();
      }
    }

    spectra_to_time(W_O.data(), nrO, nqO, Et_O.data(), ntO, fftO);

    row_stats(Et_O.data(), nrO, ntO, sumsq, maxsq);
    for (int i = 0; i < nrO; ++i) {
      Fl[i] = tf.dt * sumsq[i];
      if (snap) snap->fl.d(gO.r1[i]).d(Fl[i]).nl();
    }
    if (snap) {
      for (int t = 0; t < ntO; ++t) snap->orr.d((t * tf.dt + tf.tmin) * c::t0 * 1e15).d(Et_O[t]).d(0.0).nl();
    }

    const double ImaxO = maxsq[0];
    Imax = 0.0;
    for (int i = 0; i < nrO; ++i) Imax = std::max(Imax, maxsq[i]);
    std::printf(" %14.7g Imax = %.15g\n", static_cast<double>(static_cast<float>(z * c::Ldifr)), Imax);
    fImax.d(z * c::Ldifr).d(Imax).d(ImaxO).nl();
    fImax.flush();

    double flsum = 0.0;
    for (int i = 0; i < nrO; ++i) flsum += Fl[i] * gO.r1[i] * gO.dr1[i];
    const double Energy = 2.0 * c::pi * flsum;
    std::printf(" %14.7g Energy = %.15g\n", static_cast<double>(static_cast<float>(z * c::Ldifr)), Energy);
    fEnergy.d(z * c::Ldifr).d(Energy).nl();
    fEnergy.flush();
  };

  // Nemax and Den after the first RK stage, Ne(r) into the snapshot, which is then closed.
  auto plasma_diagnostics = [&](std::unique_ptr<Snapshot>& snap, int z_pos, double dc_kV) {
    double Nemax = 0.0;
    for (int i = 0; i < nrO; ++i) Nemax = std::max(Nemax, neMax[i]);
    std::printf(" %14.7g Ne_max = %.15g\n", static_cast<double>(static_cast<float>(z * c::Ldifr)), Nemax);
    fNemax.d(z * c::Ldifr).d(Nemax).nl();
    fNemax.flush();
    // Den = sum (Ne/N0)*(r/a0)*d(r/a0) over the optical nodes, without 2*pi.
    double Den = 0.0;
    for (int i = 0; i < nrO; ++i) Den += neMax[i] * gO.r1[i] * gO.dr1[i];
    std::printf(" %14.7g Den = %.15g\n", static_cast<double>(static_cast<float>(z * c::Ldifr)), Den);
    fDen.d(z * c::Ldifr).d(Den).nl();
    fDen.flush();
    std::printf(" %14.7g DC = %.15g kV/cm\n", static_cast<double>(static_cast<float>(z * c::Ldifr)), dc_kV);
    std::printf(" ----------------------------------------------------------- \n");

    if (snap) {
      for (int i = 0; i < nrO; ++i) snap->ne.d(gO.r1[i]).d(neMax[i]).nl();
      snap->close();
      snap.reset();
      fSnaps.i(z_pos).nl();
      fSnaps.flush();
    }
  };

  RunStatus status = RunStatus::Stopped;
  double max_step_time = 0.0;
  // After every z step of either stage: true when the run has to stop (status Finished at z_max).
  auto end_of_step = [&](double wall0) -> bool {
    const double step_time = omp_get_wtime() - wall0;
    std::printf(" step time %.3f s\n", step_time);
    std::fflush(stdout);
    max_step_time = std::max(max_step_time, step_time);

    if (cfg.z_max_cm && z * c::Ldifr >= *cfg.z_max_cm) {
      status = RunStatus::Finished;
      return true;
    }
    if (cfg.max_steps >= 0 && iz >= cfg.max_steps) return true;
    // The next step may be a snapshot step, which is slower than the slowest one seen so far.
    if (cfg.wall_limit_s && omp_get_wtime() - wall_start + 1.5 * max_step_time > *cfg.wall_limit_s) {
      std::printf(" Wall-clock limit is near, stopping\n");
      return true;
    }
    if (cfg.min_free_bytes) {
      std::error_code ec;
      const fs::space_info si = fs::space(out, ec);
      if (!ec && static_cast<double>(si.available) < *cfg.min_free_bytes) {
        std::printf(" Free disk space is %.2f GB, stopping\n", static_cast<double>(si.available) / 1e9);
        return true;
      }
    }
    if (cfg.stop_flag && *cfg.stop_flag) {
      std::printf(" Stop requested by a signal\n");
      return true;
    }
    return false;
  };

  bool stopped = false;

  // Optical stage: the optical field alone, JefThread, KdivKZ_opt and LinThread_opt.
  if (z < z_opt_end) std::printf(" --- Optical stage up to %g cm ---\n", c::opt_stage_end);
  while (!stopped && z < z_opt_end) {
    const double wall0 = omp_get_wtime();
    dz = static_cast<double>(0.02f) * c::focus / c::Ldifr * 1 / Imax;
    std::printf(" Step %lld\n", iz);

    std::unique_ptr<Snapshot> snap;
    int z_pos = 0;
    std::fill(Fl.begin(), Fl.end(), 0.0);
    if (z >= z_flag) {
      z_flag = z_flag + (z < c::snap_dense_begin / k0a0a0 ? c::snap_sparse : c::snap_opt_dense) / k0a0a0;
      z_pos = static_cast<int>(z * k0a0a0 * 1e2);
      snap = open_snapshot(out, z_pos, false);
      // No THz field yet: its rows of IS and IO are zeros.
      for (int q = 1; q < nqT; ++q) {
        const float fr = bandT.fr[nqT + q];
        snap->is.f(fr).d(0.0).nl();
        snap->io.f(fr).d(0.0).nl();
      }
    }

    optical_diagnostics(snap.get());

    // First RK stage
    run_nonlinear_opt(dz);
    plasma_diagnostics(snap, z_pos, 0.0);
    to_kr_and_divide_opt();

    // Second RK stage at E + k/2
    hankel_spectral(HO, HankelInput{E_O.data(), K_O.data(), .5, nrO}, W_O.data(), nrO, nqO, nullptr, nullptr);
    spectra_to_time(W_O.data(), nrO, nqO, Et_O.data(), ntO, fftO);
    run_nonlinear_opt(dz);
    to_kr_and_divide_opt();

    add_to(E_O.data(), K_O.data(), nO);

    z = z + dz;
    iz = iz + 1;
    std::fill(neMax.begin(), neMax.end(), 0.0);

    // LinThread_opt
    linear_step(E_O.data(), PO, gO, bandO, tf.opt_lin_start, dz);

    stopped = end_of_step(wall0);
  }

  // THz stage: the THz field joins, JefThread_THz on nt_all, the external DC field is on.
  if (!stopped) std::printf(" --- Start propagation ---\n");
  while (!stopped) {
    const double wall0 = omp_get_wtime();
    dz = static_cast<double>(0.02f) * c::focus / c::Ldifr * 1 / Imax;
    std::printf(" Step %lld\n", iz);

    std::unique_ptr<Snapshot> snap;
    int z_pos = 0;
    std::fill(Fl.begin(), Fl.end(), 0.0);
    if (z >= z_flag) {
      if ((z < c::snap_dense_begin / k0a0a0) || (z >= c::snap_dense_end / k0a0a0)) {
        z_flag = z_flag + c::snap_sparse / k0a0a0;
      } else {
        z_flag = z_flag + c::snap_dense / k0a0a0;
      }
      z_pos = static_cast<int>(z * k0a0a0 * 1e2);
      snap = open_snapshot(out, z_pos, true);
      write_rows_parallel(snap->kw_thz, nrT, [&](int i, LineBuffer& b) {
        const cplx* row = E_T.data() + static_cast<std::size_t>(i) * nqT;
        const double kr = gT.kr[i];
        for (int q = 1; q < nqT; ++q) {
          const int itt = nqT + q;
          const float angle = static_cast<float>(std::asin(kr / bandT.freqs[itt] * c::cL * c::t0 / c::a0));
          b.f(bandT.fr[itt]).f(angle).d(abs2(row[q])).nl();
        }
      });
    }
    const bool fluence = static_cast<bool>(snap);

    // Ebuf = E_THz; HnkThread(E_THz, Ebuf)
    hankel_spectral(HT, HankelInput{E_T.data(), nullptr, 0.0, nrT}, B_T.data(), fluence ? nrT : nrO, nqT,
                    PT.cut.data(), nullptr);
    if (fluence) {
      radial_power(B_T.data(), nrT, nqT, gT, intSp);
      for (int q = 1; q < nqT; ++q) {
        const int itt = nqT + q;
        snap->is.f(bandT.fr[itt]).d(2.0 * intSp[q]).nl();
        snap->io.f(bandT.fr[itt]).d(B_T[q].real()).d(B_T[q].imag()).nl();
      }
    }

    optical_diagnostics(snap.get());

    // First RK stage, field at z
    run_nonlinear(dz, z);
    plasma_diagnostics(snap, z_pos, dc_field_kV(z * c::Ldifr));
    to_kr_and_divide();

    // Second RK stage at E + k/2: Ebuf = E_THz + .5*k_THz, Et = E + .5*k
    hankel_spectral(HT, HankelInput{E_T.data(), K_T.data(), .5, nrT}, B_T.data(), nrO, nqT, PT.cut.data(), nullptr);
    hankel_spectral(HO, HankelInput{E_O.data(), K_O.data(), .5, nrO}, W_O.data(), nrO, nqO, nullptr, nullptr);
    spectra_to_time(W_O.data(), nrO, nqO, Et_O.data(), ntO, fftO);
    // The Fortran window sets DC_Field once per step; the smooth profiles are sampled at the midpoint z + dz/2.
    run_nonlinear(dz, c::dc_profile == 4 ? z : z + 0.5 * dz);
    to_kr_and_divide();

    add_to(E_O.data(), K_O.data(), nO);
    add_to(E_T.data(), K_T.data(), nT);

    z = z + dz;
    iz = iz + 1;
    std::fill(neMax.begin(), neMax.end(), 0.0);

    // LinThread
    linear_step(E_T.data(), PT, gT, bandT, 0, dz);
    linear_step(E_O.data(), PO, gO, bandO, tf.opt_lin_start, dz);

    stopped = end_of_step(wall0);
  }

  fImax.close();
  fNemax.close();
  fEnergy.close();
  fDen.close();
  fSnaps.close();
  if (status == RunStatus::Stopped && !cfg.checkpoint_path.empty()) {
    LoopState st;
    st.iz = iz;
    st.clamped_total = clamped_total;
    st.z = z;
    st.z_flag = z_flag;
    st.Imax = Imax;
    st.stage = z < z_opt_end ? 0 : 1;
    write_checkpoint(cfg.checkpoint_path, sz, st, E_O, E_T);
    std::printf(" Checkpoint written to %s at step %lld, z = %.15g cm\n", path_string(cfg.checkpoint_path).c_str(),
                iz, z * c::Ldifr);
  }
  if (clamped_total > 0) std::printf(" Ionization-table lookups clamped: %lld\n", clamped_total);
  std::printf(" That's it %.15g\n", z);
  std::fflush(stdout);
  return status;
}

}  // namespace uppe
