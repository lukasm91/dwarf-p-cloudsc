#pragma once

#include <algorithm>

#ifndef PRECISION
#define PRECISION double
#endif
using real_t = PRECISION;

namespace {

namespace yomcst {
constexpr real_t rg = 9.80665;
constexpr real_t rd = 287.0596736665907;
constexpr real_t rcpd = 1004.7088578330674;
constexpr real_t retv = 0.6077667316114637;
constexpr real_t rlvtt = 2500800.0;
constexpr real_t rlstt = 2834500.0;
constexpr real_t rlmlt = 333700.0;
constexpr real_t rtt = 273.16;
constexpr real_t rv = 461.5249933083879;
} // namespace yomcst
namespace yoethf {
constexpr real_t r2es = 380.1608703442847;
constexpr real_t r3les = 17.502;
constexpr real_t r3ies = 22.587;
constexpr real_t r4les = 32.19;
constexpr real_t r4ies = -0.7;
constexpr real_t r5les = 4217.45694;
constexpr real_t r5ies = 6185.67582;
constexpr real_t r5alvcp = 10497584.68169531;
constexpr real_t r5alscp = 17451123.253362577;
constexpr real_t ralvdcp = 2489.0792795374246;
constexpr real_t ralsdcp = 2821.2152982440934;
constexpr real_t ralfdcp = 332.1360187066693;
__device__ constexpr real_t rtwat = 273.16;
__device__ constexpr real_t rtice = 250.16000000000003;
constexpr real_t rticecu = 250.16000000000003;
constexpr real_t rtwat_rtice_r = 0.043478260869565216;
constexpr real_t rtwat_rticecu_r = 0.043478260869565216;
constexpr real_t rkoop1 = 2.583;
constexpr real_t rkoop2 = 0.0048116;
} // namespace yoethf
namespace yrecldp {
constexpr bool laericeauto = false;
constexpr bool laericesed = false;
constexpr bool laerliqautocp = false;
constexpr bool laerliqautocpb = false;
constexpr bool laerliqautolsp = false;
constexpr bool laerliqcoll = false;
constexpr bool lcldbudget = false;
constexpr bool lcldextra = false;
constexpr int naeclbc = 9;
constexpr int naecldu = 4;
constexpr int naeclom = 7;
constexpr int naeclss = 1;
constexpr int naeclsu = 11;
constexpr int naercld = 0;
constexpr int nbeta = 100;
constexpr int nclddiag = 0;
constexpr int ncldtop = 15;
constexpr real_t nshapep = 2.414213562373095;
constexpr real_t nshapeq = 2.414213562373095;
constexpr int nssopt = 1;
constexpr real_t ramid = 0.8;
constexpr real_t ramin = 1e-08;
constexpr real_t rbeta = 0.0;
constexpr real_t rbetap1 = 0.0;
constexpr real_t rccn = 125.0;
constexpr real_t rccnom = 0.13;
constexpr real_t rccnss = 0.05;
constexpr real_t rccnsu = 0.5;
constexpr real_t rclcrit = 0.0004;
constexpr real_t rclcrit_land = 0.00055;
constexpr real_t rclcrit_sea = 0.00025;
constexpr real_t rcldiff = 3e-06;
constexpr real_t rcldiff_convi = 7.0;
constexpr real_t rcldmax = 0.005;
constexpr real_t rcldtopcf = 0.01;
constexpr real_t rcldtopp = 100.0;
constexpr real_t rcl_ai = 0.069;
constexpr real_t rcl_apb1 = 714000000000.0;
constexpr real_t rcl_apb2 = 116000000.0;
constexpr real_t rcl_apb3 = 241.6;
constexpr real_t rcl_ar = 523.5987755982989;
constexpr real_t rcl_as = 0.069;
constexpr real_t rcl_bi = 2.0;
constexpr real_t rcl_br = 3.0;
constexpr real_t rcl_bs = 2.0;
constexpr real_t rcl_cdenom1 = 557000000000.0;
constexpr real_t rcl_cdenom2 = 103000000.0;
constexpr real_t rcl_cdenom3 = 204.0;
constexpr real_t rcl_ci = 16.8;
constexpr real_t rcl_const1i = 3.6231880115136998e-06;
constexpr real_t rcl_const1r = 1.382300767579509;
constexpr real_t rcl_const1s = 3.6231880115136998e-06;
constexpr real_t rcl_const2i = 6283185.307179586;
constexpr real_t rcl_const2r = 2143.2299120517614;
constexpr real_t rcl_const2s = 6283185.307179586;
constexpr real_t rcl_const3i = 596.9998475835998;
constexpr real_t rcl_const3r = 0.6349999999999998;
constexpr real_t rcl_const3s = 596.9998475835998;
constexpr real_t rcl_const4i = 0.6666666666666666;
constexpr real_t rcl_const4r = -0.20000000000000018;
constexpr real_t rcl_const4s = 0.6666666666666666;
constexpr real_t rcl_const5i = 0.9211666666666667;
constexpr real_t rcl_const5r = 8685252.965082133;
constexpr real_t rcl_const5s = 0.9211666666666667;
constexpr real_t rcl_const6i = 1.0000000948961185;
constexpr real_t rcl_const6r = -4.8;
constexpr real_t rcl_const6s = 1.0000000948961185;
constexpr real_t rcl_const7s = 90363515.76351073;
constexpr real_t rcl_const8s = 1.1756666666666666;
constexpr real_t rcl_cr = 386.8;
constexpr real_t rcl_cs = 16.8;
constexpr real_t rcl_di = 0.527;
constexpr real_t rcl_dr = 0.67;
constexpr real_t rcl_ds = 0.527;
constexpr real_t rcl_dynvisc = 1.717e-05;
constexpr real_t rcl_fac1 = 4146.902789847063;
constexpr real_t rcl_fac2 = 0.5555555555555556;
constexpr real_t rcl_fzrab = -0.66;
constexpr real_t rcl_fzrbb = 200.0;
constexpr real_t rcl_ka273 = 0.024;
constexpr real_t rcl_kkaac = 67.0;
constexpr real_t rcl_kkaau = 1350.0;
constexpr real_t rcl_kkbac = 1.15;
constexpr real_t rcl_kkbaun = -1.79;
constexpr real_t rcl_kkbauq = 2.47;
constexpr real_t rcl_kk_cloud_num_land = 300.0;
constexpr real_t rcl_kk_cloud_num_sea = 50.0;
constexpr real_t rcl_schmidt = 0.6;
constexpr real_t rcl_x1i = 2000000.0;
constexpr real_t rcl_x1r = 0.22;
constexpr real_t rcl_x1s = 2000000.0;
constexpr real_t rcl_x2i = 0.0;
constexpr real_t rcl_x2r = 2.2;
constexpr real_t rcl_x2s = 0.0;
constexpr real_t rcl_x3i = 1.0;
constexpr real_t rcl_x3s = 1.0;
constexpr real_t rcl_x4i = 0.0;
constexpr real_t rcl_x4r = 0.0;
constexpr real_t rcl_x4s = 0.0;
constexpr real_t rcovpmin = 0.1;
constexpr real_t rdensref = 1.0;
constexpr real_t rdenswat = 1000.0;
constexpr real_t rdepliqrefdepth = 500.0;
constexpr real_t rdepliqrefrate = 0.1;
constexpr real_t ricehi1 = 3.3333333333333335e-05;
constexpr real_t ricehi2 = 0.004291845493562232;
constexpr real_t riceinit = 1e-12;
constexpr real_t rkconv = 0.00016666666666666666;
constexpr real_t rkooptau = 10800.0;
constexpr real_t rlcritsnow = 3e-05;
constexpr real_t rlmin = 1e-08;
constexpr real_t rnice = 0.027;
constexpr real_t rpecons = 5.54725619859993e-05;
constexpr real_t rprc1 = 100.0;
constexpr real_t rprc2 = 0.5;
constexpr real_t rprecrhmax = 0.7;
constexpr real_t rsnowlin1 = 0.001;
constexpr real_t rsnowlin2 = 0.03;
constexpr real_t rtaumel = 7200.0;
constexpr real_t rthomo = 235.16000000000003;
constexpr real_t rvice = 0.13;
constexpr real_t rvrain = 4.0;
constexpr real_t rvrfactor = 0.00509;
constexpr real_t rvsnow = 1.0;
} // namespace yrecldp
namespace yrephli {
constexpr bool lenopert = true;
constexpr bool leppcfls = false;
constexpr bool lphylin = false;
constexpr bool lraisanen = true;
constexpr bool ltlevol = false;
constexpr real_t rlpal1 = 0.15;
constexpr real_t rlpal2 = 20.0;
constexpr real_t rlpbb = 5.0;
constexpr real_t rlpbeta = 0.2;
constexpr real_t rlpcc = 5.0;
constexpr real_t rlpdd = 5.0;
constexpr real_t rlpdrag = 0.0;
constexpr real_t rlpevap = 0.0;
constexpr real_t rlpmixl = 4000.0;
constexpr real_t rlpp00 = 30000.0;
constexpr real_t rlptrc = 266.42345596729064;
} // namespace yrephli

constexpr int NCLV = 5;
enum type {
  NCLDQL = 0,
  NCLDQI = 1,
  NCLDQR = 2,
  NCLDQS = 3,
  NCLDQV = 4,
  UNKNOWN = -99
};
enum phase { VAPOUR = 0, LIQUID = 1, ICE = 2 };

constexpr phase iphase(type t) {
  if (t == NCLDQV)
    return VAPOUR;
  else if (t == NCLDQL || t == NCLDQR)
    return LIQUID;
  else if (t == NCLDQI or t == NCLDQS)
    return ICE;
}
// ---------------------------------------------------
// Set up melting/freezing index,
// if an ice category melts/freezes, where does it go?
// ---------------------------------------------------
constexpr type imelt(type t) {
  if (t == NCLDQV)
    return UNKNOWN;
  else if (t == NCLDQL)
    return NCLDQI;
  else if (t == NCLDQR)
    return NCLDQS;
  else if (t == NCLDQI)
    return NCLDQR;
  else if (t == NCLDQS)
    return NCLDQR;
}

//  -------------------------
//  set up fall speeds in m/s
//  -------------------------
__device__ real_t zvqx(type t) {
  if (t == NCLDQV)
    return 0;
  else if (t == NCLDQL)
    return 0;
  else if (t == NCLDQI)
    return yrecldp::rvice;
  else if (t == NCLDQR)
    return yrecldp::rvrain;
  else if (t == NCLDQS)
    return yrecldp::rvsnow;
  __builtin_unreachable();
}

__device__ bool llfall(type t) {
  // Set LLFALL to false for ice (but ice still sediments!)
  // Need to rationalise this at some point
  if (t == NCLDQI)
    return false;
  else
    return zvqx(t) > 0;
}

// foedelta = 1    water
// foedelta = 0    ice
__device__ int foedelta(real_t ptare) {
  if (ptare - yomcst::rtt >= 0)
    return 1;
  else
    return 0;
}

// THERMODYNAMICAL FUNCTIONS .
// Pressure of water vapour at saturation
// PTARE = TEMPERATURE
__device__ real_t foeew(real_t ptare) {
  return yoethf::r2es *
         std::exp((yoethf::r3les * foedelta(ptare) +
                   yoethf::r3ies * (real_t(1) - foedelta(ptare))) *
                  (ptare - yomcst::rtt) /
                  (ptare - (yoethf::r4les * foedelta(ptare) +
                            yoethf::r4ies * (real_t(1) - foedelta(ptare)))));
}

constexpr real_t sqr(real_t v) { return v * v; }
constexpr real_t cube(real_t v) { return v * v * v; }
// CONSIDERATION OF MIXED PHASES
// FOEALFA is calculated to distinguish the three cases:
// FOEALFA=1            water phase
// FOEALFA=0            ice phase
// 0 < FOEALFA < 1      mixed phase
__device__ real_t foealfa(real_t ptare) {
  return std::min(real_t(1),
                  sqr((std::max(yoethf::rtice, std::min(yoethf::rtwat, ptare)) -
                       yoethf::rtice) *
                      yoethf::rtwat_rtice_r));
}

// Pressure of water vapour at saturation
__device__ real_t foeewm(real_t ptare) {
  return yoethf::r2es *
         (foealfa(ptare) * std::exp(yoethf::r3les * (ptare - yomcst::rtt) /
                                    (ptare - yoethf::r4les)) +
          (real_t(1) - foealfa(ptare)) *
              std::exp(yoethf::r3ies * (ptare - yomcst::rtt) /
                       (ptare - yoethf::r4ies)));
}

__device__ real_t foedem(real_t ptare) {
  return foealfa(ptare) * yoethf::r5alvcp *
             (real_t(1) / sqr(ptare - yoethf::r4les)) +
         (real_t(1) - foealfa(ptare)) * yoethf::r5alscp *
             (real_t(1) / sqr(ptare - yoethf::r4ies));
}

__device__ real_t foeldcpm(real_t ptare) {
  return foealfa(ptare) * yoethf::ralvdcp +
         (real_t(1) - foealfa(ptare)) * yoethf::ralsdcp;
}

// Pressure of water vapour at saturation
// This one is for the WMO definition of saturation, i.e. always
// with respect to water.
//
// Duplicate to FOEELIQ and FOEEICE for separate ice variable
// FOEELIQ always respect to water
// FOEEICE always respect to ice
// (could use FOEEW and FOEEWMO, but naming convention unclear)
__device__ real_t foeeliq(real_t ptare) {
  return yoethf::r2es * std::exp(yoethf::r3les * (ptare - yomcst::rtt) /
                                 (ptare - yoethf::r4les));
}

__device__ real_t foeeice(real_t ptare) {
  return yoethf::r2es * std::exp(yoethf::r3ies * (ptare - yomcst::rtt) /
                                 (ptare - yoethf::r4ies));
}

__device__ real_t fokoop(real_t ptare) {
  return std::min(yoethf::rkoop1 - yoethf::rkoop2 * ptare,
                  foeeliq(ptare) / foeeice(ptare));
}

constexpr real_t zepsilon =
    real_t(100) * std::numeric_limits<real_t>::epsilon();

// Some simple constants
constexpr real_t zgdcp = yomcst::rg / yomcst::rcpd;
constexpr real_t zrdcp = yomcst::rd / yomcst::rcpd;
constexpr real_t zcons1a =
    yomcst::rcpd / ((yomcst::rlmlt * yomcst::rg * yrecldp::rtaumel));
constexpr real_t zepsec = real_t(1.0e-14);
;
constexpr real_t zrg_r = real_t(1) / yomcst::rg;
constexpr real_t zrldcp = real_t(1) / (yoethf::ralsdcp - yoethf::ralvdcp);

} // namespace

// Numerical fit to wet bulb temperature
constexpr real_t ZTW1 = 1329.31;
constexpr real_t ZTW2 = 0.0074615;
constexpr real_t ZTW3 = 0.85E5;
constexpr real_t ZTW4 = 40.637;
constexpr real_t ZTW5 = 275.0;
