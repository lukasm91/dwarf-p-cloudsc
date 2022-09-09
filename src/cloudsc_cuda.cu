#include "constants.h"

#include <cooperative_groups/memcpy_async.h>
#include <cuda/pipeline>

#define CUDA_CHECK(e)                                                          \
  {                                                                            \
    cudaError_t err = (e);                                                     \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error: %s, line %d, %s: %s\n", __FILE__, __LINE__, \
              #e, cudaGetErrorString(err));                                    \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }

#include <iostream>
#include <pybind11/pybind11.h>
namespace py = pybind11;
namespace {
struct Buffer {
  Buffer(Buffer const &) = delete;
  Buffer(Buffer &&) = delete;

  Buffer(py::buffer const &a) {
    py::buffer_info info = a.request();
    alloc_sz_ = info.size * info.itemsize;
    hptr_ = (real_t *)info.ptr;

    CUDA_CHECK(cudaMalloc(&dptr_, alloc_sz_));
    CUDA_CHECK(cudaMemcpy(dptr_, hptr_, alloc_sz_, cudaMemcpyHostToDevice));
  }
  ~Buffer() {
    CUDA_CHECK(cudaMemcpy(hptr_, dptr_, alloc_sz_, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(dptr_));
    dptr_ = nullptr;
  }

  real_t *get() const { return dptr_; }

private:
  real_t *dptr_, *hptr_;
  size_t alloc_sz_;
};
} // namespace

constexpr int POS_T = 0;
constexpr int POS_Q = 1;
constexpr int POS_A = 2;
constexpr int POS_CLD = 3;

constexpr size_t stage_count = 2;
constexpr size_t smem_count = stage_count;
enum SHARED {
  PT,
  PQ,
  PA,
  TEND_T,
  TEND_Q,
  TEND_A,
  PCLV,
  TEND_CLD = PCLV + 4,
  PAP = TEND_CLD + 4,
  PAPH,
  PSUPSAT,
  PLU,
  PSNDE,
  PMFU,
  PMFD,
  PVERVEL,
  PHRSW,
  PHRLW,
  PVFL,
  PVFI,
  PLUDE,

  LAST3D = PLUDE,
  TOTAL3D = LAST3D + 1,
  FIRST2D = smem_count * TOTAL3D,

  // No multi-stage buffer
  LDCUM = FIRST2D,
  KTYPE,
  PLSM,
  PFSQLF,
  PFSQIF,
  PFCQLNG,
  PFCQNNG,
  PFSQLTUR,
  PFSQITUR,

  LAST2D = PFSQITUR,
  TOTAL = LAST2D + 1,

};
// template <int > class X{};
// using notype = typename X<TOTAL>::notype;

struct descriptor {
  __device__ descriptor(int nproma, int nk, int nfields)
      : offset_(nproma * nk * nfields * blockIdx.x), fstride_(nproma * nk) {}
  __device__ int get(int nproma, int ki, int fi) const {
    return offset_ + fi * fstride_ + ki * nproma;
  };

private:
  int offset_;
  int fstride_;
};

inline __device__ int get_index(descriptor const &d, int nproma, int ki,
                                int fi = 0) {
  return d.get(nproma, ki, fi) + threadIdx.x;
}
inline __device__ int get_index_block(descriptor const &d, int nproma, int ki,
                                      int fi = 0) {
  return d.get(nproma, ki, fi);
}

template <int nproma>
inline __device__ void
copy(cuda::pipeline<cuda::thread_scope::thread_scope_thread> &pipeline,
     descriptor const &d, real_t *__restrict__ ptr, int const spos,
     int const ki, int const koffset = 0, int const fi = 0) {
  extern __shared__ real_t shared[];
  int stage = ki % smem_count;
  cuda::memcpy_async(cooperative_groups::this_thread_block(),
                     shared + stage * SHARED::TOTAL3D * nproma + spos * nproma,
                     &ptr[get_index_block(d, nproma, ki + koffset, fi)],
                     cuda::aligned_size_t<128>(sizeof(*ptr) * nproma),
                     pipeline);
}
template <int nproma>
inline __device__ real_t &at_shared(int const spos, int const jk = 0) {
  extern __shared__ real_t shared[];
  int stage = jk % smem_count;
  return shared[stage * SHARED::TOTAL3D * nproma + spos * nproma + threadIdx.x];
}

// ---------------------------------------------------------------------
// Set version of warm-rain autoconversion/accretion
// IWARMRAIN = 1 ! Sundquist
// IWARMRAIN = 2 ! Khairoutdinov and Kogan (2000)
// ---------------------------------------------------------------------
// ---------------------------------------------------------------------
// Set version of rain evaporation
// IEVAPRAIN = 1 ! Sundquist
// IEVAPRAIN = 2 ! Abel and Boutle (2013)
// ---------------------------------------------------------------------
// ---------------------------------------------------------------------
// Set version of snow evaporation
// IEVAPSNOW = 1 ! Sundquist
// IEVAPSNOW = 2 ! New
// ---------------------------------------------------------------------
// ---------------------------------------------------------------------
// Set version of ice deposition
// IDEPICE = 1 ! Rotstayn (2001)
// IDEPICE = 2 ! New
// ---------------------------------------------------------------------
template <int iwarmrain, int ievaprain, int ievapsnow, int idepice, int nproma>
__global__ void run_cloudsc(
    int klev, int ngptot, int nproma_, real_t *__restrict__ plcrit_aer,
    real_t *__restrict__ picrit_aer, real_t *__restrict__ pre_ice,
    real_t *__restrict__ pccn, real_t *__restrict__ pnice,
    real_t *__restrict__ pt, real_t *__restrict__ pq, real_t *__restrict__ pvfl,
    real_t *__restrict__ pvfi, real_t *__restrict__ phrsw,
    real_t *__restrict__ phrlw, real_t *__restrict__ pvervel,
    real_t *__restrict__ pap, real_t *__restrict__ paph,
    real_t *__restrict__ plsm, real_t *__restrict__ ldcum,
    real_t *__restrict__ ktype, real_t *__restrict__ plu,
    real_t *__restrict__ plude, real_t *__restrict__ psnde,
    real_t *__restrict__ pmfu, real_t *__restrict__ pmfd,
    real_t *__restrict__ pa, real_t *__restrict__ pclv,
    real_t *__restrict__ psupsat, real_t *__restrict__ tendency_tmp,
    real_t *__restrict__ tendency_loc, real_t *__restrict__ prainfrac_toprfz,
    real_t *__restrict__ pcovptot, real_t *__restrict__ pfsqlf,
    real_t *__restrict__ pfsqif, real_t *__restrict__ pfcqlng,
    real_t *__restrict__ pfcqnng, real_t *__restrict__ pfsqrf,
    real_t *__restrict__ pfsqsf, real_t *__restrict__ pfcqrng,
    real_t *__restrict__ pfcqsng, real_t *__restrict__ pfsqltur,
    real_t *__restrict__ pfsqitur, real_t *__restrict__ pfplsl,
    real_t *__restrict__ pfplsn, real_t *__restrict__ pfhpsl,
    real_t *__restrict__ pfhpsn, real_t *__restrict__ tmp1,
    real_t *__restrict__ tmp2, real_t ptsphy, bool ldslphy) {

  if (nproma * blockIdx.x + threadIdx.x >= ngptot)
    return;

  descriptor descriptor_k = descriptor{nproma, klev, 1};
  descriptor descriptor_k1 = descriptor{nproma, klev + 1, 1};
  descriptor descriptor_tendency = descriptor{nproma, klev, 8};
  descriptor descriptor_pclv = descriptor{nproma, klev, 5};
  descriptor descriptor_2d = descriptor{nproma, 1, 1};

  real_t const zqtmst = real_t(1) / ptsphy;
  real_t const paph_top = paph[get_index(descriptor_k1, nproma, klev)];
  // returns paph(jl,klev+1,ibl)

  auto grid = cooperative_groups::this_grid();
  auto block = cooperative_groups::this_thread_block();

  auto pipeline = cuda::make_pipeline();

  // fluxes, summed up towwars the top
  at_shared<nproma>(SHARED::PFSQLF) = 0;
  at_shared<nproma>(SHARED::PFSQIF) = 0;
  at_shared<nproma>(SHARED::PFCQLNG) = 0;
  at_shared<nproma>(SHARED::PFCQNNG) = 0;
  at_shared<nproma>(SHARED::PFSQLTUR) = 0;
  at_shared<nproma>(SHARED::PFSQITUR) = 0;
  pfsqlf[get_index(descriptor_k1, nproma, 0)] = real_t(0);
  pfsqif[get_index(descriptor_k1, nproma, 0)] = real_t(0);
  pfsqrf[get_index(descriptor_k1, nproma, 0)] = real_t(0);
  pfsqsf[get_index(descriptor_k1, nproma, 0)] = real_t(0);
  pfcqlng[get_index(descriptor_k1, nproma, 0)] = real_t(0);
  pfcqnng[get_index(descriptor_k1, nproma, 0)] = real_t(0);
  pfcqrng[get_index(descriptor_k1, nproma, 0)] = real_t(0);
  pfcqsng[get_index(descriptor_k1, nproma, 0)] = real_t(0);
  // fluxes due to turbulence
  pfsqltur[get_index(descriptor_k1, nproma, 0)] = real_t(0);
  pfsqitur[get_index(descriptor_k1, nproma, 0)] = real_t(0);

  // not summed up
  pfplsl[get_index(descriptor_k1, nproma, 0)] = real_t(0);
  pfplsn[get_index(descriptor_k1, nproma, 0)] = real_t(0);
  pfhpsl[get_index(descriptor_k1, nproma, 0)] = real_t(0);
  pfhpsn[get_index(descriptor_k1, nproma, 0)] = real_t(0);

  // next values hold the values from the previous k-level
  real_t zqxnm1[NCLV - 1] = {};
  real_t ztp1_prev = std::numeric_limits<real_t>::quiet_NaN();
  real_t zanewm1 = 0;
  real_t zcovptot = 0;
  real_t zcovpmax = 0;
  real_t zcldtopdist = 0;
  real_t zpfplsx[NCLV - 1] = {};
  real_t za_prev = std::numeric_limits<real_t>::quiet_NaN();
  real_t pap_prev = std::numeric_limits<real_t>::quiet_NaN();
  real_t paph_ = paph[get_index(descriptor_k1, nproma, 0)];
  real_t pmf_ = pmfu[get_index(descriptor_k, nproma, 0)] +
                pmfd[get_index(descriptor_k, nproma, 0)];

  // rain fraction at top of refreezing layer
  real_t prainfrac_toprfz_ = 0;

#pragma unroll
  for (int jk = 0; jk < stage_count - 1; ++jk) {
    pipeline.producer_acquire();
    copy<nproma>(pipeline, descriptor_k, pt, SHARED::PT, jk);
    copy<nproma>(pipeline, descriptor_k, pq, SHARED::PQ, jk);
    copy<nproma>(pipeline, descriptor_k, pa, SHARED::PA, jk);
    copy<nproma>(pipeline, descriptor_tendency, tendency_tmp, SHARED::TEND_T,
                 jk, 0, POS_T);
    copy<nproma>(pipeline, descriptor_tendency, tendency_tmp, SHARED::TEND_Q,
                 jk, 0, POS_Q);
    copy<nproma>(pipeline, descriptor_tendency, tendency_tmp, SHARED::TEND_A,
                 jk, 0, POS_A);
#pragma unroll
    for (int jm = 0; jm < NCLV - 1; ++jm) {
      copy<nproma>(pipeline, descriptor_pclv, pclv, SHARED::PCLV + jm, jk, 0,
                   jm);
      copy<nproma>(pipeline, descriptor_tendency, tendency_tmp,
                   SHARED::TEND_CLD + jm, jk, 0, POS_CLD + jm);
    }
    copy<nproma>(pipeline, descriptor_k, pap, SHARED::PAP, jk);
    copy<nproma>(pipeline, descriptor_k1, paph, SHARED::PAPH, jk, 1);
    copy<nproma>(pipeline, descriptor_k, pvfl, SHARED::PVFL, jk);
    copy<nproma>(pipeline, descriptor_k, pvfi, SHARED::PVFI, jk);
    copy<nproma>(pipeline, descriptor_k, plude, SHARED::PLUDE, jk);

    if (jk >= yrecldp::ncldtop - 1) {
      copy<nproma>(pipeline, descriptor_k, psupsat, SHARED::PSUPSAT, jk);
      if (jk < klev - 1) {
        copy<nproma>(pipeline, descriptor_k, plu, SHARED::PLU, jk, 1);
        copy<nproma>(pipeline, descriptor_k, psnde, SHARED::PSNDE, jk);
        copy<nproma>(pipeline, descriptor_k, pmfu, SHARED::PMFU, jk, 1);
        copy<nproma>(pipeline, descriptor_k, pmfd, SHARED::PMFD, jk, 1);
      }
      copy<nproma>(pipeline, descriptor_k, pvervel, SHARED::PVERVEL, jk);
      copy<nproma>(pipeline, descriptor_k, phrsw, SHARED::PHRSW, jk);
      copy<nproma>(pipeline, descriptor_k, phrlw, SHARED::PHRLW, jk);
    }

    if (jk == 0) {
      copy<nproma>(pipeline, descriptor_2d, ldcum, SHARED::LDCUM, jk);
      copy<nproma>(pipeline, descriptor_2d, ktype, SHARED::KTYPE, jk);
      copy<nproma>(pipeline, descriptor_2d, plsm, SHARED::PLSM, jk);
    }
    pipeline.producer_commit();
  }

  for (int jk = 0; jk < klev; ++jk) {

    int to_load = jk + stage_count - 1;
    if (to_load < klev) {
      pipeline.producer_acquire();
      copy<nproma>(pipeline, descriptor_k, pt, SHARED::PT, to_load);
      copy<nproma>(pipeline, descriptor_k, pq, SHARED::PQ, to_load);
      copy<nproma>(pipeline, descriptor_k, pa, SHARED::PA, to_load);
      copy<nproma>(pipeline, descriptor_tendency, tendency_tmp, SHARED::TEND_T,
                   to_load, 0, POS_T);
      copy<nproma>(pipeline, descriptor_tendency, tendency_tmp, SHARED::TEND_Q,
                   to_load, 0, POS_Q);
      copy<nproma>(pipeline, descriptor_tendency, tendency_tmp, SHARED::TEND_A,
                   to_load, 0, POS_A);
#pragma unroll
      for (int jm = 0; jm < NCLV - 1; ++jm) {
        copy<nproma>(pipeline, descriptor_pclv, pclv, SHARED::PCLV + jm,
                     to_load, 0, jm);
        copy<nproma>(pipeline, descriptor_tendency, tendency_tmp,
                     SHARED::TEND_CLD + jm, to_load, 0, POS_CLD + jm);
      }
      copy<nproma>(pipeline, descriptor_k, pap, SHARED::PAP, to_load);
      copy<nproma>(pipeline, descriptor_k1, paph, SHARED::PAPH, to_load, 1);
      copy<nproma>(pipeline, descriptor_k, pvfl, SHARED::PVFL, to_load);
      copy<nproma>(pipeline, descriptor_k, pvfi, SHARED::PVFI, to_load);
      copy<nproma>(pipeline, descriptor_k, plude, SHARED::PLUDE, to_load);

      if (to_load >= yrecldp::ncldtop - 1) {
        copy<nproma>(pipeline, descriptor_k, psupsat, SHARED::PSUPSAT, to_load);
        if (to_load < klev - 1) {
          copy<nproma>(pipeline, descriptor_k, plu, SHARED::PLU, to_load, 1);
          copy<nproma>(pipeline, descriptor_k, psnde, SHARED::PSNDE, to_load);
          copy<nproma>(pipeline, descriptor_k, pmfu, SHARED::PMFU, to_load, 1);
          copy<nproma>(pipeline, descriptor_k, pmfd, SHARED::PMFD, to_load, 1);
        }
        copy<nproma>(pipeline, descriptor_k, pvervel, SHARED::PVERVEL, to_load);
        copy<nproma>(pipeline, descriptor_k, phrsw, SHARED::PHRSW, to_load);
        copy<nproma>(pipeline, descriptor_k, phrlw, SHARED::PHRLW, to_load);
      }

      if (to_load == 0) {
        copy<nproma>(pipeline, descriptor_2d, ldcum, SHARED::LDCUM, to_load);
        copy<nproma>(pipeline, descriptor_2d, ktype, SHARED::KTYPE, to_load);
        copy<nproma>(pipeline, descriptor_2d, plsm, SHARED::PLSM, to_load);
      }
      pipeline.producer_commit();
    }

    // initialization of output tendencies
    real_t tendency_loc_t_ = 0;
    real_t tendency_loc_q_ = 0;
    real_t tendency_loc_a_ = 0;

    pipeline.consumer_wait();
    block.sync();

    // non CLV initialization
    real_t zqx[NCLV];
    real_t ztp1 = at_shared<nproma>(SHARED::PT, jk) +
                  ptsphy * at_shared<nproma>(SHARED::TEND_T, jk);
    zqx[NCLDQV] = at_shared<nproma>(SHARED::PQ, jk) +
                  ptsphy * at_shared<nproma>(SHARED::TEND_Q, jk);
    real_t pa_ = at_shared<nproma>(SHARED::PA, jk);
    real_t tendency_tmp_a_ = at_shared<nproma>(SHARED::TEND_A, jk);
    real_t za = pa_ + ptsphy * tendency_tmp_a_;
    real_t zaorig = pa_ + ptsphy * tendency_tmp_a_;

    // initialization for CLV family
    real_t zqx0[NCLV - 1];
#pragma unroll
    for (int jm = 0; jm < NCLV - 1; ++jm) {
      zqx[jm] = at_shared<nproma>(SHARED::PCLV + jm, jk) +
                ptsphy * at_shared<nproma>(SHARED::TEND_CLD + jm, jk);
      zqx0[jm] = zqx[jm];
    }

    real_t zlneg[NCLV - 1] = {};
    // Tidy up very small cloud cover or total cloud water
    if (zqx[NCLDQL] + zqx[NCLDQI] < yrecldp::rlmin || za < yrecldp::ramin) {

      // Evaporate small cloud liquid water amounts
      zlneg[NCLDQL] = zlneg[NCLDQL] + zqx[NCLDQL];
      real_t zqadj = zqx[NCLDQL] * zqtmst;
      tendency_loc_q_ += zqadj;
      tendency_loc_t_ -= yoethf::ralvdcp * zqadj;
      zqx[NCLDQV] += zqx[NCLDQL];
      zqx[NCLDQL] = real_t(0);

      // Evaporate small cloud ice water amounts
      zlneg[NCLDQI] = zlneg[NCLDQI] + zqx[NCLDQI];
      zqadj = zqx[NCLDQI] * zqtmst;
      tendency_loc_q_ += zqadj;
      tendency_loc_t_ -= yoethf::ralsdcp * zqadj;
      zqx[NCLDQV] += zqx[NCLDQI];
      zqx[NCLDQI] = real_t(0);

      // Set cloud cover to zero
      za = real_t(0);
    }

    // Tidy up small CLV variables
#pragma unroll
    for (int jm = 0; jm < NCLV - 1; ++jm) {
      if (zqx[jm] < yrecldp::rlmin) {
        zlneg[jm] += zqx[jm];
        real_t zqadj = zqx[jm] * zqtmst;
        tendency_loc_q_ += zqadj;
        if (iphase(static_cast<type>(jm)) == LIQUID)
          tendency_loc_t_ -= yoethf::ralvdcp * zqadj;
        if (iphase(static_cast<type>(jm)) == ICE)
          tendency_loc_t_ -= yoethf::ralsdcp * zqadj;
        zqx[NCLDQV] = zqx[NCLDQV] + zqx[jm];
        zqx[jm] = 0;
      }
    }

    ////
    // Define saturation values

    real_t pap_ = at_shared<nproma>(SHARED::PAP, jk);
    real_t paph_next = at_shared<nproma>(SHARED::PAPH, jk);
    // old *diagnostic* mixed phase saturation
    real_t zfoealfa = foealfa(ztp1);
    real_t zfoeewmt = min(foeewm(ztp1) / pap_, real_t(0.5));
    real_t zqsmix = zfoeewmt;
    zqsmix = zqsmix / (real_t(1) - yomcst::retv * zqsmix);

    // ice saturation T<273K
    // liquid water saturation for T>273K
    real_t zalfa = foedelta(ztp1);
    real_t zfoeew = min(
        (zalfa * foeeliq(ztp1) + (real_t(1) - zalfa) * foeeice(ztp1)) / pap_,
        real_t(0.5));
    real_t zqsice = zfoeew / (real_t(1) - yomcst::retv * zfoeew);

    // liquid water saturation
    real_t zfoeeliqt = min(foeeliq(ztp1) / pap_, real_t(0.5));
    real_t zqsliq = zfoeeliqt;
    zqsliq = zqsliq / (real_t(1) - yomcst::retv * zqsliq);

    // ice water saturation
    // [ DELETED ]

    // Ensure cloud fraction is between 0 and 1
    za = max(real_t(0), min(real_t(1), za));

    // Calculate liq/ice fractions (no longer a diagnostic relationship)
    real_t zli = zqx[NCLDQL] + zqx[NCLDQI];
    real_t zliqfrac, zicefrac;
    if (zli > yrecldp::rlmin) {
      zliqfrac = zqx[NCLDQL] / zli;
      zicefrac = real_t(1) - zliqfrac;
    } else {
      zliqfrac = real_t(0);
      zicefrac = real_t(0);
    }

    real_t zqxn2d[NCLV - 1];
    real_t plude_ = at_shared<nproma>(SHARED::PLUDE, jk);
    if (jk >= yrecldp::ncldtop - 1) {

      //////
      //        2.       *** CONSTANTS AND PARAMETERS ***
      // Calculate L in updrafts of bl-clouds
      //  Specify QS, P/PS for tropopause (for c2)
      //  And initialize variables

      //////
      //           3.       *** PHYSICS ***
      // START OF VERTICAL LOOP

      // 3.0 INITIALIZE VARIABLES

      real_t zqxfg[NCLV];
      // First guess microphysics
#pragma unroll
      for (int jm = 0; jm < NCLV; ++jm)
        zqxfg[jm] = zqx[jm];

      // set arrays to zero
      real_t zldefr = real_t(0);
      real_t zqpretot = real_t(0);

      // solvers for cloud fraction
      real_t zsolab = real_t(0);
      real_t zsolac = real_t(0);

      real_t zicetot = real_t(0);

      // reset matrix so missing pathways are set
      real_t zsolqa[NCLV][NCLV] = {};
      real_t zsolqb[NCLV][NCLV] = {};

      // reset new microphysics variables
      real_t zfallsrce[NCLV] = {};
      real_t zfallsink[NCLV] = {};
      real_t zconvsrce[NCLV] = {};
      real_t zconvsink[NCLV] = {};
      real_t zpsupsatsrce[NCLV] = {};

      // derived variables needed
      real_t zdp = paph_next - paph_;           // dp
      real_t zgdp = yomcst::rg / zdp;           // g/dp
      real_t zrho = pap_ / (yomcst::rd * ztp1); // p/rt air density

      real_t zdtgdp = ptsphy * zgdp;                              // dt g/dp
      real_t zrdtgdp = zdp * (real_t(1) / (ptsphy * yomcst::rg)); // 1/(dt g/dp)

      // Calculate dqs/dT correction factor
      // Reminder: RETV=RV/RD-1

      // liquid
      real_t zfacw = yoethf::r5les / (sqr(ztp1 - yoethf::r4les));
      real_t zcor = real_t(1) / (real_t(1) - yomcst::retv * zfoeeliqt);
      real_t zdqsliqdt = zfacw * zcor * zqsliq;
      real_t zcorqsliq = real_t(1) + yoethf::ralvdcp * zdqsliqdt;

      // ice
      real_t zfaci = yoethf::r5ies / (sqr(ztp1 - yoethf::r4ies));
      zcor = real_t(1) / (real_t(1) - yomcst::retv * zfoeew);
      real_t zdqsicedt = zfaci * zcor * zqsice;
      real_t zcorqsice = real_t(1) + yoethf::ralsdcp * zdqsicedt;

      // diagnostic mixed
      real_t zalfaw = zfoealfa;
      real_t zfac = zalfaw * zfacw + (real_t(1) - zalfaw) * zfaci;
      zcor = real_t(1) / (real_t(1) - yomcst::retv * zfoeewmt);
      real_t zdqsmixdt = zfac * zcor * zqsmix;
      real_t zcorqsmix = real_t(1) + foeldcpm(ztp1) * zdqsmixdt;

      // evaporation/sublimation limits
      real_t zevaplimmix = max((zqsmix - zqx[NCLDQV]) / zcorqsmix, real_t(0));
      real_t zevaplimice = max((zqsice - zqx[NCLDQV]) / zcorqsice, real_t(0));

      // in-cloud consensate amount
      real_t ztmpa = real_t(1) / max(za, zepsec);
      real_t zliqcld = zqx[NCLDQL] * ztmpa;
      real_t zicecld = zqx[NCLDQI] * ztmpa;
      real_t zlicld = zliqcld + zicecld;

      // Evaporate very small amounts of liquid and ice
      if (zqx[NCLDQL] < yrecldp::rlmin) {
        zsolqa[NCLDQV][NCLDQL] = zqx[NCLDQL];
        zsolqa[NCLDQL][NCLDQV] = -zqx[NCLDQL];
      }

      if (zqx[NCLDQI] < yrecldp::rlmin) {
        zsolqa[NCLDQV][NCLDQI] = zqx[NCLDQI];
        zsolqa[NCLDQI][NCLDQV] = -zqx[NCLDQI];
      }

      //  3.1  ICE SUPERSATURATION ADJUSTMENT
      // Note that the supersaturation adjustment is made with respect to
      // liquid saturation:  when T>0C
      // ice saturation:     when T<0C
      //                     with an adjustment made to allow for ice
      //                     supersaturation in the clear sky
      // Note also that the KOOP factor automatically clips the supersaturation
      // to a maximum set by the liquid water saturation mixing ratio
      // important for temperatures near to but below 0C
      //-----------------------------------------------------------------------

      // 3.1.1 Supersaturation limit (from Koop)
      // Needs to be set for all temperatures
      real_t zfokoop = fokoop(ztp1);

      if (ztp1 >= yomcst::rtt || yrecldp::nssopt == 0) {
        zfac = real_t(1);
        zfaci = real_t(1);
      } else {
        zfac = za + zfokoop * (real_t(1.0) - za);
        zfaci = ptsphy / yrecldp::rkooptau;
      }

      // 3.1.2 Calculate supersaturation wrt Koop including dqs/dT
      //       correction factor
      // [#Note: QSICE or QSLIQ]
      // Calculate supersaturation to add to cloud
      real_t zsupsat;
      if (za > real_t(1) - yrecldp::ramin) {
        zsupsat = max((zqx[NCLDQV] - zfac * zqsice) / zcorqsice, real_t(0));
      } else {
        // calculate environmental humidity supersaturation
        real_t zqp1env =
            (zqx[NCLDQV] - za * zqsice) / max(real_t(1) - za, zepsilon);
        zsupsat =
            max(((real_t(1) - za) * (zqp1env - zfac * zqsice)) / zcorqsice,
                real_t(0));
      }

      // Here the supersaturation is turned into liquid water
      // However, if the temperature is below the threshold for homogeneous
      // freezing then the supersaturation is turned instantly to ice.
      if (zsupsat > zepsec) {

        if (ztp1 > yrecldp::rthomo) {
          // Turn supersaturation into liquid water
          zsolqa[NCLDQL][NCLDQV] += zsupsat;
          zsolqa[NCLDQV][NCLDQL] -= zsupsat;
          // Include liquid in first guess
          zqxfg[NCLDQL] += zsupsat;
        } else {
          // Turn supersaturation into ice water
          zsolqa[NCLDQI][NCLDQV] += zsupsat;
          zsolqa[NCLDQV][NCLDQI] -= zsupsat;
          // Add ice to first guess for deposition term
          zqxfg[NCLDQI] += zsupsat;
        }

        // Increase cloud amount using RKOOPTAU timescale
        zsolac = (real_t(1) - za) * zfaci;
      }

      // 3.1.3 Include supersaturation from previous timestep
      // (Calculated in sltENDIF semi-lagrangian LDSLPHY=T)
      real_t psupsat_ = at_shared<nproma>(SHARED::PSUPSAT, jk);
      if (psupsat_ > zepsec) {
        if (ztp1 > yrecldp::rthomo) {
          // Turn supersaturation into liquid water
          zsolqa[NCLDQL][NCLDQL] += psupsat_;
          zpsupsatsrce[NCLDQL] = psupsat_;
          // Add liquid to first guess for deposition term
          zqxfg[NCLDQL] += psupsat_;
          // Store cloud budget diagnostics if required
        } else {
          // Turn supersaturation into ice water
          zsolqa[NCLDQI][NCLDQI] += psupsat_;
          zpsupsatsrce[NCLDQI] = psupsat_;
          // Add ice to first guess for deposition term
          zqxfg[NCLDQI] += psupsat_;
          // Store cloud budget diagnostics if required
        }

        // Increase cloud amount using RKOOPTAU timescale
        zsolac = (real_t(1) - za) * zfaci;
        // Store cloud budget diagnostics if required
      }

      // 3.2  DETRAINMENT FROM CONVECTION
      // * Diagnostic T-ice/liq split retained for convection
      //    Note: This link is now flexible and a future convection
      //    scheme can detrain explicit seperate budgets of:
      //    cloud water, ice, rain and snow
      // * There is no (1-ZA) multiplier term on the cloud detrainment
      //    term, since is now written in mass-flux terms
      // [#Note: Should use ZFOEALFACU used in convection rather than ZFOEALFA]
      if (jk < klev - 1 && jk >= yrecldp::ncldtop - 1) {

        plude_ *= zdtgdp;

        real_t plu_ = at_shared<nproma>(SHARED::PLU, jk);
        bool ldcum_ = at_shared<nproma>(SHARED::LDCUM);
        if (ldcum_ && plude_ > yrecldp::rlmin && plu_ > zepsec) {

          zsolac = zsolac + plude_ / plu_;
          // *diagnostic temperature split*
          zalfaw = zfoealfa;
          zconvsrce[NCLDQL] = zalfaw * plude_;
          zconvsrce[NCLDQI] = (real_t(1) - zalfaw) * plude_;
          zsolqa[NCLDQL][NCLDQL] += zconvsrce[NCLDQL];
          zsolqa[NCLDQI][NCLDQI] += zconvsrce[NCLDQI];

        } else {

          plude_ = real_t(0);
        }
        // *convective snow detrainment source
        if (ldcum_)
          zsolqa[NCLDQS][NCLDQS] +=
              at_shared<nproma>(SHARED::PSNDE, jk) * zdtgdp;
      }
      //  3.3  SUBSIDENCE COMPENSATING CONVECTIVE UPDRAUGHTS
      // Three terms:
      // * Convective subsidence source of cloud from layer above
      // * Evaporation of cloud within the layer
      // * Subsidence sink of cloud to the layer below (Implicit solution)

      //-----------------------------------------------
      // Subsidence source from layer above
      //               and
      // Evaporation of cloud within the layer
      if (jk >= yrecldp::ncldtop) {

        // Now have to work out how much liquid evaporates at arrival point
        // since there is no prognostic memory for in-cloud humidity, i.e.
        // we always assume cloud is saturated.

        real_t zdtdp = (zrdcp * real_t(0.5) * (ztp1_prev + ztp1)) / paph_;
        real_t zdtforc = zdtdp * (pap_ - pap_prev);
        //[#Note: Diagnostic mixed phase should be replaced below]
        real_t zdqs = zanewm1 * zdtforc * zdqsmixdt;

        real_t zmf = max(real_t(0), pmf_ * zdtgdp);

        real_t zlfinalsum = real_t(0);

#pragma unroll
        for (int jm = 0; jm < NCLV - 1; ++jm) {
          if (!llfall(static_cast<type>(jm)) &&
              iphase(static_cast<type>(jm)) != VAPOUR) {
            real_t zlcust = zmf * zqxnm1[jm];
            // record total flux for enthalpy budget:
            zconvsrce[jm] = zconvsrce[jm] + zlcust;

            real_t zlfinal = max(real_t(0), zlcust - zdqs); // lim to zero
            // no supersaturation allowed incloud ---v
            real_t zevap = min((zlcust - zlfinal), zevaplimmix);
            zlfinal = zlcust - zevap;
            zlfinalsum = zlfinalsum + zlfinal; // sum

            zsolqa[jm][jm] += zlcust; // whole sum
            zsolqa[NCLDQV][jm] += zevap;
            zsolqa[jm][NCLDQV] -= zevap;
          }
        }

        //  Reset the cloud contribution if no cloud water survives to this
        //  level:
        real_t zacust = zmf * zanewm1;
        if (zlfinalsum < zepsec)
          zacust = real_t(0);
        zsolac += zacust;
      }

      real_t pmf_next;
      // Subsidence sink of cloud to the layer below
      // (Implicit - re. CFL limit on convective mass flux)
      if (jk < klev - 1) {
        pmf_next = at_shared<nproma>(SHARED::PMFU, jk) +
                   at_shared<nproma>(SHARED::PMFD, jk);

        real_t zmfdn = max(real_t(0), pmf_next * zdtgdp);

        zsolab = zsolab + zmfdn;
        zsolqb[NCLDQL][NCLDQL] += zmfdn;
        zsolqb[NCLDQI][NCLDQI] += zmfdn;

        // record sink for cloud budget and enthalpy budget diagnostics
        zconvsink[NCLDQL] = zmfdn;
        zconvsink[NCLDQI] = zmfdn;
      }

      // 3.4  EROSION OF CLOUDS BY TURBULENT MIXING
      // NOTE: In default tiedtke scheme this process decreases the cloud
      //       area but leaves the specific cloud water content
      //       within clouds unchanged

      // Define turbulent erosion rate
      real_t zldifdt = yrecldp::rcldiff * ptsphy; // original version
      // Increase by factor of 5 for convective points
      if (at_shared<nproma>(SHARED::KTYPE) > 0 && plude_ > zepsec)
        zldifdt = yrecldp::rcldiff_convi * zldifdt;

      // At the moment, works on mixed RH profile and partitioned ice/liq
      // fraction so that it is similar to previous scheme Should apply RHw for
      // liquid cloud and RHi for ice cloud separately
      if (zli > zepsec) {
        // Calculate environmental humidity
        //      ZQE=(ZQX(NCLDQV)-ZA*ZQSMIX)/&
        //    &      MAX(ZEPSEC,1.0_JPRB-ZA)
        //      ZE=ZLDIFDT(JL)*MAX(ZQSMIX-ZQE,0.0_JPRB)
        real_t ze = zldifdt * max(zqsmix - zqx[NCLDQV], real_t(0));
        real_t zleros = za * ze;
        zleros = min(zleros, zevaplimmix);
        zleros = min(zleros, zli);
        real_t zaeros = zleros / zlicld; // if linear term

        // Erosion is -ve LINEAR in L,A
        zsolac = zsolac - zaeros; // linear

        zsolqa[NCLDQV][NCLDQL] += zliqfrac * zleros;
        zsolqa[NCLDQL][NCLDQV] -= zliqfrac * zleros;
        zsolqa[NCLDQV][NCLDQI] += zicefrac * zleros;
        zsolqa[NCLDQI][NCLDQV] -= zicefrac * zleros;
      }

      // 3.4  CONDENSATION/EVAPORATION DUE TO DQSAT/DT
      // calculate dqs/dt
      // Note: For the separate prognostic Qi and Ql, one would ideally use
      // Qsat/DT wrt liquid/Koop here, since the physics is that new clouds
      // forms by liquid droplets [liq] or when aqueous aerosols [Koop] form.
      // These would then instantaneous freeze if T<-38C or lead to ice growth
      // by deposition in warmer mixed phase clouds.  However, since we do
      // not have a separate prognostic equation for in-cloud humidity or a
      // statistical scheme approach in place, the depositional growth of ice
      // in the mixed phase can not be modelled and we resort to supersaturation
      // wrt ice instanteously converting to ice over one timestep
      // (see Tompkins et al. QJRMS 2007 for details)
      // Thus for the initial implementation the diagnostic mixed phase is
      // retained for the moment, and the level of approximation noted.

      real_t zdtdp = (zrdcp * ztp1) / pap_;
      real_t zdpmxdt = zdp * zqtmst;
      real_t zmfdn = real_t(0);
      if (jk < klev - 1)
        zmfdn = pmf_next;
      real_t zwtot = at_shared<nproma>(SHARED::PVERVEL, jk) +
                     real_t(0.5) * yomcst::rg * (pmf_ + zmfdn);
      if (jk < klev - 1)
        pmf_ = pmf_next;
      zwtot = min(zdpmxdt, max(-zdpmxdt, zwtot));
      real_t zzzdt = at_shared<nproma>(SHARED::PHRSW, jk) +
                     at_shared<nproma>(SHARED::PHRLW, jk);
      real_t zdtdiab =
          min(zdpmxdt * zdtdp, max(-zdpmxdt * zdtdp, zzzdt)) * ptsphy +
          yoethf::ralfdcp * zldefr;
      // Note: ZLDEFR should be set to the difference between the mixed phase
      // functions in the convection and cloud scheme, but this is not
      // calculated, so is zero and the functions must be the same
      real_t zdtforc = zdtdp * zwtot * ptsphy + zdtdiab;
      real_t zqold = zqsmix;
      real_t ztold = ztp1;
      ztp1 += zdtforc;
      ztp1 = max(ztp1, real_t(160));

      // Formerly a call to CUADJTQ(..., ICALL=5)
      real_t zqp = real_t(1) / pap_;
      real_t zqsat = foeewm(ztp1) * zqp;
      zqsat = min(real_t(0.5), zqsat);
      zcor = real_t(1) / (real_t(1) - yomcst::retv * zqsat);
      zqsat = zqsat * zcor;
      real_t zcond =
          (zqsmix - zqsat) / (real_t(1) + zqsat * zcor * foedem(ztp1));
      ztp1 += foeldcpm(ztp1) * zcond;
      zqsmix -= zcond;
      zqsat = foeewm(ztp1) * zqp;
      zqsat = min(real_t(0.5), zqsat);
      zcor = real_t(1) / (real_t(1) - yomcst::retv * zqsat);
      zqsat = zqsat * zcor;
      real_t zcond1 =
          (zqsmix - zqsat) / (real_t(1) + zqsat * zcor * foedem(ztp1));
      ztp1 = ztp1 + foeldcpm(ztp1) * zcond1;
      zqsmix = zqsmix - zcond1;

      real_t zdqs = zqsmix - zqold;
      zqsmix = zqold;
      ztp1 = ztold;

      // 3.4a  ZDQS(JL) > 0:  EVAPORATION OF CLOUDS
      // ----------------------------------------------------------------------
      // Erosion term is LINEAR in L
      // Changed to be uniform distribution in cloud region

      // Previous function based on DELTA DISTRIBUTION in cloud:
      if (zdqs > real_t(0)) {
        // If subsidence evaporation term is turned off, then need to use
        // updated liquid and cloud here? ZLEVAP =
        // MAX(ZA+ZACUST(JL),1.0_JPRB)*MIN(ZDQS(JL),ZLICLD(JL)+ZLFINALSUM(JL))
        real_t zlevap = za * min(zdqs, zlicld);
        zlevap = min(zlevap, zevaplimmix);
        zlevap = min(zlevap, max(zqsmix - zqx[NCLDQV], real_t(0)));

        zsolqa[NCLDQV][NCLDQL] += zliqfrac * zlevap;
        zsolqa[NCLDQL][NCLDQV] -= zliqfrac * zlevap;

        zsolqa[NCLDQV][NCLDQI] += zicefrac * zlevap;
        zsolqa[NCLDQI][NCLDQV] -= zicefrac * zlevap;
      }

      // 3.4b ZDQS(JL) < 0: FORMATION OF CLOUDS
      // (1) Increase of cloud water in existing clouds
      if (za > zepsec && zdqs <= -yrecldp::rlmin) {

        real_t zlcond1 = max(-zdqs, real_t(0)); // new limiter

        // old limiter (significantly improves upper tropospheric humidity rms)
        real_t zcdmax;
        if (za > real_t(0.99)) {
          zcor = real_t(1) / (real_t(1) - yomcst::retv * zqsmix);
          zcdmax = (zqx[NCLDQV] - zqsmix) /
                   (real_t(1) + zcor * zqsmix * foedem(ztp1));
        } else {
          zcdmax = (zqx[NCLDQV] - za * zqsmix) / za;
        }
        zlcond1 = max(min(zlcond1, zcdmax), real_t(0));
        // end old limiter

        zlcond1 = za * zlcond1;
        if (zlcond1 < yrecldp::rlmin)
          zlcond1 = real_t(0);

        // All increase goes into liquid unless so cold cloud homogeneously
        // freezes Include new liquid formation in first guess value, otherwise
        // liquid remains at cold temperatures until next timestep.
        if (ztp1 > yrecldp::rthomo) {
          zsolqa[NCLDQL][NCLDQV] += zlcond1;
          zsolqa[NCLDQV][NCLDQL] -= zlcond1;
          zqxfg[NCLDQL] += zlcond1;
        } else {
          zsolqa[NCLDQI][NCLDQV] += zlcond1;
          zsolqa[NCLDQV][NCLDQI] -= zlcond1;
          zqxfg[NCLDQI] += zlcond1;
        }
      }

      // (2) Generation of new clouds (da/dt>0)

      if (zdqs <= -yrecldp::rlmin && za < real_t(1) - zepsec) {

        // Critical relative humidity
        real_t zrhc = yrecldp::ramid;
        real_t zsigk = pap_ / paph_top;
        // Increase RHcrit to 1.0 towards the surface (eta>0.8)
        if (zsigk > real_t(0.8))
          zrhc = yrecldp::ramid + (real_t(1) - yrecldp::ramid) *
                                      sqr((zsigk - real_t(0.8)) / real_t(0.2));

        // Supersaturation options
        real_t zqe;
        if (yrecldp::nssopt == 0) {
          // No scheme
          zqe = (zqx[NCLDQV] - za * zqsice) / max(zepsec, real_t(1) - za);
          zqe = max(real_t(0), zqe);
        } else if (yrecldp::nssopt == 1) {
          // Tompkins
          zqe = (zqx[NCLDQV] - za * zqsice) / max(zepsec, real_t(1) - za);
          zqe = max(real_t(0), zqe);
        } else if (yrecldp::nssopt == 2) {
          // Lohmann and Karcher
          zqe = zqx[NCLDQV];
        } else if (yrecldp::nssopt == 3) {
          // Gierens
          zqe = zqx[NCLDQV] + zli;
        }

        if (ztp1 >= yomcst::rtt || yrecldp::nssopt == 0) {
          // No ice supersaturation allowed
          zfac = real_t(1);
        } else {
          // Ice supersaturation
          zfac = zfokoop;
        }

        if (zqe >= zrhc * zqsice * zfac && zqe < zqsice * zfac) {
          // note: not **2 on 1-a term if ZQE is used.
          // Added correction term ZFAC to numerator 15/03/2010
          real_t zacond = -((real_t(1) - za) * zfac * zdqs) /
                          max(real_t(2) * (zfac * zqsice - zqe), zepsec);

          zacond = min(zacond, real_t(1) - za); // put the limiter back

          // Linear term:
          // Added correction term ZFAC 15/03/2010
          real_t zlcond2 = -zfac * zdqs * real_t(0.5) * zacond; // mine linear

          // new limiter formulation
          real_t zzdl =
              (real_t(2) * (zfac * zqsice - zqe)) / max(zepsec, real_t(1) - za);
          // Added correction term ZFAC 15/03/2010
          if (zfac * zdqs < -zzdl) {
            real_t zlcondlim =
                (za - real_t(1)) * zfac * zdqs - zfac * zqsice + zqx[NCLDQV];
            zlcond2 = min(zlcond2, zlcondlim);
          }
          zlcond2 = max(zlcond2, real_t(0));

          if (zlcond2 < yrecldp::rlmin || (real_t(1) - za) < zepsec) {
            zlcond2 = real_t(0);
            zacond = real_t(0);
          }
          if (zlcond2 == real_t(0))
            zacond = real_t(0);

          // Large-scale generation is LINEAR in A and LINEAR in L
          zsolac = zsolac + zacond; // linear

          // All increase goes into liquid unless so cold cloud homogeneously
          // freezes Include new liquid formation in first guess value,
          // otherwise liquid remains at cold temperatures until next timestep.
          if (ztp1 > yrecldp::rthomo) {
            zsolqa[NCLDQL][NCLDQV] += zlcond2;
            zsolqa[NCLDQV][NCLDQL] -= zlcond2;
            zqxfg[NCLDQL] += zlcond2;
          } else {
            // homogeneous freezing
            zsolqa[NCLDQI][NCLDQV] += zlcond2;
            zsolqa[NCLDQV][NCLDQI] -= zlcond2;
            zqxfg[NCLDQI] += zlcond2;
          }
        }
      }

      // 3.7 Growth of ice by vapour deposition
      // Following Rotstayn et al. 2001:
      // does not use the ice nuclei number from cloudaer.F90
      // but rather a simple Meyers et al. 1992 form based on the
      // supersaturation and assuming clouds are saturated with
      // respect to liquid water (well mixed), (or Koop adjustment)
      // Growth considered as sink of liquid water if present so
      // Bergeron-Findeisen adjustment in autoconversion term no longer needed

      //- Ice deposition following Rotstayn et al. (2001)
      //-  (monodisperse ice particle size distribution)
      if (idepice == 1) {

        // Calculate distance from cloud top
        // defined by cloudy layer below a layer with cloud frac <0.01
        // ZDZ = ZDP(JL)/(ZRHO(JL)*RG)

        if (za_prev < yrecldp::rcldtopcf && za >= yrecldp::rcldtopcf)
          zcldtopdist = 0;
        else
          zcldtopdist += zdp / (zrho * yomcst::rg);

        // only treat depositional growth if liquid present. due to fact
        // that can not model ice growth from vapour without additional
        // in-cloud water vapour variable
        if (ztp1 < yomcst::rtt && zqxfg[NCLDQL] > yrecldp::rlmin) {
          // T<273K

          real_t zvpice = (foeeice(ztp1) * yomcst::rv) / yomcst::rd;
          real_t zvpliq = zvpice * zfokoop;
          real_t zicenuclei =
              real_t(1000) *
              exp((real_t(12.96) * (zvpliq - zvpice)) / zvpliq - real_t(0.639));

          //  2.4e-2 is conductivity of air
          //  8.8 = 700**1/3 = density of ice to the third
          real_t zadd = (yomcst::rlstt *
                         (yomcst::rlstt / ((yomcst::rv * ztp1)) - real_t(1))) /
                        ((real_t(2.4e-2) * ztp1));
          real_t zbdd = (yomcst::rv * ztp1 * pap_) / ((real_t(2.21) * zvpice));
          real_t zcvds = (real_t(7.8) * pow(zicenuclei / zrho, real_t(0.666)) *
                          (zvpliq - zvpice)) /
                         ((real_t(8.87) * (zadd + zbdd) * zvpice));

          // RICEINIT=1.E-12_JPRB is initial mass of ice particle
          real_t zice0 = max(zicecld, (zicenuclei * yrecldp::riceinit) / zrho);

          // new value of ice:
          real_t zinew =
              pow((real_t(0.666) * zcvds * ptsphy + pow(zice0, real_t(0.666))),
                  real_t(1.5));

          // grid-mean deposition rate:
          real_t zdepos = max(za * (zinew - zice0), real_t(0));

          // Limit deposition to liquid water amount
          // If liquid is all frozen, ice would use up reservoir of water
          // vapour in excess of ice saturation mixing ratio - However this
          // can not be represented without a in-cloud humidity variable. Using
          // the grid-mean humidity would imply a large artificial horizontal
          // flux from the clear sky to the cloudy area. We thus rely on the
          // supersaturation check to clean up any remaining supersaturation
          zdepos = min(zdepos, zqxfg[NCLDQL]); // limit to liquid water amount

          // At top of cloud, reduce deposition rate near cloud top to account
          // for small scale turbulent processes, limited ice nucleation and ice
          // fallout
          //      ZDEPOS =
          //      ZDEPOS*MIN(RDEPLIQREFRATE+ZCLDTOPDIST(JL)/RDEPLIQREFDEPTH,1.0_JPRB)
          // Change to include dependence on ice nuclei concentration
          // to increase deposition rate with decreasing temperatures
          real_t zinfactor = min(zicenuclei / real_t(15000), real_t(1));
          zdepos = zdepos *
                   min(zinfactor + (real_t(1) - zinfactor) *
                                       (yrecldp::rdepliqrefrate +
                                        zcldtopdist / yrecldp::rdepliqrefdepth),
                       real_t(1));

          // add to matrix
          zsolqa[NCLDQI][NCLDQL] += zdepos;
          zsolqa[NCLDQL][NCLDQI] -= zdepos;
          zqxfg[NCLDQI] += zdepos;
          zqxfg[NCLDQL] -= zdepos;
        }
        // Ice deposition assuming ice PSD
      } else if (idepice == 2) {
        // Calculate distance from cloud top
        // defined by cloudy layer below a layer with cloud frac <0.01
        // ZDZ = ZDP(JL)/(ZRHO(JL)*RG)

        if (za_prev < yrecldp::rcldtopcf && za >= yrecldp::rcldtopcf)
          zcldtopdist = real_t(0);
        else
          zcldtopdist = zcldtopdist + zdp / ((zrho * yomcst::rg));

        // only treat depositional growth if liquid present. due to fact
        // that can not model ice growth from vapour without additional
        // in-cloud water vapour variable
        if (ztp1 < yomcst::rtt && zqxfg[NCLDQL] > yrecldp::rlmin) {
          // T<273K

          real_t zvpice = (foeeice(ztp1) * yomcst::rv) / yomcst::rd;
          real_t zvpliq = zvpice * zfokoop;
          real_t zicenuclei =
              real_t(1000) *
              exp((real_t(12.96) * (zvpliq - zvpice)) / zvpliq - real_t(0.639));

          // RICEINIT=1.E-12_JPRB is initial mass of ice particle
          real_t zice0 = max(zicecld, (zicenuclei * yrecldp::riceinit) / zrho);

          // Particle size distribution
          real_t ztcg = real_t(1);
          real_t zfacx1i = real_t(1);

          real_t zaplusb = yrecldp::rcl_apb1 * zvpice -
                           yrecldp::rcl_apb2 * zvpice * ztp1 +
                           pap_ * yrecldp::rcl_apb3 * pow(ztp1, real_t(3));
          real_t zcorrfac = pow(real_t(1) / zrho, real_t(0.5));
          real_t zcorrfac2 = pow(ztp1 / real_t(273.0), real_t(1.5)) *
                             (real_t(393) / (ztp1 + real_t(120)));

          real_t zpr02 =
              (zrho * zice0 * yrecldp::rcl_const1i) / ((ztcg * zfacx1i));

          real_t zterm1 = ((zvpliq - zvpice) * sqr(ztp1) * zvpice * zcorrfac2 *
                           ztcg * yrecldp::rcl_const2i * zfacx1i) /
                          ((zrho * zaplusb * zvpice));
          real_t zterm2 =
              real_t(.65) * yrecldp::rcl_const6i *
                  pow(zpr02, yrecldp::rcl_const4i) +
              (yrecldp::rcl_const3i * pow(zcorrfac, real_t(0.5)) *
               pow(zrho, real_t(0.5)) * pow(zpr02, yrecldp::rcl_const5i)) /
                  pow(zcorrfac2, real_t(0.5));

          real_t zdepos = max(za * zterm1 * zterm2 * ptsphy, real_t(0));

          // Limit deposition to liquid water amount
          // If liquid is all frozen, ice would use up reservoir of water
          // vapour in excess of ice saturation mixing ratio - However this
          // can not be represented without a in-cloud humidity variable. Using
          // the grid-mean humidity would imply a large artificial horizontal
          // flux from the clear sky to the cloudy area. We thus rely on the
          // supersaturation check to clean up any remaining supersaturation
          zdepos = min(zdepos, zqxfg[NCLDQL]); // limit to liquid water amount

          // At top of cloud, reduce deposition rate near cloud top to account
          // for small scale turbulent processes, limited ice nucleation and ice
          // fallout Change to include dependence on ice nuclei concentration to
          // increase deposition rate with decreasing temperatures
          real_t zinfactor = min(zicenuclei / real_t(15000), real_t(1));
          zdepos = zdepos *
                   min(zinfactor + (real_t(1) - zinfactor) *
                                       (yrecldp::rdepliqrefrate +
                                        zcldtopdist / yrecldp::rdepliqrefdepth),
                       real_t(1));

          // add to matrix
          zsolqa[NCLDQI][NCLDQL] += zdepos;
          zsolqa[NCLDQL][NCLDQI] -= zdepos;
          zqxfg[NCLDQI] += zdepos;
          zqxfg[NCLDQL] -= zdepos;
        }
      }

      //////
      //              4  *** PRECIPITATION PROCESSES ***

      //----------------------------------
      // revise in-cloud consensate amount
      //----------------------------------
      ztmpa = real_t(1) / max(za, zepsec);
      zliqcld = zqxfg[NCLDQL] * ztmpa;
      zicecld = zqxfg[NCLDQI] * ztmpa;
      zlicld = zliqcld + zicecld;

      // 4.2 SEDIMENTATION/FALLING OF *ALL* MICROPHYSICAL SPECIES
      //     now that rain, snow, graupel species are prognostic
      //     the precipitation flux can be defined directly level by level
      //     There is no vertical memory required from the flux variable

#pragma unroll
      for (int jm = 0; jm < NCLV - 1; ++jm) {
        if (llfall(static_cast<type>(jm)) || jm == NCLDQI) {
          // source from layer above
          if (jk > yrecldp::ncldtop - 1) {
            zfallsrce[jm] = zpfplsx[jm] * zdtgdp;
            zsolqa[jm][jm] += zfallsrce[jm];
            zqxfg[jm] += zfallsrce[jm];
            // use first guess precip
            zqpretot += zqxfg[jm];
          }
          // sink to next layer, constant fall speed
          // if aerosol effect then override
          //  note that for T>233K this is the same as above.
          real_t zfall;
          if (yrecldp::laericesed &&
              jm == NCLDQI) { // TODO no memcpy_async because false
            real_t zre_ice = pre_ice[get_index(descriptor_k, nproma, jk)];
            // the exponent value is from
            // morrison et al. jas 2005 appendix
            zfall = real_t(0.002) * zre_ice; // ** 1.0
            zfall = zfall * zrho;
          } else {
            zfall = zvqx(static_cast<type>(jm)) * zrho;
          }
          // modified by Heymsfield and Iaquinta JAS 2000

          zfallsink[jm] = zdtgdp * zfall;
          // Cloud budget diagnostic stored at end as implicit
        }
      }
      // Precip cover overlap using MAX-RAN Overlap
      // Since precipitation is now prognostic we must
      //   1) apply an arbitrary minimum coverage (0.3) if precip>0
      //   2) abandon the 2-flux clr/cld treatment
      //   3) Thus, since we have no memory of the clear sky precip
      //      fraction, we mimic the previous method by reducing
      //      ZCOVPTOT(JL), which has the memory, proportionally with
      //      the precip evaporation rate, taking cloud fraction
      //      into account
      //   #3 above leads to much smoother vertical profiles of
      //   precipitation fraction than the Klein-Jakob scheme which
      //   monotonically increases precip fraction and then resets
      //   it to zero in a step function once clear-sky precip reaches
      //   zero.
      real_t zraincld;
      real_t zsnowcld;
      real_t zcovpclr;
      if (zqpretot > zepsec) {
        zcovptot = real_t(1) -
                   ((real_t(1) - zcovptot) * (real_t(1) - max(za, za_prev))) /
                       (real_t(1) - min(za_prev, real_t(1) - real_t(1.e-6)));
        zcovptot = max(zcovptot, yrecldp::rcovpmin);

        zcovpclr = max(real_t(0), zcovptot - za); // clear sky proportion
        zraincld = zqxfg[NCLDQR] / zcovptot;
        zsnowcld = zqxfg[NCLDQS] / zcovptot;
        zcovpmax = max(zcovptot, zcovpmax);
      } else {
        zcovptot = real_t(0); // no flux - reset cover

        zcovpclr = real_t(0); // reset clear sky proportion
        zraincld = real_t(0);
        zsnowcld = real_t(0);
        zcovpmax = real_t(0); // reset max cover for zzrh calc
      }

      // 4.3a AUTOCONVERSION TO SNOW
      if (ztp1 <= yomcst::rtt) {
        // Snow Autoconversion rate follow Lin et al. 1983
        if (zicecld > zepsec) {

          real_t zlcrit, zzco;
          zzco = ptsphy * yrecldp::rsnowlin1 *
                 exp(yrecldp::rsnowlin2 * (ztp1 - yomcst::rtt));

          if (yrecldp::laericeauto) { // TODO no memcpy_async because false
            zlcrit = picrit_aer[get_index(descriptor_k, nproma, jk)];
            zzco = zzco * pow(yrecldp::rnice /
                                  pnice[get_index(descriptor_k, nproma, jk)],
                              real_t(0.333));
          } else {
            zlcrit = yrecldp::rlcritsnow;
          }

          real_t zsnowaut = zzco * (real_t(1) - exp(-sqr(zicecld / zlcrit)));
          zsolqb[NCLDQS][NCLDQI] += zsnowaut;
        }
      }

      // 4.3b AUTOCONVERSION WARM CLOUDS
      //   Collection and accretion will require separate treatment
      //   but for now we keep this simple treatment
      if (zliqcld > zepsec) {

        if (iwarmrain == 1) {
          real_t zzco = yrecldp::rkconv * ptsphy;
          real_t zlcrit;
          if (yrecldp::laerliqautolsp) { // TODO no memcpy_async because false
            zlcrit = plcrit_aer[get_index(descriptor_k, nproma, jk)];
            // 0.3 = n**0.333 with n=125 cm-3
            zzco = zzco * pow(yrecldp::rccn /
                                  pccn[get_index(descriptor_k, nproma, jk)],
                              real_t(0.333));
          } else {
            // Modify autoconversion threshold dependent on:
            //  land (polluted, high CCN, smaller droplets, higher threshold)
            //  sea  (clean, low CCN, larger droplets, lower threshold)
            if (at_shared<nproma>(SHARED::PLSM) > real_t(0.5))
              zlcrit = yrecldp::rclcrit_land; // land
            else
              zlcrit = yrecldp::rclcrit_sea; // ocean
          }

          // Parameters for cloud collection by rain and snow.
          // Note that with new prognostic variable it is now possible
          // to REPLACE this with an explicit collection parametrization
          real_t zprecip =
              (zpfplsx[NCLDQS] + zpfplsx[NCLDQR]) / max(zepsec, zcovptot);
          real_t zcfpr =
              real_t(1) + yrecldp::rprc1 * sqrt(max(zprecip, real_t(0)));

          if (yrecldp::laerliqcoll) { // TODO no memcpy_async because false
            // 5.0 = n**0.333 with n=125 cm-3
            zcfpr *=
                pow(yrecldp::rccn / pccn[get_index(descriptor_k, nproma, jk)],
                    real_t(0.333));
          }

          zzco *= zcfpr;
          zlcrit /= max(zcfpr, zepsec);

          real_t zrainaut;
          if (zliqcld / zlcrit < real_t(20))
            // Security for exp for some compilers
            zrainaut = zzco * (real_t(1) - exp(-sqr(zliqcld / zlcrit)));
          else
            zrainaut = zzco;

          // rain freezes instantly
          if (ztp1 <= yomcst::rtt)
            zsolqb[NCLDQS][NCLDQL] += zrainaut;
          else
            zsolqb[NCLDQR][NCLDQL] += zrainaut;

          // Warm-rain process follow Khairoutdinov and Kogan (2000)
        } else if (iwarmrain == 2) {

          real_t zconst, zlcrit;
          if (at_shared<nproma>(SHARED::PLSM) > real_t(0.5)) {
            // land
            zconst = yrecldp::rcl_kk_cloud_num_land;
            zlcrit = yrecldp::rclcrit_land;
          } else {
            // ocean
            zconst = yrecldp::rcl_kk_cloud_num_sea;
            zlcrit = yrecldp::rclcrit_sea;
          }

          real_t zrainacc; // currently needed for diags
          real_t zrainaut; // currently needed for diags
          if (zliqcld > zlcrit) {

            zrainaut = real_t(1.5) * za * ptsphy * yrecldp::rcl_kkaau *
                       pow(zliqcld, yrecldp::rcl_kkbauq) *
                       pow(zconst, yrecldp::rcl_kkbaun);

            zrainaut = min(zrainaut, zqxfg[NCLDQL]);
            if (zrainaut < zepsec)
              zrainaut = real_t(0);

            zrainacc = real_t(2) * za * ptsphy * yrecldp::rcl_kkaac *
                       pow(zliqcld * zraincld, yrecldp::rcl_kkbac);

            zrainacc = min(zrainacc, zqxfg[NCLDQL]);
            if (zrainacc < zepsec)
              zrainacc = real_t(0);

          } else {
            zrainaut = real_t(0);
            zrainacc = real_t(0);
          }

          // If temperature < 0, then autoconversion produces snow rather than
          // rain Explicit
          if (ztp1 <= yomcst::rtt) {
            zsolqa[NCLDQS][NCLDQL] += zrainaut;
            zsolqa[NCLDQL][NCLDQS] -= zrainaut;

            zsolqa[NCLDQS][NCLDQL] += zrainacc;
            zsolqa[NCLDQL][NCLDQS] -= zrainacc;
          } else {
            zsolqa[NCLDQR][NCLDQL] += zrainaut;
            zsolqa[NCLDQL][NCLDQR] -= zrainaut;

            zsolqa[NCLDQR][NCLDQL] += zrainacc;
            zsolqa[NCLDQL][NCLDQR] -= zrainacc;
          }
        }
      }
      // RIMING - COLLECTION OF CLOUD LIQUID DROPS BY SNOW AND ICE
      //      only active if T<0degC and supercooled liquid water is present
      //      AND if not Sundquist autoconversion (as this includes riming)
      if (iwarmrain > 1) {

        if (ztp1 <= yomcst::rtt && zliqcld > zepsec) {

          // Fallspeed air density correction
          real_t zfallcorr = pow(yrecldp::rdensref / zrho, real_t(0.4));

          // Riming of snow by cloud water - implicit in lwc
          if (zsnowcld > zepsec && zcovptot > real_t(0.01)) {

            // Calculate riming term
            // Factor of liq water taken out because implicit
            real_t zsnowrime = real_t(0.3) * zcovptot * ptsphy *
                               yrecldp::rcl_const7s * zfallcorr *
                               pow(zrho * zsnowcld * yrecldp::rcl_const1s,
                                   yrecldp::rcl_const8s);

            // Limit snow riming term
            zsnowrime = min(zsnowrime, real_t(1));

            zsolqb[NCLDQS][NCLDQL] += zsnowrime;
          }

          // Riming of ice by cloud water - implicit in lwc
          // NOT YET ACTIVE
          //      IF (ZICECLD(JL)>ZEPSEC .AND. ZA>0.01_JPRB) THEN
          //
          //        ! Calculate riming term
          //        ! Factor of liq water taken out because implicit
          //        ZSNOWRIME(JL) = ZA*PTSPHY*RCL_CONST7S*ZFALLCORR &
          //     & *(ZRHO(JL)*ZICECLD(JL)*RCL_CONST1S)**RCL_CONST8S
          //
          //        ! Limit ice riming term
          //        ZSNOWRIME(JL)=MIN(ZSNOWRIME(JL),1.0_JPRB)
          //
          //        ZSOLQB(JL,NCLDQI,NCLDQL) = ZSOLQB(JL,NCLDQI,NCLDQL) +
          //        ZSNOWRIME(JL)
          //
          //      ENDIF
        }
      }
      // 4.4a  MELTING OF SNOW and ICE
      //       with new implicit solver this also has to treat snow or ice
      //       precipitating from the level above... i.e. local ice AND flux.
      //       in situ ice and snow: could arise from LS advection or warming
      //       falling ice and snow: arrives by precipitation process
      zicetot = zqxfg[NCLDQI] + zqxfg[NCLDQS];
      real_t zmeltmax = real_t(0);

      // If there are frozen hydrometeors present and dry-bulb temperature >
      // 0degC
      if (zicetot > zepsec && ztp1 > yomcst::rtt) {

        // Calculate subsaturation
        real_t zsubsat = max(zqsice - zqx[NCLDQV], real_t(0));

        // Calculate difference between dry-bulb (ZTP1, IBL) and the
        // temperature at which the wet-bulb=0degC (RTT-ZSUBSAT*....) using an
        // approx. Melting only occurs if the wet-bulb temperature >0 i.e.
        // warming of ice particle due to melting > cooling due to
        // evaporation.
        real_t ztdmtw0 =
            ztp1 - yomcst::rtt -
            zsubsat * (ZTW1 + ZTW2 * (pap_ - ZTW3) - ZTW4 * (ztp1 - ZTW5));
        // Not implicit yet...
        // Ensure ZCONS1 is positive so that ZMELTMAX=0 if ZTDMTW0<0
        real_t zcons1 = abs((ptsphy * (real_t(1) + real_t(0.5) * ztdmtw0)) /
                            yrecldp::rtaumel);
        zmeltmax = max(ztdmtw0 * zcons1 * zrldcp, real_t(0));
      }

      // Loop over frozen hydrometeors (ice, snow)
#pragma unroll
      for (int jm = 0; jm < NCLV - 1; ++jm) {
        if (iphase(static_cast<type>(jm)) == ICE) {
          int jn = imelt(static_cast<type>(jm));
          if (zmeltmax > zepsec && zicetot > zepsec) {
            // Apply melting in same proportion as frozen hydrometeor
            // fractions
            real_t zalfa = zqxfg[jm] / zicetot;
            real_t zmelt = min(zqxfg[jm], zalfa * zmeltmax);
            // needed in first guess
            // This implies that zqpretot has to be recalculated below
            // since is not conserved here if ice falls and liquid doesn't
            zqxfg[jm] -= zmelt;
            zqxfg[jn] += zmelt;
            zsolqa[jn][jm] += zmelt;
            zsolqa[jm][jn] -= zmelt;
          }
        }
      }

      // 4.4b  FREEZING of RAIN

      // If rain present
      if (zqx[NCLDQR] > zepsec) {

        if (ztp1 <= yomcst::rtt && ztp1_prev > yomcst::rtt) {
          // Base of melting layer/top of refreezing layer so
          // store rain/snow fraction for precip type diagnosis
          // If mostly rain, then supercooled rain slow to freeze
          // otherwise faster to freeze (snow or ice pellets)
          zqpretot = max(zqx[NCLDQS] + zqx[NCLDQR], zepsec);
          prainfrac_toprfz_ = zqx[NCLDQR] / zqpretot;
        }

        // If temperature less than zero
        if (ztp1 < yomcst::rtt) {

          real_t zfrzmax;
          if (prainfrac_toprfz_ > real_t(0.8)) {

            // Majority of raindrops completely melted
            // Refreezing is by slow heterogeneous freezing

            // Slope of rain particle size distribution
            real_t zlambda = pow(yrecldp::rcl_fac1 / ((zrho * zqx[NCLDQR])),
                                 yrecldp::rcl_fac2);

            // Calculate freezing rate based on Bigg(1953) and Wisner(1972)
            real_t ztemp = yrecldp::rcl_fzrab * (ztp1 - yomcst::rtt);
            real_t zfrz = ptsphy * (yrecldp::rcl_const5r / zrho) *
                          (exp(ztemp) - real_t(1)) *
                          pow(zlambda, yrecldp::rcl_const6r);
            zfrzmax = max(zfrz, real_t(0));

          } else {

            // Majority of raindrops only partially melted
            // Refreeze with a shorter timescale (reverse of melting...for
            // now)

            real_t zcons1 = abs(
                (ptsphy * (real_t(1) + real_t(0.5) * (yomcst::rtt - ztp1))) /
                yrecldp::rtaumel);
            zfrzmax = max((yomcst::rtt - ztp1) * zcons1 * zrldcp, real_t(0));
          }

          if (zfrzmax > zepsec) {
            real_t zfrz = min(zqx[NCLDQR], zfrzmax);
            zsolqa[NCLDQS][NCLDQR] += zfrz;
            zsolqa[NCLDQR][NCLDQS] -= zfrz;
          }
        }
      }

      // 4.4c  FREEZING of LIQUID
      // not implicit yet...
      real_t zfrzmax = max((yrecldp::rthomo - ztp1) * zrldcp, real_t(0));

      if (zfrzmax > zepsec && zqxfg[NCLDQL] > zepsec) {
        int jn = imelt(NCLDQL);
        real_t zfrz = min(zqxfg[NCLDQL], zfrzmax);
        zsolqa[jn][NCLDQL] += zfrz;
        zsolqa[NCLDQL][jn] -= zfrz;
      }

      // 4.5   EVAPORATION OF RAIN/SNOW
      // Rain evaporation scheme from Sundquist
      if (ievaprain == 1) {
        real_t zzrh = yrecldp::rprecrhmax +
                      ((real_t(1) - yrecldp::rprecrhmax) * zcovpmax) /
                          max(zepsec, real_t(1) - za);
        zzrh = min(max(zzrh, yrecldp::rprecrhmax), real_t(1));

        real_t zqe = (zqx[NCLDQV] - za * zqsliq) / max(zepsec, real_t(1) - za);
        // humidity in moistest ZCOVPCLR part of domain
        zqe = max(real_t(0), min(zqe, zqsliq));
        if (zcovpclr > zepsec && zqxfg[NCLDQR] > zepsec &&
            zqe < zzrh * zqsliq) {
          // note: zpreclr is a rain flux
          real_t zpreclr = (zqxfg[NCLDQR] * zcovpclr) /
                           copysign(max(abs(zcovptot * zdtgdp), zepsilon),
                                    zcovptot * zdtgdp);

          // actual microphysics formula in zbeta
          real_t zbeta1 =
              ((sqrt(pap_ / paph_top) / yrecldp::rvrfactor) * zpreclr) /
              max(zcovpclr, zepsec);

          real_t zbeta = yomcst::rg * yrecldp::rpecons * real_t(0.5) *
                         pow(zbeta1, real_t(0.5777));

          real_t zdenom = real_t(1) + zbeta * ptsphy * zcorqsliq;
          real_t zdpr =
              ((zcovpclr * zbeta * (zqsliq - zqe)) / zdenom) * zdp * zrg_r;
          real_t zdpevap = zdpr * zdtgdp;

          // add evaporation term to explicit sink.
          // this has to be explicit since if treated in the implicit
          // term evaporation can not reduce rain to zero and model
          // produces small amounts of rainfall everywhere.

          // Evaporate rain
          real_t zevap = min(zdpevap, zqxfg[NCLDQR]);

          zsolqa[NCLDQV][NCLDQR] += zevap;
          zsolqa[NCLDQR][NCLDQV] -= zevap;

          // Reduce the total precip coverage proportional to evaporation
          // to mimic the previous scheme which had a diagnostic
          // 2-flux treatment, abandoned due to the new prognostic precip
          zcovptot = max(yrecldp::rcovpmin,
                         zcovptot - max(real_t(0), ((zcovptot - za) * zevap) /
                                                       zqxfg[NCLDQR]));

          // update fg field
          zqxfg[NCLDQR] -= zevap;
        }

        // Rain evaporation scheme based on Abel and Boutle (2013)
      } else if (ievaprain == 2) {

        // Calculate relative humidity limit for rain evaporation
        // to avoid cloud formation and saturation of the grid box
        // Limit RH for rain evaporation dependent on precipitation fraction
        real_t zzrh = yrecldp::rprecrhmax +
                      ((real_t(1) - yrecldp::rprecrhmax) * zcovpmax) /
                          max(zepsec, real_t(1) - za);
        zzrh = min(max(zzrh, yrecldp::rprecrhmax), real_t(1));

        // Critical relative humidity
        // ZRHC=RAMID
        // ZSIGK=PAP(JL,JK, IBL)/PAPH(JL,KLEV+1, IBL)
        // Increase RHcrit to 1.0 towards the surface (eta>0.8)
        // IF(ZSIGK > 0.8_JPRB) THEN
        //  ZRHC=RAMID+(1.0_JPRB-RAMID)*((ZSIGK-0.8_JPRB)/0.2_JPRB)**2
        // ENDIF
        // ZZRH = MIN(ZRHC,ZZRH)

        // Further limit RH for rain evaporation to 80% (RHcrit in free
        // troposphere)
        zzrh = min(real_t(0.8), zzrh);

        real_t zqe = max(real_t(0), min(zqx[NCLDQV], zqsliq));

        bool llo1 =
            zcovpclr > zepsec && zqxfg[NCLDQR] > zepsec && zqe < zzrh * zqsliq;

        if (llo1) {

          // Abel and Boutle (2012) evaporation
          // Calculate local precipitation (kg/kg)
          real_t zpreclr = zqxfg[NCLDQR] / zcovptot;

          // Fallspeed air density correction
          real_t zfallcorr = pow(yrecldp::rdensref / zrho, real_t(0.4));

          // Saturation vapour pressure with respect to liquid phase
          real_t zesatliq = (yomcst::rv / yomcst::rd) * foeeliq(ztp1);

          // Slope of particle size distribution
          real_t zlambda = pow(yrecldp::rcl_fac1 / ((zrho * zpreclr)),
                               yrecldp::rcl_fac2); // zpreclr=kg/kg

          real_t zevap_denom =
              yrecldp::rcl_cdenom1 * zesatliq -
              yrecldp::rcl_cdenom2 * ztp1 * zesatliq +
              yrecldp::rcl_cdenom3 * pow(ztp1, real_t(3)) * pap_;

          // Temperature dependent conductivity
          real_t zcorr2 = (pow(ztp1 / real_t(273), real_t(1.5)) * real_t(393)) /
                          (ztp1 + real_t(120));

          real_t zsubsat = max(zzrh * zqsliq - zqe, real_t(0));

          real_t zbeta =
              (real_t(0.5) / zqsliq) * pow(ztp1, real_t(2)) * zesatliq *
              yrecldp::rcl_const1r * (zcorr2 / zevap_denom) *
              (real_t(0.78) / pow(zlambda, yrecldp::rcl_const4r) +
               (yrecldp::rcl_const2r * pow(zrho * zfallcorr, real_t(0.5))) /
                   ((pow(zcorr2, real_t(0.5)) *
                     pow(zlambda, yrecldp::rcl_const3r))));

          real_t zdenom = real_t(1) + zbeta * ptsphy; //*zcorqsliq(jl)
          real_t zdpevap = (zcovpclr * zbeta * ptsphy * zsubsat) / zdenom;

          // Add evaporation term to explicit sink.
          // this has to be explicit since if treated in the implicit
          // term evaporation can not reduce rain to zero and model
          // produces small amounts of rainfall everywhere.

          // Limit rain evaporation
          real_t zevap = min(zdpevap, zqxfg[NCLDQR]);

          zsolqa[NCLDQV][NCLDQR] += zevap;
          zsolqa[NCLDQR][NCLDQV] -= zevap;

          // Reduce the total precip coverage proportional to evaporation
          // to mimic the previous scheme which had a diagnostic
          // 2-flux treatment, abandoned due to the new prognostic precip
          zcovptot = max(yrecldp::rcovpmin,
                         zcovptot - max(real_t(0), ((zcovptot - za) * zevap) /
                                                       zqxfg[NCLDQR]));

          // Update fg field
          zqxfg[NCLDQR] -= -zevap;
        }
      }

      // 4.5   EVAPORATION OF SNOW
      // Snow
      if (ievapsnow == 1) {

        real_t zzrh = yrecldp::rprecrhmax +
                      ((real_t(1) - yrecldp::rprecrhmax) * zcovpmax) /
                          max(zepsec, real_t(1) - za);
        zzrh = min(max(zzrh, yrecldp::rprecrhmax), real_t(1));
        real_t zqe = (zqx[NCLDQV] - za * zqsice) / max(zepsec, real_t(1) - za);

        // humidity in moistest ZCOVPCLR part of domain
        zqe = max(real_t(0), min(zqe, zqsice));
        if (zcovpclr > zepsec && zqxfg[NCLDQS] > zepsec &&
            zqe < zzrh * zqsice) {

          // note: zpreclr is a rain flux a
          real_t zpreclr = (zqxfg[NCLDQS] * zcovpclr) /
                           copysign(max(abs(zcovptot * zdtgdp), zepsilon),
                                    zcovptot * zdtgdp);

          // actual microphysics formula in zbeta

          real_t zbeta1 =
              ((sqrt(pap_ / paph_top) / yrecldp::rvrfactor) * zpreclr) /
              max(zcovpclr, zepsec);

          real_t zbeta =
              yomcst::rg * yrecldp::rpecons * pow(zbeta1, real_t(0.5777));

          real_t zdenom = real_t(1) + zbeta * ptsphy * zcorqsice;
          real_t zdpr =
              ((zcovpclr * zbeta * (zqsice - zqe)) / zdenom) * zdp * zrg_r;
          real_t zdpevap = zdpr * zdtgdp;

          // add evaporation term to explicit sink.
          // this has to be explicit since if treated in the implicit
          // term evaporation can not reduce snow to zero and model
          // produces small amounts of snowfall everywhere.

          // Evaporate snow
          real_t zevap = min(zdpevap, zqxfg[NCLDQS]);

          zsolqa[NCLDQV][NCLDQS] += zevap;
          zsolqa[NCLDQS][NCLDQV] -= zevap;

          // Reduce the total precip coverage proportional to evaporation
          // to mimic the previous scheme which had a diagnostic
          // 2-flux treatment, abandoned due to the new prognostic precip
          zcovptot = max(yrecldp::rcovpmin,
                         zcovptot - max(real_t(0), ((zcovptot - za) * zevap) /
                                                       zqxfg[NCLDQS]));

          // Update first guess field
          zqxfg[NCLDQS] -= zevap;
        }
      } else if (ievapsnow == 2) {
        // Calculate relative humidity limit for snow evaporation
        real_t zzrh = yrecldp::rprecrhmax +
                      ((real_t(1) - yrecldp::rprecrhmax) * zcovpmax) /
                          max(zepsec, real_t(1) - za);
        zzrh = min(max(zzrh, yrecldp::rprecrhmax), real_t(1));
        real_t zqe = (zqx[NCLDQV] - za * zqsice) / max(zepsec, real_t(1) - za);

        // humidity in moistest ZCOVPCLR part of domain
        zqe = max(real_t(0), min(zqe, zqsice));
        if (zcovpclr > zepsec && zqx[NCLDQS] > zepsec && zqe < zzrh * zqsice) {

          // Calculate local precipitation (kg/kg)
          real_t zpreclr = zqx[NCLDQS] / zcovptot;
          real_t zvpice = (foeeice(ztp1) * yomcst::rv) / yomcst::rd;

          // Particle size distribution
          // ZTCG increases Ni with colder temperatures - essentially a
          // Fletcher or Meyers scheme?
          real_t ztcg = real_t(
              1); // v1 exp(yrecldp::rcl_x3i*(real_t(273.15)-ztp1)/real_t(8.18))
          // ZFACX1I modification is based on Andrew Barrett's results
          real_t zfacx1s = real_t(1); // v1 (zice0/real_t(1.e-5))**real_t(0.627)

          real_t zaplusb = yrecldp::rcl_apb1 * zvpice -
                           yrecldp::rcl_apb2 * zvpice * ztp1 +
                           pap_ * yrecldp::rcl_apb3 * pow(ztp1, 3);
          real_t zcorrfac = pow(real_t(1) / zrho, real_t(0.5));
          real_t zcorrfac2 = pow(ztp1 / real_t(273.0), real_t(1.5)) *
                             (real_t(393) / (ztp1 + real_t(120)));

          real_t zpr02 =
              (zrho * zpreclr * yrecldp::rcl_const1s) / ((ztcg * zfacx1s));

          real_t zterm1 = ((zqsice - zqe) * sqr(ztp1) * zvpice * zcorrfac2 *
                           ztcg * yrecldp::rcl_const2s * zfacx1s) /
                          ((zrho * zaplusb * zqsice));
          real_t zterm2 =
              real_t(0.65) * yrecldp::rcl_const6s *
                  pow(zpr02, yrecldp::rcl_const4s) +
              (yrecldp::rcl_const3s * pow(zcorrfac, real_t(0.5)) *
               pow(zrho, real_t(0.5)) * pow(zpr02, yrecldp::rcl_const5s)) /
                  pow(zcorrfac2, real_t(0.5));

          real_t zdpevap = max(zcovpclr * zterm1 * zterm2 * ptsphy, real_t(0));

          // Limit evaporation to snow amount
          real_t zevap = min(zdpevap, zevaplimice);
          zevap = min(zevap, zqx[NCLDQS]);

          zsolqa[NCLDQV][NCLDQS] += zevap;
          zsolqa[NCLDQS][NCLDQV] -= zevap;

          // Reduce the total precip coverage proportional to evaporation
          // to mimic the previous scheme which had a diagnostic
          // 2-flux treatment, abandoned due to the new prognostic precip
          zcovptot = max(yrecldp::rcovpmin,
                         zcovptot - max(real_t(0), ((zcovptot - za) * zevap) /
                                                       zqx[NCLDQS]));

          // Update first guess field
          zqxfg[NCLDQS] = zqxfg[NCLDQS] - zevap;
        }
      }

      //--------------------------------------
      // Evaporate small precipitation amounts
      //--------------------------------------
#pragma unroll
      for (int jm = 0; jm < NCLV - 1; ++jm) {
        if (llfall(static_cast<type>(jm))) {
          if (zqxfg[jm] < yrecldp::rlmin) {
            zsolqa[NCLDQV][jm] += zqxfg[jm];
            zsolqa[jm][NCLDQV] -= zqxfg[jm];
          }
        }
      }

      /////
      //            5.0  *** SOLVERS FOR A AND L ***
      // now use an implicit solution rather than exact solution

      // 5.1 solver for cloud cover
      real_t zanew = (za + zsolac) / (real_t(1) + zsolab);
      zanew = min(zanew, real_t(1));
      if (zanew < yrecldp::ramin)
        zanew = real_t(0);
      real_t zda = zanew - zaorig;
      // variables needed for next level
      zanewm1 = zanew;

      // 5.2 solver for the microphysics

      // Truncate explicit sinks to avoid negatives
      // Note: Species are treated in the order in which they run out
      // since the clipping will alter the balance for the other vars

      // scale the sink terms, in the correct order,
      // recalculating the scale factor each time
#pragma unroll
      for (int jm = 0; jm < NCLV; ++jm) {

        // recalculate sum
        real_t psum_solqa = real_t(0);
#pragma unroll
        for (int jn = 0; jn < NCLV; ++jn) {
          psum_solqa = psum_solqa + zsolqa[jm][jn];
        }
        real_t zsinksum = -psum_solqa;
        // recalculate scaling factor
        real_t zmm = max(zqx[jm], zepsec);
        real_t zrr = max(zsinksum, zmm);
        real_t zzratio = zmm / zrr;
        // scale
#pragma unroll
        for (int jn = 0; jn < NCLV; ++jn) {
          if (zsolqa[jm][jn] < real_t(0)) {
            zsolqa[jm][jn] *= zzratio;
            zsolqa[jn][jm] *= zzratio;
          }
        }
      }

      // 5.2.2 Solver

      // set the LHS of equation
      real_t zqlhs[NCLV][NCLV];
#pragma unroll
      for (int jm = 0; jm < NCLV; ++jm) {
#pragma unroll
        for (int jn = 0; jn < NCLV; ++jn) {
          // diagonals: microphysical sink terms+transport
          if (jm == jn) {
            zqlhs[jn][jm] = real_t(1) + zfallsink[jm];
            for (int jo = 0; jo < NCLV; ++jo) {
              zqlhs[jn][jm] = zqlhs[jn][jm] + zsolqb[jo][jn];
            }
            // non-diagonals: microphysical source terms
          } else {
            zqlhs[jn][jm] =
                -zsolqb[jn][jm]; // here is the delta t - missing from doc.
          }
        }
      }

      real_t zqxn[NCLV];
      // set the RHS of equation
#pragma unroll
      for (int jm = 0; jm < NCLV; ++jm) {
        // sum the explicit source and sink
        real_t zexplicit = real_t(0);
#pragma unroll
        for (int jn = 0; jn < NCLV; ++jn) {
          zexplicit += zsolqa[jm][jn]; // sum over middle index
        }
        zqxn[jm] = zqx[jm] + zexplicit;
      }

      // *** solve by LU decomposition: ***

      // Note: This fast way of solving NCLVxNCLV system
      //       assumes a good behaviour (i.e. non-zero diagonal
      //       terms with comparable orders) of the matrix stored
      //       in ZQLHS. For the moment this is the case but
      //       be aware to preserve it when doing eventual
      //       modifications.

      // Non pivoting recursive factorization
#pragma unroll
      for (int jn = 0; jn < NCLV - 1; ++jn) {
        // number of steps
#pragma unroll
        for (int jm = jn + 1; jm < NCLV; ++jm) {
          // row index
          zqlhs[jm][jn] = zqlhs[jm][jn] / zqlhs[jn][jn];
#pragma unroll
          for (int ik = jn + 1; ik < NCLV; ++ik) {
            // column index
            zqlhs[jm][ik] = zqlhs[jm][ik] - zqlhs[jm][jn] * zqlhs[jn][ik];
          }
        }
      }

      // Backsubstitution
      //  step 1
#pragma unroll
      for (int jn = 1; jn < NCLV; ++jn) {
#pragma unroll
        for (int jm = 0; jm < jn - 1; ++jm) {
          zqxn[jn] = zqxn[jn] - zqlhs[jn][jm] * zqxn[jm];
        }
      }
      //  step 2
      zqxn[NCLV - 1] = zqxn[NCLV - 1] / zqlhs[NCLV - 1][NCLV - 1];
#pragma unroll
      for (int jn = NCLV - 2; jn >= 0; --jn) {
#pragma unroll
        for (int jm = jn + 1; jm < NCLV; ++jm) {
          zqxn[jn] = zqxn[jn] - zqlhs[jn][jm] * zqxn[jm];
        }
        zqxn[jn] = zqxn[jn] / zqlhs[jn][jn];
      }

      // Ensure no small values (including negatives) remain in cloud
      // variables nor precipitation rates. Evaporate l,i,r,s to water vapour.
      // Latent heating taken into account below
#pragma unroll
      for (int jn = 0; jn < NCLV - 1; ++jn) {
        if (zqxn[jn] < zepsec) {
          zqxn[NCLDQV] = zqxn[NCLDQV] + zqxn[jn];
          zqxn[jn] = real_t(0);
        }
      }

      // variables needed for next level
#pragma unroll
      for (int jm = 0; jm < NCLV - 1; ++jm) {
        zqxnm1[jm] = zqxn[jm];
      }
#pragma unroll
      for (int jm = 0; jm < NCLV - 1; ++jm) {
        zqxn2d[jm] = zqxn[jm];
      }

      // 5.3 Precipitation/sedimentation fluxes to next level
      //     diagnostic precipitation fluxes
      //     It is this scaled flux that must be used for source to next layer

#pragma unroll
      for (int jm = 0; jm < NCLV - 1; ++jm) {
        zpfplsx[jm] = zfallsink[jm] * zqxn[jm] * zrdtgdp;
      }

      // Ensure precipitation fraction is zero if no precipitation
      zqpretot = zpfplsx[NCLDQS] + zpfplsx[NCLDQR];
      if (zqpretot < zepsec)
        zcovptot = real_t(0);

        //////
        //              6  *** UPDATE TENDANCIES ***

        // 6.1 Temperature and CLV budgets

#pragma unroll
      for (int jm = 0; jm < NCLV - 1; ++jm) {

        // calculate fluxes in and out of box for conservation of TL
        real_t zfluxq = zpsupsatsrce[jm] + zconvsrce[jm] + zfallsrce[jm] -
                        (zfallsink[jm] + zconvsink[jm]) * zqxn[jm];

        if (iphase(static_cast<type>(jm)) == LIQUID)
          tendency_loc_t_ +=
              yoethf::ralvdcp * (zqxn[jm] - zqx[jm] - zfluxq) * zqtmst;

        else if (iphase(static_cast<type>(jm)) == ICE)
          tendency_loc_t_ +=
              yoethf::ralsdcp * (zqxn[jm] - zqx[jm] - zfluxq) * zqtmst;

        // New prognostic tendencies - ice,liquid rain,snow
        // Note: CLV arrays use PCLV in calculation of tendency while humidity
        //       uses ZQX. This is due to clipping at start of cloudsc which
        //       include the tendency already in TENDENCY_LOC_T and
        //       TENDENCY_LOC_q. ZQX was reset
        tendency_loc[get_index(descriptor_tendency, nproma, jk, POS_CLD + jm)] =
            (zqxn[jm] - zqx0[jm]) * zqtmst;
      }
      tendency_loc[get_index(descriptor_tendency, nproma, jk,
                             POS_CLD + NCLV - 1)] = real_t(0);

      // 6.2 Humidity budget
      tendency_loc_q_ += (zqxn[NCLDQV] - zqx[NCLDQV]) * zqtmst;

      // 6.3 cloud cover
      tendency_loc_a_ += zda * zqtmst;

      // j Copy precipitation fraction into output variable
      pcovptot[get_index(descriptor_k, nproma, jk)] = zcovptot;

    } else {
      pcovptot[get_index(descriptor_k, nproma, jk)] = real_t(0);

#pragma unroll
      for (int jm = 0; jm < NCLV; ++jm)
        tendency_loc[get_index(descriptor_tendency, nproma, jk, POS_CLD + jm)] =
            real_t(0);
#pragma unroll
      for (int jm = 0; jm < NCLV - 1; ++jm)
        zqxn2d[jm] = real_t(0);
#pragma unroll
      for (int jm = 0; jm < NCLV - 1; ++jm)
        zpfplsx[jm] = real_t(0); // precip fluxes at next level
    }
    tendency_loc[get_index(descriptor_tendency, nproma, jk, POS_T)] =
        tendency_loc_t_;
    tendency_loc[get_index(descriptor_tendency, nproma, jk, POS_A)] =
        tendency_loc_a_;
    tendency_loc[get_index(descriptor_tendency, nproma, jk, POS_Q)] =
        tendency_loc_q_;

    //                      END OF VERTICAL LOOP

    //////////
    //             8  *** FLUX/DIAGNOSTICS COMPUTATIONS ***

    real_t zgdph_r = -zrg_r * (paph_next - paph_) * zqtmst;
    real_t pfsqlf_ = at_shared<nproma>(SHARED::PFSQLF);
    real_t pfsqrf_ = pfsqlf_;
    real_t pfsqif_ = at_shared<nproma>(SHARED::PFSQIF);
    real_t pfsqsf_ = pfsqif_;
    real_t pfcqlng_ = at_shared<nproma>(SHARED::PFCQLNG);
    real_t pfcqnng_ = at_shared<nproma>(SHARED::PFCQNNG);
    real_t pfcqrng_ = pfcqlng_;
    real_t pfcqsng_ = pfcqnng_;
    real_t pfsqltur_ = at_shared<nproma>(SHARED::PFSQLTUR);
    real_t pfsqitur_ = at_shared<nproma>(SHARED::PFSQITUR);

    real_t zalfaw = zfoealfa;

    // Liquid , LS scheme minus detrainment
    real_t pvfl_ = at_shared<nproma>(SHARED::PVFL, jk);
    pfsqlf[get_index(descriptor_k1, nproma, jk + 1)] =
        at_shared<nproma>(PFSQLF) =
            pfsqlf_ +
            (zqxn2d[NCLDQL] - zqx0[NCLDQL] + pvfl_ * ptsphy - zalfaw * plude_) *
                zgdph_r;
    // liquid, negative numbers
    pfcqlng[get_index(descriptor_k1, nproma, jk + 1)] =
        at_shared<nproma>(PFCQLNG) = pfcqlng_ + zlneg[NCLDQL] * zgdph_r;

    // liquid, vertical diffusion
    pfsqltur[get_index(descriptor_k1, nproma, jk + 1)] =
        at_shared<nproma>(PFSQLTUR) = pfsqltur_ + pvfl_ * ptsphy * zgdph_r;

    // Rain, LS scheme
    pfsqrf[get_index(descriptor_k1, nproma, jk + 1)] =
        pfsqrf_ + (zqxn2d[NCLDQR] - zqx0[NCLDQR]) * zgdph_r;
    // , IBLrain, negative numbers
    pfcqrng[get_index(descriptor_k1, nproma, jk + 1)] =
        pfcqrng_ + zlneg[NCLDQR] * zgdph_r;

    // Ice , LS scheme minus detrainment
    real_t pvfi_ = at_shared<nproma>(SHARED::PVFI, jk);
    pfsqif[get_index(descriptor_k1, nproma, jk + 1)] =
        at_shared<nproma>(PFSQIF) =
            pfsqif_ + (zqxn2d[NCLDQI] - zqx0[NCLDQI] + pvfi_ * ptsphy -
                       (real_t(1) - zalfaw) * plude_) *
                          zgdph_r;
    // ice, negative numbers
    pfcqnng[get_index(descriptor_k1, nproma, jk + 1)] =
        at_shared<nproma>(PFCQNNG) = pfcqnng_ + zlneg[NCLDQI] * zgdph_r;

    // ice, vertical diffusion
    pfsqitur[get_index(descriptor_k1, nproma, jk + 1)] =
        at_shared<nproma>(PFSQITUR) = pfsqitur_ + pvfi_ * ptsphy * zgdph_r;

    // snow, LS scheme
    pfsqsf[get_index(descriptor_k1, nproma, jk + 1)] =
        pfsqsf_ + (zqxn2d[NCLDQS] - zqx0[NCLDQS]) * zgdph_r;
    // snow, negative numbers
    pfcqsng[get_index(descriptor_k1, nproma, jk + 1)] =
        pfcqsng_ + zlneg[NCLDQS] * zgdph_r;

    // Copy general precip arrays back into PFP arrays for GRIB archiving
    // Add rain and liquid fluxes, ice and snow fluxes
    real_t pfplsl_ = zpfplsx[NCLDQR] + zpfplsx[NCLDQL];
    pfplsl[get_index(descriptor_k1, nproma, jk + 1)] = pfplsl_;
    real_t pfplsn_ = zpfplsx[NCLDQS] + zpfplsx[NCLDQI];
    pfplsn[get_index(descriptor_k1, nproma, jk + 1)] = pfplsn_;

    // enthalpy flux due to precipitation
    pfhpsl[get_index(descriptor_k1, nproma, jk + 1)] = -yomcst::rlvtt * pfplsl_;
    pfhpsn[get_index(descriptor_k1, nproma, jk + 1)] = -yomcst::rlstt * pfplsn_;

    ztp1_prev = ztp1;
    za_prev = za;
    pap_prev = pap_;
    paph_ = paph_next;

    plude[get_index(descriptor_k, nproma, jk)] = plude_;
    pipeline.consumer_release();

    block.sync();
  }

  prainfrac_toprfz[get_index(descriptor_2d, nproma, 0)] = prainfrac_toprfz_;
}

void run(int klev, int ngptot, int nproma, py::dict c, py::dict f, py::dict s,
         py::dict o, int config, bool perftest) {
  Buffer plcrit_aer{f["plcrit_aer"]};
  Buffer picrit_aer{f["picrit_aer"]};
  Buffer pre_ice{f["pre_ice"]};
  Buffer pccn{f["pccn"]};
  Buffer pnice{f["pnice"]};
  Buffer pt{f["pt"]};
  Buffer pq{f["pq"]};
  /* Buffer pvfa{f["pvfa"]}; */
  Buffer pvfl{f["pvfl"]};
  Buffer pvfi{f["pvfi"]};
  /* Buffer pdyna{f["pdyna"]}; */
  /* Buffer pdynl{f["pdynl"]}; */
  /* Buffer pdyni{f["pdyni"]}; */
  Buffer phrsw{f["phrsw"]};
  Buffer phrlw{f["phrlw"]};
  Buffer pvervel{f["pvervel"]};
  Buffer pap{f["pap"]};
  Buffer paph{f["paph"]};
  Buffer plsm{f["plsm"]};
  Buffer ldcum{f["ldcum"]};
  Buffer ktype{f["ktype"]};
  Buffer plu{f["plu"]};
  Buffer plude{f["plude"]};
  Buffer psnde{f["psnde"]};
  Buffer pmfu{f["pmfu"]};
  Buffer pmfd{f["pmfd"]};
  Buffer pa{f["pa"]};
  Buffer pclv{f["pclv"]};
  Buffer psupsat{f["psupsat"]};
  /* Buffer pextra{f["pextra"]}; */
  Buffer tendency_tmp{f["tendency_tmp"]};
  Buffer tendency_loc{f["tendency_loc"]};

  real_t ptsphy = py::float_(s["ptsphy"]);
  bool ldslphy = py::bool_(s["ldslphy"]);
  /* bool ldmaincall = py::bool_(s["ldmaincall"]); */

  Buffer prainfrac_toprfz{o["prainfrac_toprfz"]};
  Buffer pcovptot{o["pcovptot"]};
  Buffer pfsqlf{o["pfsqlf"]};
  Buffer pfsqif{o["pfsqif"]};
  Buffer pfcqlng{o["pfcqlng"]};
  Buffer pfcqnng{o["pfcqnng"]};
  Buffer pfsqrf{o["pfsqrf"]};
  Buffer pfsqsf{o["pfsqsf"]};
  Buffer pfcqrng{o["pfcqrng"]};
  Buffer pfcqsng{o["pfcqsng"]};
  Buffer pfsqltur{o["pfsqltur"]};
  Buffer pfsqitur{o["pfsqitur"]};
  Buffer pfplsl{o["pfplsl"]};
  Buffer pfplsn{o["pfplsn"]};
  Buffer pfhpsl{o["pfhpsl"]};
  Buffer pfhpsn{o["pfhpsn"]};
  Buffer tmp1{o["tmp1"]};
  Buffer tmp2{o["tmp2"]};

  int nthreads = nproma;
  int nblocks = (ngptot + nthreads - 1) / nthreads;

  decltype(&run_cloudsc<1, 1, 1, 1, 1>) fptr = nullptr;

  if (config == 1)
    switch (nproma) {
    case 64:
      fptr = run_cloudsc<2, 2, 1, 1, 64>;
      break;
    case 128:
      fptr = run_cloudsc<2, 2, 1, 1, 128>;
      break;
    case 256:
      fptr = run_cloudsc<2, 2, 1, 1, 256>;
      break;
    }
  /*
  else if (config == 2)
    fptr = run_cloudsc<1, 2, 1, 1>;
  else if (config == 3)
    fptr = run_cloudsc<2, 1, 1, 1>;
  else if (config == 4)
    fptr = run_cloudsc<2, 2, 2, 1>;
  else if (config == 5)
    fptr = run_cloudsc<2, 2, 1, 2>;
    */
  else {
    std::cerr << "Unsupported function configuration";
    std::abort();
  }

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  int nreps = perftest ? 30 : 1;

  int smem_size = SHARED::TOTAL * nproma * sizeof(real_t);

  // CUDA_CHECK(cudaFuncSetCacheConfig(fptr, cudaFuncCachePreferShared));
  CUDA_CHECK(cudaFuncSetAttribute(
      fptr, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

  for (int i = 0; i < nreps; ++i) {
    if (i == 10)
      CUDA_CHECK(cudaEventRecord(start));
    fptr<<<nblocks, nthreads, smem_size>>>(
        klev, ngptot, nproma, plcrit_aer.get(), picrit_aer.get(), pre_ice.get(),
        pccn.get(), pnice.get(), pt.get(), pq.get(), pvfl.get(), pvfi.get(),
        phrsw.get(), phrlw.get(), pvervel.get(), pap.get(), paph.get(),
        plsm.get(), ldcum.get(), ktype.get(), plu.get(), plude.get(),
        psnde.get(), pmfu.get(), pmfd.get(), pa.get(), pclv.get(),
        psupsat.get(), tendency_tmp.get(), tendency_loc.get(),
        prainfrac_toprfz.get(), pcovptot.get(), pfsqlf.get(), pfsqif.get(),
        pfcqlng.get(), pfcqnng.get(), pfsqrf.get(), pfsqsf.get(), pfcqrng.get(),
        pfcqsng.get(), pfsqltur.get(), pfsqitur.get(), pfplsl.get(),
        pfplsn.get(), pfhpsl.get(), pfhpsn.get(), tmp1.get(), tmp2.get(),
        ptsphy, ldslphy);
    CUDA_CHECK(cudaGetLastError());
  }
  if (nreps > 1)
    CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaDeviceSynchronize());

  if (perftest) {
    float time;
    CUDA_CHECK(cudaEventElapsedTime(&time, start, stop));
    time = time / (nreps - 10) / 1000;
    std::cout << "Elapsed time per run [ms]: " << time * 1000 << '\n';
    std::cout << "Performance [MP/s]:        " << ngptot * klev / time / 1e6
              << '\n';
  }

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
}

PYBIND11_MODULE(cloudsc_cuda, m) { m.def("run", &run, "Run the cloud scheme"); }
