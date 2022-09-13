import numpy as np
import math


def run_cloud_scheme(
    klev,
    ngptot,
    nproma,
    c,
    f,
    s,
    o,
    dtype,
    iwarmrain=2,
    ievaprain=2,
    ievapsnow=1,
    idepice=1,
):

    ## cloud scheme

    NCLDQL = 0  # liquid cloud water
    NCLDQI = 1  # ice cloud water
    NCLDQR = 2  # rain water
    NCLDQS = 3  # snow
    NCLDQV = 4  # vapour
    NCLV = NCLDQV + 1

    # -----------------------------------------------
    # Define species phase, 0=vapour, 1=liquid, 2=ice
    # -----------------------------------------------
    VAPOUR = 0
    LIQUID = 1
    ICE = 2

    def iphase(i):
        if i == NCLDQV:
            return VAPOUR
        elif i == NCLDQL or i == NCLDQR:
            return LIQUID
        elif i == NCLDQI or i == NCLDQS:
            return ICE

    # ---------------------------------------------------
    # Set up melting/freezing index,
    # if an ice category melts/freezes, where does it go?
    # ---------------------------------------------------
    def imelt(i):
        if i == NCLDQV:
            return -99
        elif i == NCLDQL:
            return NCLDQI
        elif i == NCLDQR:
            return NCLDQS
        elif i == NCLDQI:
            return NCLDQR
        elif i == NCLDQS:
            return NCLDQR

    #  -------------------------
    #  set up fall speeds in m/s
    #  -------------------------
    def zvqx(i):
        if i == NCLDQV:
            return 0
        elif i == NCLDQL:
            return 0
        elif i == NCLDQI:
            return c.yrecldp.rvice
        elif i == NCLDQR:
            return c.yrecldp.rvrain
        elif i == NCLDQS:
            return c.yrecldp.rvsnow

    def llfall(i):
        # Set LLFALL to false for ice (but ice still sediments!)
        # Need to rationalise this at some point
        if i == NCLDQI:
            return False
        else:
            return zvqx(i) > 0

    # foedelta = 1    water
    # foedelta = 0    ice
    def foedelta(ptare):
        if ptare - c.yomcst.rtt >= 0:
            return 1
        else:
            return 0

    # THERMODYNAMICAL FUNCTIONS .

    # Pressure of water vapour at saturation
    # PTARE = TEMPERATURE
    def foeew(ptare):
        return c.yoethf.r2es * math.exp(
            (c.yoethf.r3les * foedelta(ptare) + c.yoethf.r3ies * (1 - foedelta(ptare)))
            * (ptare - c.yomcst.rtt)
            / (
                ptare
                - (
                    c.yoethf.r4les * foedelta(ptare)
                    + c.yoethf.r4ies * (1 - foedelta(ptare))
                )
            )
        )

    # CONSIDERATION OF MIXED PHASES
    # FOEALFA is calculated to distinguish the three cases:
    # FOEALFA=1            water phase
    # FOEALFA=0            ice phase
    # 0 < FOEALFA < 1      mixed phase
    def foealfa(ptare):
        return min(
            1,
            (
                (max(c.yoethf.rtice, min(c.yoethf.rtwat, ptare)) - c.yoethf.rtice)
                * c.yoethf.rtwat_rtice_r
            )
            ** 2,
        )

    # Pressure of water vapour at saturation
    def foeewm(ptare):
        return c.yoethf.r2es * (
            foealfa(ptare)
            * math.exp(
                c.yoethf.r3les * (ptare - c.yomcst.rtt) / (ptare - c.yoethf.r4les)
            )
            + (1 - foealfa(ptare))
            * math.exp(
                c.yoethf.r3ies * (ptare - c.yomcst.rtt) / (ptare - c.yoethf.r4ies)
            )
        )

    def foedem(ptare):
        return foealfa(ptare) * c.yoethf.r5alvcp * (
            1 / (ptare - c.yoethf.r4les) ** 2
        ) + (1 - foealfa(ptare)) * c.yoethf.r5alscp * (
            1 / (ptare - c.yoethf.r4ies) ** 2
        )

    def foeldcpm(ptare):
        return (
            foealfa(ptare) * c.yoethf.ralvdcp + (1 - foealfa(ptare)) * c.yoethf.ralsdcp
        )

    # Pressure of water vapour at saturation
    # This one is for the WMO definition of saturation, i.e. always
    # with respect to water.
    #
    # Duplicate to FOEELIQ and FOEEICE for separate ice variable
    # FOEELIQ always respect to water
    # FOEEICE always respect to ice
    # (could use FOEEW and FOEEWMO, but naming convention unclear)

    def foeeliq(ptare):
        return c.yoethf.r2es * math.exp(
            c.yoethf.r3les * (ptare - c.yomcst.rtt) / (ptare - c.yoethf.r4les)
        )

    def foeeice(ptare):
        return c.yoethf.r2es * math.exp(
            c.yoethf.r3ies * (ptare - c.yomcst.rtt) / (ptare - c.yoethf.r4ies)
        )

    def fokoop(ptare):
        return min(
            c.yoethf.rkoop1 - c.yoethf.rkoop2 * ptare, foeeliq(ptare) / foeeice(ptare)
        )

    for point in range(ngptot):
        jl = point % nproma
        block = point // nproma

        #  ---------------------------------------------------------------------
        # 0.0     Beginning of timestep book-keeping
        #  ---------------------------------------------------------------------

        # setup constants
        zepsilon = 100 * np.finfo(dtype).eps

        # ---------------------
        # Some simple constants
        # ---------------------
        zqtmst = 1 / s.ptsphy
        zgdcp = c.yomcst.rg / c.yomcst.rcpd
        zrdcp = c.yomcst.rd / c.yomcst.rcpd
        zcons1a = c.yomcst.rcpd / ((c.yomcst.rlmlt * c.yomcst.rg * c.yrecldp.rtaumel))
        zepsec = 1.0e-14
        zrg_r = 1 / c.yomcst.rg
        zrldcp = 1 / (c.yoethf.ralsdcp - c.yoethf.ralvdcp)

        #  ! -----------------------------------------------
        #  ! INITIALIZATION OF OUTPUT TENDENCIES
        #  ! -----------------------------------------------
        for jk in range(klev):
            f.tendency_loc_t[block, jk, jl] = 0
            f.tendency_loc_q[block, jk, jl] = 0
            f.tendency_loc_a[block, jk, jl] = 0

            for jm in range(NCLV - 1):
                f.tendency_loc_cld[block, jm, jk, jl] = 0

            #  These were uninitialized : meaningful only when we compare error differences
            o.pcovptot[block, jk, jl] = 0
            f.tendency_loc_cld[block, NCLV - 1, jk, jl] = 0

        #  ######################################################################
        #               1.  *** INITIAL VALUES FOR VARIABLES ***
        #  ######################################################################

        ztp1 = np.empty(klev, dtype=dtype)
        zqx = np.empty((5, klev), dtype=dtype)
        za = np.empty(klev, dtype=dtype)
        zaorig = np.empty(klev, dtype=dtype)
        zqx0 = np.empty((5, klev), dtype=dtype)
        zpfplsx = np.empty((5, klev + 1), dtype=dtype)
        zqxn2d = np.empty((5, klev), dtype=dtype)
        zlneg = np.empty((5, klev), dtype=dtype)

        #  ----------------------
        #  non CLV initialization
        #   ----------------------
        for jk in range(klev):
            ztp1[jk] = f.pt[block, jk, jl] + s.ptsphy * f.tendency_tmp_t[block, jk, jl]
            zqx[NCLDQV, jk] = (
                f.pq[block, jk, jl] + s.ptsphy * f.tendency_tmp_q[block, jk, jl]
            )
            za[jk] = f.pa[block, jk, jl] + s.ptsphy * f.tendency_tmp_a[block, jk, jl]

            zqx0[NCLDQV, jk] = zqx[NCLDQV, jk]
            zaorig[jk] = za[jk]

        #  -------------------------------------
        #  initialization for CLV family
        #  -------------------------------------
        for jm in range(NCLV - 1):
            for jk in range(klev):
                zqx[jm, jk] = (
                    f.pclv[block, jm, jk, jl]
                    + s.ptsphy * f.tendency_tmp_cld[block, jm, jk, jl]
                )
                zqx0[jm, jk] = zqx[jm, jk]

        #  -------------
        #   zero arrays
        #  -------------
        zpfplsx[:, :] = 0
        zqxn2d[:, :] = 0
        zlneg[:, :] = 0

        o.prainfrac_toprfz[block, jl] = 0  # rain fraction at top of refreezing layer
        llrainliq = True  # Assume all raindrops are liquid initially

        #  ----------------------------------------------------
        #  Tidy up very small cloud cover or total cloud water
        #  ----------------------------------------------------
        for jk in range(klev):
            if (
                zqx[NCLDQL, jk] + zqx[NCLDQI, jk] < c.yrecldp.rlmin
                or za[jk] < c.yrecldp.ramin
            ):
                # Evaporate small cloud liquid water amounts
                zlneg[NCLDQL, jk] = zlneg[NCLDQL, jk] + zqx[NCLDQL, jk]
                zqadj = zqx[NCLDQL, jk] * zqtmst
                f.tendency_loc_q[block, jk, jl] = (
                    f.tendency_loc_q[block, jk, jl] + zqadj
                )
                f.tendency_loc_t[block, jk, jl] = (
                    f.tendency_loc_t[block, jk, jl] - c.yoethf.ralvdcp * zqadj
                )
                zqx[NCLDQV, jk] = zqx[NCLDQV, jk] + zqx[NCLDQL, jk]
                zqx[NCLDQL, jk] = 0

                # Evaporate small cloud ice water amounts
                zlneg[NCLDQI, jk] = zlneg[NCLDQI, jk] + zqx[NCLDQI, jk]
                zqadj = zqx[NCLDQI, jk] * zqtmst
                f.tendency_loc_q[block, jk, jl] = (
                    f.tendency_loc_q[block, jk, jl] + zqadj
                )
                f.tendency_loc_t[block, jk, jl] = (
                    f.tendency_loc_t[block, jk, jl] - c.yoethf.ralsdcp * zqadj
                )
                zqx[NCLDQV, jk] = zqx[NCLDQV, jk] + zqx[NCLDQI, jk]
                zqx[NCLDQI, jk] = 0

                # Set cloud cover to zero
                za[jk] = 0

        #  ! ---------------------------------
        #  ! Tidy up small CLV variables
        #  ! ---------------------------------
        for jm in range(NCLV - 1):
            for jk in range(klev):
                if zqx[jm, jk] < c.yrecldp.rlmin:
                    zlneg[jm, jk] = zlneg[jm, jk] + zqx[jm, jk]
                    zqadj = zqx[jm, jk] * zqtmst
                    f.tendency_loc_q[block, jk, jl] = (
                        f.tendency_loc_q[block, jk, jl] + zqadj
                    )
                    if iphase(jm) == LIQUID:
                        f.tendency_loc_t[block, jk, jl] = (
                            f.tendency_loc_t[block, jk, jl] - c.yoethf.ralvdcp * zqadj
                        )
                    elif iphase(jm) == ICE:
                        f.tendency_loc_t[block, jk, jl] = (
                            f.tendency_loc_t[block, jk, jl] - c.yoethf.ralsdcp * zqadj
                        )
                    zqx[NCLDQV, jk] = zqx[NCLDQV, jk] + zqx[jm, jk]
                    zqx[jm, jk] = 0

        zfoeewmt = np.empty(klev, dtype=dtype)
        zfoeew = np.empty(klev, dtype=dtype)
        zfoeeliqt = np.empty(klev, dtype=dtype)
        zfoealfa = np.empty(klev + 1, dtype=dtype)
        zqsmix = np.empty(klev, dtype=dtype)
        zqsliq = np.empty(klev, dtype=dtype)
        zqsice = np.empty(klev, dtype=dtype)

        #  ------------------------------
        #  Define saturation values
        #  ------------------------------
        for jk in range(klev):
            # ----------------------------------------
            #  old *diagnostic* mixed phase saturation
            # ----------------------------------------
            zfoealfa[jk] = foealfa(ztp1[jk])
            zfoeewmt[jk] = min(foeewm(ztp1[jk]) / f.pap[block, jk, jl], 0.5)
            zqsmix[jk] = zfoeewmt[jk]
            zqsmix[jk] = zqsmix[jk] / (1 - c.yomcst.retv * zqsmix[jk])

            # ---------------------------------------------
            #  ice saturation T<273K
            #  liquid water saturation for T>273K
            # ---------------------------------------------
            zalfa = foedelta(ztp1[jk])
            zfoeew[jk] = min(
                (zalfa * foeeliq(ztp1[jk]) + (1 - zalfa) * foeeice(ztp1[jk]))
                / f.pap[block, jk, jl],
                0.5,
            )
            zfoeew[jk] = min(0.5, zfoeew[jk])
            zqsice[jk] = zfoeew[jk] / (1 - c.yomcst.retv * zfoeew[jk])

            # ----------------------------------
            #  liquid water saturation
            # ----------------------------------
            zfoeeliqt[jk] = min(foeeliq(ztp1[jk]) / f.pap[block, jk, jl], 0.5)
            zqsliq[jk] = zfoeeliqt[jk]
            zqsliq[jk] = zqsliq[jk] / (1 - c.yomcst.retv * zqsliq[jk])

            # ----------------------------------
            # DISABLED: ice water saturation
            # ----------------------------------
            # ZFOEEICET(JL,JK)=MIN(FOEEICE(ZTP1(JL,JK))/PAP(JL,JK),0.5_JPRB)
            # ZQSICE(JL,JK)=ZFOEEICET(JL,JK)
            # ZQSICE(JL,JK)=ZQSICE(JL,JK)/(1.0_JPRB-RETV*ZQSICE(JL,JK))

        zli = np.empty(klev, dtype=dtype)
        zicefrac = np.empty(klev, dtype=dtype)
        zliqfrac = np.empty(klev, dtype=dtype)
        for jk in range(klev):

            # ------------------------------------------
            #  Ensure cloud fraction is between 0 and 1
            # ------------------------------------------
            za[jk] = max(0, min(1, za[jk]))

            # -------------------------------------------------------------------
            #  Calculate liq/ice fractions (no longer a diagnostic relationship)
            # -------------------------------------------------------------------
            zli[jk] = zqx[NCLDQL, jk] + zqx[NCLDQI, jk]
            if zli[jk] > c.yrecldp.rlmin:
                zliqfrac[jk] = zqx[NCLDQL, jk] / zli[jk]
                zicefrac[jk] = 1 - zliqfrac[jk]
            else:
                zliqfrac[jk] = 0
                zicefrac[jk] = 0

        # ######################################################################
        #         2.       *** CONSTANTS AND PARAMETERS ***
        # ######################################################################
        #   Calculate L in updrafts of bl-clouds
        #   Specify QS, P/PS for tropopause (for c2)
        #   And initialize variables
        # ------------------------------------------

        # ---------------------------------
        #  Find tropopause level (ZTRPAUS)
        # ---------------------------------
        ztrpaus = 0.1
        zpaphd = 1 / f.paph[block, klev, jl]
        for jk in range(klev - 1):
            zsig = f.pap[block, jk, jl] * zpaphd
            if zsig > 0.1 and zsig < 0.4 and ztp1[jk] > ztp1[jk + 1]:
                ztrpaus = zsig

        # -----------------------------
        #  Reset single level variables
        # -----------------------------

        zanewm1 = 0
        zda = 0
        zcovpclr = 0
        zcovpmax = 0
        zcovptot = 0
        zcldtopdist = 0
        # TODO WARN THIS IS GOING TO BE WRONG
        zqxnm1 = np.ones(NCLV, dtype=dtype)

        # ######################################################################
        #            3.       *** PHYSICS ***
        # ######################################################################

        # ----------------------------------------------------------------------
        #                        START OF VERTICAL LOOP
        # ----------------------------------------------------------------------

        for jk in range(c.yrecldp.ncldtop - 1, klev):

            # ----------------------------------------------------------------------
            #  3.0 INITIALIZE VARIABLES
            # ----------------------------------------------------------------------

            # ---------------------------------
            #  First guess microphysics
            # ---------------------------------
            zqxfg = np.empty(NCLV, dtype=dtype)
            for jm in range(NCLV):
                zqxfg[jm] = zqx[jm, jk]

            # ---------------------------------
            #  Set KLON arrays to zero
            # ---------------------------------

            zlicld = 0
            zrainaut = 0  # currently needed for diags
            zrainacc = 0  # currently needed for diags
            zsnowaut = 0  # needed
            zldefr = 0
            zacust = 0  # set later when needed
            zqpretot = 0
            zlfinalsum = 0

            # required for first guess call
            zlcond1 = 0
            zlcond2 = 0
            zsupsat = 0
            zlevapl = 0
            zlevapi = 0

            # -------------------------------------
            #  solvers for cloud fraction
            # -------------------------------------
            zsolab = 0
            zsolac = 0

            zicetot = 0

            # ------------------------------------------
            #  reset matrix so missing pathways are set
            # ------------------------------------------
            zsolqa = np.zeros((NCLV, NCLV), dtype=dtype)
            zsolqb = np.zeros((NCLV, NCLV), dtype=dtype)

            # ----------------------------------
            #  reset new microphysics variables
            # ----------------------------------
            zfallsrce = np.zeros(NCLV, dtype=dtype)
            zfallsink = np.zeros(NCLV, dtype=dtype)
            zconvsrce = np.zeros(NCLV, dtype=dtype)
            zconvsink = np.zeros(NCLV, dtype=dtype)
            zpsupsatsrce = np.zeros(NCLV, dtype=dtype)

            # -------------------------
            #  derived variables needed
            # -------------------------

            zdp = f.paph[block, jk + 1, jl] - f.paph[block, jk, jl]  # dp
            zgdp = c.yomcst.rg / zdp  # g/dp
            zrho = f.pap[block, jk, jl] / ((c.yomcst.rd * ztp1[jk]))  # p/rt air density

            zdtgdp = s.ptsphy * zgdp  # dt g/dp
            zrdtgdp = zdp * (1 / ((s.ptsphy * c.yomcst.rg)))  # 1/(dt g/dp)

            # ------------------------------------
            #  Calculate dqs/dT correction factor
            # ------------------------------------
            #  Reminder: RETV=RV/RD-1

            # liquid
            zfacw = c.yoethf.r5les / ((ztp1[jk] - c.yoethf.r4les) ** 2)
            zcor = 1 / (1 - c.yomcst.retv * zfoeeliqt[jk])
            zdqsliqdt = zfacw * zcor * zqsliq[jk]
            zcorqsliq = 1 + c.yoethf.ralvdcp * zdqsliqdt

            # ice
            zfaci = c.yoethf.r5ies / ((ztp1[jk] - c.yoethf.r4ies) ** 2)
            zcor = 1 / (1 - c.yomcst.retv * zfoeew[jk])
            zdqsicedt = zfaci * zcor * zqsice[jk]
            zcorqsice = 1 + c.yoethf.ralsdcp * zdqsicedt

            # diagnostic mixed
            zalfaw = zfoealfa[jk]
            zalfawm = zalfaw
            zfac = zalfaw * zfacw + (1 - zalfaw) * zfaci
            zcor = 1 / (1 - c.yomcst.retv * zfoeewmt[jk])
            zdqsmixdt = zfac * zcor * zqsmix[jk]
            zcorqsmix = 1 + foeldcpm(ztp1[jk]) * zdqsmixdt

            # evaporation/sublimation limits
            zevaplimmix = max((zqsmix[jk] - zqx[NCLDQV, jk]) / zcorqsmix, 0)
            zevaplimliq = max((zqsliq[jk] - zqx[NCLDQV, jk]) / zcorqsliq, 0)
            zevaplimice = max((zqsice[jk] - zqx[NCLDQV, jk]) / zcorqsice, 0)

            # --------------------------------
            #  in-cloud consensate amount
            # --------------------------------
            ztmpa = 1 / max(za[jk], zepsec)
            zliqcld = zqx[NCLDQL, jk] * ztmpa
            zicecld = zqx[NCLDQI, jk] * ztmpa
            zlicld = zliqcld + zicecld

            # ------------------------------------------------
            #  Evaporate very small amounts of liquid and ice
            # ------------------------------------------------

            if zqx[NCLDQL, jk] < c.yrecldp.rlmin:
                zsolqa[NCLDQV, NCLDQL] = zqx[NCLDQL, jk]
                zsolqa[NCLDQL, NCLDQV] = -zqx[NCLDQL, jk]

            if zqx[NCLDQI, jk] < c.yrecldp.rlmin:
                zsolqa[NCLDQV, NCLDQI] = zqx[NCLDQI, jk]
                zsolqa[NCLDQI, NCLDQV] = -zqx[NCLDQI, jk]

            # ---------------------------------------------------------------------
            #   3.1  ICE SUPERSATURATION ADJUSTMENT
            # ---------------------------------------------------------------------
            #  Note that the supersaturation adjustment is made with respect to
            #  liquid saturation:  when T>0C
            #  ice saturation:     when T<0C
            #                      with an adjustment made to allow for ice
            #                      supersaturation in the clear sky
            #  Note also that the KOOP factor automatically clips the supersaturation
            #  to a maximum set by the liquid water saturation mixing ratio
            #  important for temperatures near to but below 0C
            # -----------------------------------------------------------------------

            # -----------------------------------
            #  3.1.1 Supersaturation limit (from Koop)
            # -----------------------------------
            #  Needs to be set for all temperatures
            zfokoop = fokoop(ztp1[jk])

            if ztp1[jk] >= c.yomcst.rtt or c.yrecldp.nssopt == 0:
                zfac = 1
                zfaci = 1
            else:
                zfac = za[jk] + zfokoop * (1 - za[jk])
                zfaci = s.ptsphy / c.yrecldp.rkooptau

            # -------------------------------------------------------------------
            #  3.1.2 Calculate supersaturation wrt Koop including dqs/dT
            #        correction factor
            #  [#Note: QSICE or QSLIQ]
            # -------------------------------------------------------------------

            # Calculate supersaturation to add to cloud
            if za[jk] > 1 - c.yrecldp.ramin:
                zsupsat = max((zqx[NCLDQV, jk] - zfac * zqsice[jk]) / zcorqsice, 0)
            else:
                # Calculate environmental humidity supersaturation
                zqp1env = (zqx[NCLDQV, jk] - za[jk] * zqsice[jk]) / max(
                    1 - za[jk], zepsilon
                )
                zsupsat = max(
                    ((1 - za[jk]) * (zqp1env - zfac * zqsice[jk])) / zcorqsice, 0
                )

            # -------------------------------------------------------------------
            #  Here the supersaturation is turned into liquid water
            #  However, if the temperature is below the threshold for homogeneous
            #  freezing then the supersaturation is turned instantly to ice.
            # --------------------------------------------------------------------

            if zsupsat > zepsec:

                if ztp1[jk] > c.yrecldp.rthomo:
                    # Turn supersaturation into liquid water
                    zsolqa[NCLDQL, NCLDQV] = zsolqa[NCLDQL, NCLDQV] + zsupsat
                    zsolqa[NCLDQV, NCLDQL] = zsolqa[NCLDQV, NCLDQL] - zsupsat
                    # Include liquid in first guess
                    zqxfg[NCLDQL] = zqxfg[NCLDQL] + zsupsat
                else:
                    # Turn supersaturation into ice water
                    zsolqa[NCLDQI, NCLDQV] = zsolqa[NCLDQI, NCLDQV] + zsupsat
                    zsolqa[NCLDQV, NCLDQI] = zsolqa[NCLDQV, NCLDQI] - zsupsat
                    # Add ice to first guess for deposition term
                    zqxfg[NCLDQI] = zqxfg[NCLDQI] + zsupsat

                # Increase cloud amount using RKOOPTAU timescale
                zsolac = (1 - za[jk]) * zfaci

            # -------------------------------------------------------
            #  3.1.3 Include supersaturation from previous timestep
            #  (Calculated in sltENDIF semi-lagrangian LDSLPHY=T)
            # -------------------------------------------------------
            if f.psupsat[block, jk, jl] > zepsec:
                if ztp1[jk] > c.yrecldp.rthomo:
                    # Turn supersaturation into liquid water
                    zsolqa[NCLDQL, NCLDQL] = (
                        zsolqa[NCLDQL, NCLDQL] + f.psupsat[block, jk, jl]
                    )
                    zpsupsatsrce[NCLDQL] = f.psupsat[block, jk, jl]
                    # Add liquid to first guess for deposition term
                    zqxfg[NCLDQL] = zqxfg[NCLDQL] + f.psupsat[block, jk, jl]
                    # Store cloud budget diagnostics if required
                else:
                    # Turn supersaturation into ice water
                    zsolqa[NCLDQI, NCLDQI] = (
                        zsolqa[NCLDQI, NCLDQI] + f.psupsat[block, jk, jl]
                    )
                    zpsupsatsrce[NCLDQI] = f.psupsat[block, jk, jl]
                    # Add ice to first guess for deposition term
                    zqxfg[NCLDQI] = zqxfg[NCLDQI] + f.psupsat[block, jk, jl]
                    # Store cloud budget diagnostics if required

                # Increase cloud amount using RKOOPTAU timescale
                zsolac = (1 - za[jk]) * zfaci
                # Store cloud budget diagnostics if required

            # on JL

            # ---------------------------------------------------------------------
            #   3.2  DETRAINMENT FROM CONVECTION
            # ---------------------------------------------------------------------
            #  * Diagnostic T-ice/liq split retained for convection
            #     Note: This link is now flexible and a future convection
            #     scheme can detrain explicit seperate budgets of:
            #     cloud water, ice, rain and snow
            #  * There is no (1-ZA) multiplier term on the cloud detrainment
            #     term, since is now written in mass-flux terms
            #  [#Note: Should use ZFOEALFACU used in convection rather than ZFOEALFA]
            # ---------------------------------------------------------------------
            if jk < klev - 1 and jk >= c.yrecldp.ncldtop - 1:
                f.plude[block, jk, jl] = f.plude[block, jk, jl] * zdtgdp

                if (
                    f.ldcum[block, jl]
                    and f.plude[block, jk, jl] > c.yrecldp.rlmin
                    and f.plu[block, jk + 1, jl] > zepsec
                ):

                    zsolac = zsolac + f.plude[block, jk, jl] / f.plu[block, jk + 1, jl]
                    # *diagnostic temperature split*
                    zalfaw = zfoealfa[jk]
                    zconvsrce[NCLDQL] = zalfaw * f.plude[block, jk, jl]
                    zconvsrce[NCLDQI] = (1 - zalfaw) * f.plude[block, jk, jl]
                    zsolqa[NCLDQL, NCLDQL] = zsolqa[NCLDQL, NCLDQL] + zconvsrce[NCLDQL]
                    zsolqa[NCLDQI, NCLDQI] = zsolqa[NCLDQI, NCLDQI] + zconvsrce[NCLDQI]

                else:

                    f.plude[block, jk, jl] = 0

                # *convective snow detrainment source
                if f.ldcum[block, jl]:
                    zsolqa[NCLDQS, NCLDQS] = (
                        zsolqa[NCLDQS, NCLDQS] + f.psnde[block, jk, jl] * zdtgdp
                    )
            # JK<KLEV

            # ---------------------------------------------------------------------
            #   3.3  SUBSIDENCE COMPENSATING CONVECTIVE UPDRAUGHTS
            # ---------------------------------------------------------------------
            #  Three terms:
            #  * Convective subsidence source of cloud from layer above
            #  * Evaporation of cloud within the layer
            #  * Subsidence sink of cloud to the layer below (Implicit solution)
            # ---------------------------------------------------------------------

            # -----------------------------------------------
            #  Subsidence source from layer above
            #                and
            #  Evaporation of cloud within the layer
            # -----------------------------------------------
            if jk > c.yrecldp.ncldtop - 1:
                zlcust = np.zeros(NCLV, dtype=dtype)

                zmf = max(0, (f.pmfu[block, jk, jl] + f.pmfd[block, jk, jl]) * zdtgdp)
                zacust = zmf * zanewm1

                for jm in range(NCLV):
                    if not llfall(jm) and iphase(jm) > 0:
                        zlcust[jm] = zmf * zqxnm1[jm]
                        # record total flux for enthalpy budget:
                        zconvsrce[jm] = zconvsrce[jm] + zlcust[jm]

                # Now have to work out how much liquid evaporates at arrival point
                # since there is no prognostic memory for in-cloud humidity, i.e.
                # we always assume cloud is saturated.

                zdtdp = (zrdcp * 0.5 * (ztp1[jk - 1] + ztp1[jk])) / f.paph[
                    block, jk, jl
                ]
                zdtforc = zdtdp * (f.pap[block, jk, jl] - f.pap[block, jk - 1, jl])
                # [#Note: Diagnostic mixed phase should be replaced below]
                zdqs = zanewm1 * zdtforc * zdqsmixdt

                for jm in range(NCLV):
                    if not llfall(jm) and iphase(jm) > 0:
                        zlfinal = max(0, zlcust[jm] - zdqs)  # lim to zero
                        # no supersaturation allowed incloud ---v
                        zevap = min((zlcust[jm] - zlfinal), zevaplimmix)
                        zlfinal = zlcust[jm] - zevap
                        zlfinalsum = zlfinalsum + zlfinal  # sum

                        zsolqa[jm, jm] = zsolqa[jm, jm] + zlcust[jm]  # whole sum
                        zsolqa[NCLDQV, jm] = zsolqa[NCLDQV, jm] + zevap
                        zsolqa[jm, NCLDQV] = zsolqa[jm, NCLDQV] - zevap

                # Reset the cloud contribution if no cloud water survives to this level:
                if zlfinalsum < zepsec:
                    zacust = 0
                zsolac = zsolac + zacust
            # on  JK>NCLDTOP

            # ---------------------------------------------------------------------
            #  Subsidence sink of cloud to the layer below
            #  (Implicit - re. CFL limit on convective mass flux)
            # ---------------------------------------------------------------------

            if jk < klev - 1:

                zmfdn = max(
                    0, (f.pmfu[block, jk + 1, jl] + f.pmfd[block, jk + 1, jl]) * zdtgdp
                )

                zsolab = zsolab + zmfdn
                zsolqb[NCLDQL, NCLDQL] = zsolqb[NCLDQL, NCLDQL] + zmfdn
                zsolqb[NCLDQI, NCLDQI] = zsolqb[NCLDQI, NCLDQI] + zmfdn

                # Record sink for cloud budget and enthalpy budget diagnostics
                zconvsink[NCLDQL] = zmfdn
                zconvsink[NCLDQI] = zmfdn

            # ----------------------------------------------------------------------
            #  3.4  EROSION OF CLOUDS BY TURBULENT MIXING
            # ----------------------------------------------------------------------
            #  NOTE: In default tiedtke scheme this process decreases the cloud
            #        area but leaves the specific cloud water content
            #        within clouds unchanged
            # ----------------------------------------------------------------------

            #  ------------------------------
            #  Define turbulent erosion rate
            #  ------------------------------
            zldifdt = c.yrecldp.rcldiff * s.ptsphy  # original version
            # Increase by factor of 5 for convective points
            if f.ktype[block, jl] > 0 and f.plude[block, jk, jl] > zepsec:
                zldifdt = c.yrecldp.rcldiff_convi * zldifdt

            #  At the moment, works on mixed RH profile and partitioned ice/liq fraction
            #  so that it is similar to previous scheme
            #  Should apply RHw for liquid cloud and RHi for ice cloud separately
            if zli[jk] > zepsec:
                # Calculate environmental humidity
                # Disabled:
                #      ZQE=(ZQX(JL,JK,NCLDQV)-ZA(JL,JK)*ZQSMIX(JL,JK))/&
                #    &      MAX(ZEPSEC,1.0_JPRB-ZA(JL,JK))
                #      ZE=ZLDIFDT(JL)*MAX(ZQSMIX(JL,JK)-ZQE,0.0_JPRB)
                ze = zldifdt * max(zqsmix[jk] - zqx[NCLDQV, jk], 0)
                zleros = za[jk] * ze
                zleros = min(zleros, zevaplimmix)
                zleros = min(zleros, zli[jk])
                zaeros = zleros / zlicld  # if linear term

                # erosion is -ve linear in l,a
                zsolac = zsolac - zaeros  # linear

                zsolqa[NCLDQV, NCLDQL] = zsolqa[NCLDQV, NCLDQL] + zliqfrac[jk] * zleros
                zsolqa[NCLDQL, NCLDQV] = zsolqa[NCLDQL, NCLDQV] - zliqfrac[jk] * zleros
                zsolqa[NCLDQV, NCLDQI] = zsolqa[NCLDQV, NCLDQI] + zicefrac[jk] * zleros
                zsolqa[NCLDQI, NCLDQV] = zsolqa[NCLDQI, NCLDQV] - zicefrac[jk] * zleros

            # ----------------------------------------------------------------------
            #  3.4  CONDENSATION/EVAPORATION DUE TO DQSAT/DT
            # ----------------------------------------------------------------------
            #   calculate dqs/dt
            #   Note: For the separate prognostic Qi and Ql, one would ideally use
            #   Qsat/DT wrt liquid/Koop here, since the physics is that new clouds
            #   forms by liquid droplets [liq] or when aqueous aerosols [Koop] form.
            #   These would then instantaneous freeze if T<-38C or lead to ice growth
            #   by deposition in warmer mixed phase clouds.  However, since we do
            #   not have a separate prognostic equation for in-cloud humidity or a
            #   statistical scheme approach in place, the depositional growth of ice
            #   in the mixed phase can not be modelled and we resort to supersaturation
            #   wrt ice instanteously converting to ice over one timestep
            #   (see Tompkins et al. QJRMS 2007 for details)
            #   Thus for the initial implementation the diagnostic mixed phase is
            #   retained for the moment, and the level of approximation noted.
            # ----------------------------------------------------------------------

            zdtdp = (zrdcp * ztp1[jk]) / f.pap[block, jk, jl]
            zdpmxdt = zdp * zqtmst
            zmfdn = 0
            if jk < klev - 1:
                zmfdn = f.pmfu[block, jk + 1, jl] + f.pmfd[block, jk + 1, jl]
            zwtot = f.pvervel[block, jk, jl] + 0.5 * c.yomcst.rg * (
                f.pmfu[block, jk, jl] + f.pmfd[block, jl, jl] + zmfdn
            )
            zwtot = min(zdpmxdt, max(-zdpmxdt, zwtot))
            zzzdt = f.phrsw[block, jk, jl] + f.phrlw[block, jk, jl]
            zdtdiab = (
                min(zdpmxdt * zdtdp, max(-zdpmxdt * zdtdp, zzzdt)) * s.ptsphy
                + c.yoethf.ralfdcp * zldefr
            )
            # note: zldefr should be set to the difference between the mixed phase functions
            # in the convection and cloud scheme, but this is not calculated, so is zero and
            # the functions must be the same
            zdtforc = zdtdp * zwtot * s.ptsphy + zdtdiab
            zqold = zqsmix[jk]
            ztold = ztp1[jk]
            ztp1[jk] = ztp1[jk] + zdtforc
            ztp1[jk] = max(ztp1[jk], 160.0)

            # formerly a call to cuadjtq(..., icall=5)
            zqp = 1 / f.pap[block, jk, jl]
            zqsat = foeewm(ztp1[jk]) * zqp
            zqsat = min(0.5, zqsat)
            zcor = 1 / (1 - c.yomcst.retv * zqsat)
            zqsat = zqsat * zcor
            zcond = (zqsmix[jk] - zqsat) / (1 + zqsat * zcor * foedem(ztp1[jk]))
            ztp1[jk] = ztp1[jk] + foeldcpm(ztp1[jk]) * zcond
            zqsmix[jk] = zqsmix[jk] - zcond
            zqsat = foeewm(ztp1[jk]) * zqp
            zqsat = min(0.5, zqsat)
            zcor = 1 / (1 - c.yomcst.retv * zqsat)
            zqsat = zqsat * zcor
            zcond1 = (zqsmix[jk] - zqsat) / (1 + zqsat * zcor * foedem(ztp1[jk]))
            ztp1[jk] = ztp1[jk] + foeldcpm(ztp1[jk]) * zcond1
            zqsmix[jk] = zqsmix[jk] - zcond1

            zdqs = zqsmix[jk] - zqold
            zqsmix[jk] = zqold
            ztp1[jk] = ztold

            # ----------------------------------------------------------------------
            #  3.4a  ZDQS(JL) > 0:  EVAPORATION OF CLOUDS
            #  ----------------------------------------------------------------------
            #  Erosion term is LINEAR in L
            #  Changed to be uniform distribution in cloud region

            # Previous function based on DELTA DISTRIBUTION in cloud:
            if zdqs > 0:
                # If subsidence evaporation term is turned off, then need to use updated
                # liquid and cloud here?
                # DISABLED: ZLEVAP = MAX(ZA(JL,JK)+ZACUST(JL),1.0_JPRB)*MIN(ZDQS(JL),ZLICLD(JL)+ZLFINALSUM(JL))
                zlevap = za[jk] * min(zdqs, zlicld)
                zlevap = min(zlevap, zevaplimmix)
                zlevap = min(zlevap, max(zqsmix[jk] - zqx[NCLDQV, jk], 0))

                # for first guess call
                zlevapl = zliqfrac[jk] * zlevap
                zlevapi = zicefrac[jk] * zlevap

                zsolqa[NCLDQV, NCLDQL] = zsolqa[NCLDQV, NCLDQL] + zliqfrac[jk] * zlevap
                zsolqa[NCLDQL, NCLDQV] = zsolqa[NCLDQL, NCLDQV] - zliqfrac[jk] * zlevap

                zsolqa[NCLDQV, NCLDQI] = zsolqa[NCLDQV, NCLDQI] + zicefrac[jk] * zlevap
                zsolqa[NCLDQI, NCLDQV] = zsolqa[NCLDQI, NCLDQV] - zicefrac[jk] * zlevap

            # ----------------------------------------------------------------------
            #  3.4b ZDQS(JL) < 0: FORMATION OF CLOUDS
            # ----------------------------------------------------------------------
            # (1) Increase of cloud water in existing clouds
            if za[jk] > zepsec and zdqs <= -c.yrecldp.rlmin:

                zlcond1 = max(-zdqs, 0)  # new limiter

                # old limiter (significantly improves upper tropospheric humidity rms)
                if za[jk] > 0.99:
                    zcor = 1 / (1 - c.yomcst.retv * zqsmix[jk])
                    zcdmax = (zqx[NCLDQV, jk] - zqsmix[jk]) / (
                        1 + zcor * zqsmix[jk] * foedem(ztp1[jk])
                    )
                else:
                    zcdmax = (zqx[NCLDQV, jk] - za[jk] * zqsmix[jk]) / za[jk]
                zlcond1 = max(min(zlcond1, zcdmax), 0)
                # end old limiter

                zlcond1 = za[jk] * zlcond1
                if zlcond1 < c.yrecldp.rlmin:
                    zlcond1 = 0

                # -------------------------------------------------------------------------
                # All increase goes into liquid unless so cold cloud homogeneously freezes
                # Include new liquid formation in first guess value, otherwise liquid
                # remains at cold temperatures until next timestep.
                # -------------------------------------------------------------------------
                if ztp1[jk] > c.yrecldp.rthomo:
                    zsolqa[NCLDQL, NCLDQV] = zsolqa[NCLDQL, NCLDQV] + zlcond1
                    zsolqa[NCLDQV, NCLDQL] = zsolqa[NCLDQV, NCLDQL] - zlcond1
                    zqxfg[NCLDQL] = zqxfg[NCLDQL] + zlcond1
                else:
                    zsolqa[NCLDQI, NCLDQV] = zsolqa[NCLDQI, NCLDQV] + zlcond1
                    zsolqa[NCLDQV, NCLDQI] = zsolqa[NCLDQV, NCLDQI] - zlcond1
                    zqxfg[NCLDQI] = zqxfg[NCLDQI] + zlcond1

            #  ! (2) Generation of new clouds (da/dt>0)

            if zdqs <= -c.yrecldp.rlmin and za[jk] < 1 - zepsec:

                # ---------------------------
                #  Critical relative humidity
                # ---------------------------
                zrhc = c.yrecldp.ramid
                zsigk = f.pap[block, jk, jl] / f.paph[block, klev, jl]
                # increase rhcrit to 1.0 towards the surface (eta>0.8)
                if zsigk > 0.8:
                    zrhc = (
                        c.yrecldp.ramid
                        + (1 - c.yrecldp.ramid) * ((zsigk - 0.8) / 0.2) ** 2
                    )

                # Commented out for CY37R1 to reduce humidity in high trop and strat
                #      ! Increase RHcrit to 1.0 towards the tropopause (trop-0.2) and above
                #      ZBOTT=ZTRPAUS(JL)+0.2_JPRB
                #      IF(ZSIGK < ZBOTT) THEN
                #        ZRHC=RAMID+(1.0_JPRB-RAMID)*MIN(((ZBOTT-ZSIGK)/0.2_JPRB)**2,1.0_JPRB)
                #      ENDIF

                # ---------------------------
                #  Supersaturation options
                # ---------------------------
                if c.yrecldp.nssopt == 0:
                    # No scheme
                    zqe = (zqx[NCLDQV, jk] - za[jk] * zqsice[jk]) / max(
                        zepsec, 1 - za[jk]
                    )
                    zqe = max(0, zqe)
                elif c.yrecldp.nssopt == 1:
                    # Tompkins
                    zqe = (zqx[NCLDQV, jk] - za[jk] * zqsice[jk]) / max(
                        zepsec, 1 - za[jk]
                    )
                    zqe = max(0, zqe)
                elif c.yrecldp.nssopt == 2:
                    # Lohmann and Karcher
                    zqe = zqx[NCLDQV, jk]
                elif c.yrecldp.nssopt == 3:
                    # Gierens
                    zqe = zqx[NCLDQV, jk] + zli[jk]

                if ztp1[jk] >= c.yomcst.rtt or c.yrecldp.nssopt == 0:
                    # No ice supersaturation allowed
                    zfac = 1
                else:
                    # Ice supersaturation
                    zfac = zfokoop

                if zqe >= zrhc * zqsice[jk] * zfac and zqe < zqsice[jk] * zfac:
                    # note: not **2 on 1-a term if ZQE is used.
                    # Added correction term ZFAC to numerator 15/03/2010
                    zacond = -((1 - za[jk]) * zfac * zdqs) / max(
                        2 * (zfac * zqsice[jk] - zqe), zepsec
                    )

                    zacond = min(zacond, 1 - za[jk])  # put the limiter back

                    # Linear term:
                    # Added correction term ZFAC 15/03/2010
                    zlcond2 = -zfac * zdqs * 0.5 * zacond  # mine linear

                    # new limiter formulation
                    zzdl = (2 * (zfac * zqsice[jk] - zqe)) / max(zepsec, 1 - za[jk])
                    # Added correction term ZFAC 15/03/2010
                    if zfac * zdqs < -zzdl:
                        # DISABLED: ZLCONDLIM=(ZA(JL,JK)-1.0_JPRB)*ZDQS(JL)-ZQSICE(JL,JK)+ZQX(JL,JK,NCLDQV)
                        zlcondlim = (
                            (za[jk] - 1) * zfac * zdqs
                            - zfac * zqsice[jk]
                            + zqx[NCLDQV, jk]
                        )
                        zlcond2 = min(zlcond2, zlcondlim)
                    zlcond2 = max(zlcond2, 0)

                    if zlcond2 < c.yrecldp.rlmin or 1 - za[jk] < zepsec:
                        zlcond2 = 0
                        zacond = 0
                    if zlcond2 == 0:
                        zacond = 0

                    # Large-scale generation is LINEAR in A and LINEAR in L
                    zsolac = zsolac + zacond  # linear

                    # ------------------------------------------------------------------------
                    #  All increase goes into liquid unless so cold cloud homogeneously freezes
                    #  Include new liquid formation in first guess value, otherwise liquid
                    #  remains at cold temperatures until next timestep.
                    # ------------------------------------------------------------------------
                    if ztp1[jk] > c.yrecldp.rthomo:
                        zsolqa[NCLDQL, NCLDQV] = zsolqa[NCLDQL, NCLDQV] + zlcond2
                        zsolqa[NCLDQV, NCLDQL] = zsolqa[NCLDQV, NCLDQL] - zlcond2
                        zqxfg[NCLDQL] = zqxfg[NCLDQL] + zlcond2
                    else:
                        # homogeneous freezing
                        zsolqa[NCLDQI, NCLDQV] = zsolqa[NCLDQI, NCLDQV] + zlcond2
                        zsolqa[NCLDQV, NCLDQI] = zsolqa[NCLDQV, NCLDQI] - zlcond2
                        zqxfg[NCLDQI] = zqxfg[NCLDQI] + zlcond2

            # ----------------------------------------------------------------------
            #  3.7 Growth of ice by vapour deposition
            # ----------------------------------------------------------------------
            #  Following Rotstayn et al. 2001:
            #  does not use the ice nuclei number from cloudaer.F90
            #  but rather a simple Meyers et al. 1992 form based on the
            #  supersaturation and assuming clouds are saturated with
            #  respect to liquid water (well mixed), (or Koop adjustment)
            #  Growth considered as sink of liquid water if present so
            #  Bergeron-Findeisen adjustment in autoconversion term no longer needed
            # ----------------------------------------------------------------------

            # --------------------------------------------------------
            # -
            # - Ice deposition following Rotstayn et al. (2001)
            # -  (monodisperse ice particle size distribution)
            # -
            # --------------------------------------------------------
            if idepice == 1:

                # --------------------------------------------------------------
                #  Calculate distance from cloud top
                #  defined by cloudy layer below a layer with cloud frac <0.01
                #  ZDZ = ZDP(JL)/(ZRHO(JL)*RG)
                # --------------------------------------------------------------

                if za[jk - 1] < c.yrecldp.rcldtopcf and za[jk] >= c.yrecldp.rcldtopcf:
                    zcldtopdist = 0
                else:
                    zcldtopdist = zcldtopdist + zdp / (zrho * c.yomcst.rg)

                # --------------------------------------------------------------
                #  only treat depositional growth if liquid present. due to fact
                #  that can not model ice growth from vapour without additional
                #  in-cloud water vapour variable
                # --------------------------------------------------------------
                if ztp1[jk] < c.yomcst.rtt and zqxfg[NCLDQL] > c.yrecldp.rlmin:
                    # T<273K

                    zvpice = (foeeice(ztp1[jk]) * c.yomcst.rv) / c.yomcst.rd
                    zvpliq = zvpice * zfokoop
                    zicenuclei = 1000 * math.exp(
                        (12.96 * (zvpliq - zvpice)) / zvpliq - 0.639
                    )

                    # ------------------------------------------------
                    #    2.4e-2 is conductivity of air
                    #    8.8 = 700**1/3 = density of ice to the third
                    # ------------------------------------------------
                    zadd = (
                        c.yomcst.rlstt
                        * (c.yomcst.rlstt / ((c.yomcst.rv * ztp1[jk])) - 1)
                    ) / ((2.4e-2 * ztp1[jk]))
                    zbdd = (c.yomcst.rv * ztp1[jk] * f.pap[block, jk, jl]) / (
                        (2.21 * zvpice)
                    )
                    zcvds = (7.8 * (zicenuclei / zrho) ** 0.666 * (zvpliq - zvpice)) / (
                        (8.87 * (zadd + zbdd) * zvpice)
                    )

                    # -----------------------------------------------------
                    #  RICEINIT=1.E-12_JPRB is initial mass of ice particle
                    # -----------------------------------------------------
                    zice0 = max(zicecld, (zicenuclei * c.yrecldp.riceinit) / zrho)

                    # ------------------
                    #  new value of ice:
                    # ------------------
                    zinew = (0.666 * zcvds * s.ptsphy + zice0**0.666) ** 1.5

                    # ---------------------------
                    #  grid-mean deposition rate:
                    # ---------------------------
                    zdepos = max(za[jk] * (zinew - zice0), 0)

                    # --------------------------------------------------------------------
                    #  Limit deposition to liquid water amount
                    #  If liquid is all frozen, ice would use up reservoir of water
                    #  vapour in excess of ice saturation mixing ratio - However this
                    #  can not be represented without a in-cloud humidity variable. Using
                    #  the grid-mean humidity would imply a large artificial horizontal
                    #  flux from the clear sky to the cloudy area. We thus rely on the
                    #  supersaturation check to clean up any remaining supersaturation
                    # --------------------------------------------------------------------
                    zdepos = min(zdepos, zqxfg[NCLDQL])  # limit to liquid water amount

                    # --------------------------------------------------------------------
                    #  At top of cloud, reduce deposition rate near cloud top to account for
                    #  small scale turbulent processes, limited ice nucleation and ice fallout
                    # --------------------------------------------------------------------
                    #       ZDEPOS = ZDEPOS*MIN(RDEPLIQREFRATE+ZCLDTOPDIST(JL)/RDEPLIQREFDEPTH,1.0_JPRB)
                    #  Change to include dependence on ice nuclei concentration
                    #  to increase deposition rate with decreasing temperatures
                    zinfactor = min(zicenuclei / 15000, 1)
                    zdepos = zdepos * min(
                        zinfactor
                        + (1 - zinfactor)
                        * (
                            c.yrecldp.rdepliqrefrate
                            + zcldtopdist / c.yrecldp.rdepliqrefdepth
                        ),
                        1,
                    )

                    # --------------
                    #  add to matrix
                    # --------------
                    zsolqa[NCLDQI, NCLDQL] = zsolqa[NCLDQI, NCLDQL] + zdepos
                    zsolqa[NCLDQL, NCLDQI] = zsolqa[NCLDQL, NCLDQI] - zdepos
                    zqxfg[NCLDQI] = zqxfg[NCLDQI] + zdepos
                    zqxfg[NCLDQL] = zqxfg[NCLDQL] - zdepos

            # --------------------------------------------------------
            # -
            # - Ice deposition assuming ice PSD
            # -
            # --------------------------------------------------------
            elif idepice == 2:

                # --------------------------------------------------------------
                #  Calculate distance from cloud top
                #  defined by cloudy layer below a layer with cloud frac <0.01
                #  ZDZ = ZDP(JL)/(ZRHO(JL)*RG)
                # --------------------------------------------------------------

                if za[jk - 1] < c.yrecldp.rcldtopcf and za[jk] >= c.yrecldp.rcldtopcf:
                    zcldtopdist = 0
                else:
                    zcldtopdist = zcldtopdist + zdp / (zrho * c.yomcst.rg)

                # --------------------------------------------------------------
                #  only treat depositional growth if liquid present. due to fact
                #  that can not model ice growth from vapour without additional
                #  in-cloud water vapour variable
                # --------------------------------------------------------------
                if ztp1[jk] < c.yomcst.rtt and zqxfg[NCLDQL] > c.yrecldp.rlmin:
                    #  ! T<273K

                    zvpice = (foeeice(ztp1[jk]) * c.yomcst.rv) / c.yomcst.rd
                    zvpliq = zvpice * zfokoop
                    zicenuclei = 1000 * math.exp(
                        (12.96 * (zvpliq - zvpice)) / zvpliq - 0.639
                    )

                    # -----------------------------------------------------
                    #  RICEINIT=1.E-12_JPRB is initial mass of ice particle
                    # -----------------------------------------------------
                    zice0 = max(zicecld, (zicenuclei * c.yrecldp.riceinit) / zrho)

                    # Particle size distribution
                    ztcg = 1
                    zfacx1i = 1

                    zaplusb = (
                        c.yrecldp.rcl_apb1 * zvpice
                        - c.yrecldp.rcl_apb2 * zvpice * ztp1[jk]
                        + f.pap[block, jk, jl] * c.yrecldp.rcl_apb3 * ztp1[jk] ** 3
                    )
                    zcorrfac = (1 / zrho) ** 0.5
                    zcorrfac2 = ((ztp1[jk] / 273) ** 1.5) * (393 / (ztp1[jk] + 120))

                    zpr02 = (zrho * zice0 * c.yrecldp.rcl_const1i) / ((ztcg * zfacx1i))

                    zterm1 = (
                        (zvpliq - zvpice)
                        * ztp1[jk] ** 2
                        * zvpice
                        * zcorrfac2
                        * ztcg
                        * c.yrecldp.rcl_const2i
                        * zfacx1i
                    ) / ((zrho * zaplusb * zvpice))
                    zterm2 = (
                        0.65 * c.yrecldp.rcl_const6i * zpr02**c.yrecldp.rcl_const4i
                        + (
                            c.yrecldp.rcl_const3i
                            * zcorrfac**0.5
                            * zrho**0.5
                            * zpr02**c.yrecldp.rcl_const5i
                        )
                        / zcorrfac2**0.5
                    )

                    zdepos = max(za[jk] * zterm1 * zterm2 * s.ptsphy, 0)

                    # --------------------------------------------------------------------
                    #  Limit deposition to liquid water amount
                    #  If liquid is all frozen, ice would use up reservoir of water
                    #  vapour in excess of ice saturation mixing ratio - However this
                    #  can not be represented without a in-cloud humidity variable. Using
                    #  the grid-mean humidity would imply a large artificial horizontal
                    #  flux from the clear sky to the cloudy area. We thus rely on the
                    #  supersaturation check to clean up any remaining supersaturation
                    # --------------------------------------------------------------------
                    zdepos = min(zdepos, zqxfg[NCLDQL])  # limit to liquid water amount

                    # --------------------------------------------------------------------
                    #  At top of cloud, reduce deposition rate near cloud top to account for
                    #  small scale turbulent processes, limited ice nucleation and ice fallout
                    # --------------------------------------------------------------------
                    #  Change to include dependence on ice nuclei concentration
                    #  to increase deposition rate with decreasing temperatures
                    zinfactor = min(zicenuclei / 15000, 1)
                    zdepos = zdepos * min(
                        zinfactor
                        + (1 - zinfactor)
                        * (
                            c.yrecldp.rdepliqrefrate
                            + zcldtopdist / c.yrecldp.rdepliqrefdepth
                        ),
                        1,
                    )

                    # --------------
                    # add to matrix
                    # --------------
                    zsolqa[NCLDQI, NCLDQL] = zsolqa[NCLDQI, NCLDQL] + zdepos
                    zsolqa[NCLDQL, NCLDQI] = zsolqa[NCLDQL, NCLDQI] - zdepos
                    zqxfg[NCLDQI] = zqxfg[NCLDQI] + zdepos
                    zqxfg[NCLDQL] = zqxfg[NCLDQL] - zdepos

            # END IF
            # on IDEPICE

            # ######################################################################
            #               4  *** PRECIPITATION PROCESSES ***
            # ######################################################################

            # ----------------------------------
            #  revise in-cloud consensate amount
            # ----------------------------------
            ztmpa = 1 / max(za[jk], zepsec)
            zliqcld = zqxfg[NCLDQL] * ztmpa
            zicecld = zqxfg[NCLDQI] * ztmpa
            zlicld = zliqcld + zicecld

            # ----------------------------------------------------------------------
            #  4.2 SEDIMENTATION/FALLING OF *ALL* MICROPHYSICAL SPECIES
            #      now that rain, snow, graupel species are prognostic
            #      the precipitation flux can be defined directly level by level
            #      There is no vertical memory required from the flux variable
            # ----------------------------------------------------------------------

            for jm in range(NCLV):
                if llfall(jm) or jm == NCLDQI:
                    # ------------------------
                    # source from layer above
                    # ------------------------
                    if jk > c.yrecldp.ncldtop - 1:
                        zfallsrce[jm] = zpfplsx[jm, jk] * zdtgdp
                        zsolqa[jm, jm] = zsolqa[jm, jm] + zfallsrce[jm]
                        zqxfg[jm] = zqxfg[jm] + zfallsrce[jm]
                        # use first guess precip----------v
                        zqpretot = zqpretot + zqxfg[jm]
                    # -------------------------------------------------
                    # sink to next layer, constant fall speed
                    # -------------------------------------------------
                    # if aerosol effect then override
                    #  note that for T>233K this is the same as above.
                    if c.yrecldp.laericesed and jm == NCLDQI:
                        zre_ice = f.pre_ice[block, jk, jl]
                        # The exponent value is from
                        # Morrison et al. JAS 2005 Appendix
                        zfall = 0.002 * zre_ice**1 * zrho
                    else:
                        zfall = zvqx(jm) * zrho
                    # -------------------------------------------------
                    # modified by Heymsfield and Iaquinta JAS 2000
                    # -------------------------------------------------
                    # DISABLED:
                    # ZFALL = ZFALL*((PAP(JL,JK)*RICEHI1)**(-0.178_JPRB)) &
                    #            &*((ZTP1(JL,JK)*RICEHI2)**(-0.394_JPRB))

                    zfallsink[jm] = zdtgdp * zfall
                    # Cloud budget diagnostic stored at end as implicit
                    # jl

            # ---------------------------------------------------------------
            #  Precip cover overlap using MAX-RAN Overlap
            #  Since precipitation is now prognostic we must
            #    1) apply an arbitrary minimum coverage (0.3) if precip>0
            #    2) abandon the 2-flux clr/cld treatment
            #    3) Thus, since we have no memory of the clear sky precip
            #       fraction, we mimic the previous method by reducing
            #       ZCOVPTOT(JL), which has the memory, proportionally with
            #       the precip evaporation rate, taking cloud fraction
            #       into account
            #    #3 above leads to much smoother vertical profiles of
            #    precipitation fraction than the Klein-Jakob scheme which
            #    monotonically increases precip fraction and then resets
            #    it to zero in a step function once clear-sky precip reaches
            #    zero.
            # ---------------------------------------------------------------
            if zqpretot > zepsec:
                zcovptot = 1 - ((1 - zcovptot) * (1 - max(za[jk], za[jk - 1]))) / (
                    1 - min(za[jk - 1], 1 - 1.0e-06)
                )
                zcovptot = max(zcovptot, c.yrecldp.rcovpmin)
                zcovpclr = max(0, zcovptot - za[jk])  # clear sky proportion
                zraincld = zqxfg[NCLDQR] / zcovptot
                zsnowcld = zqxfg[NCLDQS] / zcovptot
                zcovpmax = max(zcovptot, zcovpmax)
            else:
                zraincld = 0
                zsnowcld = 0
                zcovptot = 0  # no flux - reset cover
                zcovpclr = 0  # reset clear sky proportion
                zcovpmax = 0  # reset max cover for zzrh calc

            # ----------------------------------------------------------------------
            #  4.3a AUTOCONVERSION TO SNOW
            # ----------------------------------------------------------------------

            if ztp1[jk] <= c.yomcst.rtt:
                # -----------------------------------------------------
                #     Snow Autoconversion rate follow Lin et al. 1983
                # -----------------------------------------------------
                if zicecld > zepsec:

                    zzco = (
                        s.ptsphy
                        * c.yrecldp.rsnowlin1
                        * math.exp(c.yrecldp.rsnowlin2 * (ztp1[jk] - c.yomcst.rtt))
                    )

                    if c.yrecldp.laericeauto:
                        zlcrit = picrit_aer[jk]
                        # 0.3 = n**0.333 with N=0.027
                        zzco = (
                            zzco * (c.yrecldp.rnice / f.pnice[block, jk, jl]) ** 0.333
                        )
                    else:
                        zlcrit = c.yrecldp.rlcritsnow

                    zsnowaut = zzco * (1 - math.exp(-((zicecld / zlcrit) ** 2)))
                    zsolqb[NCLDQS, NCLDQI] = zsolqb[NCLDQS, NCLDQI] + zsnowaut

            # ----------------------------------------------------------------------
            #  4.3b AUTOCONVERSION WARM CLOUDS
            #    Collection and accretion will require separate treatment
            #    but for now we keep this simple treatment
            # ----------------------------------------------------------------------

            if zliqcld > zepsec:

                # --------------------------------------------------------
                # -
                # - Warm-rain process follow Sundqvist (1989)
                # -
                # --------------------------------------------------------
                if iwarmrain == 1:

                    zzco = c.yrecldp.rkconv * s.ptsphy

                    if c.yrecldp.laerliqautolsp:
                        zlcrit = plcrit_aer[jk]
                        # 0.3 = N**0.333 with N=125 cm-3
                        zzco = zzco * (c.yrecldp.rccn / f.pccn[block, jk, jl]) ** 0.333
                    else:
                        # Modify autoconversion threshold dependent on:
                        #  land (polluted, high CCN, smaller droplets, higher threshold)
                        #  sea  (clean, low CCN, larger droplets, lower threshold)
                        if f.plsm[block, jl] > 0.5:
                            zlcrit = c.yrecldp.rclcrit_land  # land
                        else:
                            zlcrit = c.yrecldp.rclcrit_sea  # ocean

                    # ------------------------------------------------------------------
                    # Parameters for cloud collection by rain and snow.
                    # Note that with new prognostic variable it is now possible
                    # to REPLACE this with an explicit collection parametrization
                    # ------------------------------------------------------------------
                    zprecip = (zpfplsx[NCLDQS, jk] + zpfplsx[NCLDQR, jk]) / max(
                        zepsec, zcovptot
                    )
                    zcfpr = 1 + c.yrecldp.rprc1 * math.sqrt(max(zprecip, 0))
                    # DISABLED:
                    #      ZCFPR=1.0_JPRB + RPRC1*SQRT(MAX(ZPRECIP,0.0_JPRB))*&
                    #       &ZCOVPTOT(JL)/(MAX(ZA(JL,JK),ZEPSEC))

                    if c.yrecldp.laerliqcoll:
                        # 5.0 = N**0.333 with N=125 cm-3
                        zcfpr = (
                            zcfpr * (c.yrecldp.rccn / f.pccn[block, jk, jl]) ** 0.333
                        )

                    zzco = zzco * zcfpr
                    zlcrit = zlcrit / max(zcfpr, zepsec)

                    if zliqcld / zlcrit < 20:
                        # Security for exp for some compilers
                        zrainaut = zzco * (1 - math.exp(-((zliqcld / zlcrit) ** 2)))
                    else:
                        zrainaut = zzco

                    # rain freezes instantly
                    if ztp1[jk] <= c.yomcst.rtt:
                        zsolqb[NCLDQS, NCLDQL] = zsolqb[NCLDQS, NCLDQL] + zrainaut
                    else:
                        zsolqb[NCLDQR, NCLDQL] = zsolqb[NCLDQR, NCLDQL] + zrainaut

                # --------------------------------------------------------
                # -
                # - Warm-rain process follow Khairoutdinov and Kogan (2000)
                # -
                # --------------------------------------------------------
                elif iwarmrain == 2:

                    if f.plsm[block, jl] > 0.5:
                        # land
                        zconst = c.yrecldp.rcl_kk_cloud_num_land
                        zlcrit = c.yrecldp.rclcrit_land
                    else:
                        # ocean
                        zconst = c.yrecldp.rcl_kk_cloud_num_sea
                        zlcrit = c.yrecldp.rclcrit_sea

                    if zliqcld > zlcrit:

                        zrainaut = (
                            1.5
                            * za[jk]
                            * s.ptsphy
                            * c.yrecldp.rcl_kkaau
                            * zliqcld**c.yrecldp.rcl_kkbauq
                            * zconst**c.yrecldp.rcl_kkbaun
                        )

                        zrainaut = min(zrainaut, zqxfg[NCLDQL])
                        if zrainaut < zepsec:
                            zrainaut = 0

                        zrainacc = (
                            2
                            * za[jk]
                            * s.ptsphy
                            * c.yrecldp.rcl_kkaac
                            * (zliqcld * zraincld) ** c.yrecldp.rcl_kkbac
                        )

                        zrainacc = min(zrainacc, zqxfg[NCLDQL])
                        if zrainacc < zepsec:
                            zrainacc = 0

                    else:
                        zrainaut = 0
                        zrainacc = 0

                    # If temperature < 0, then autoconversion produces snow rather than rain
                    # Explicit
                    if ztp1[jk] <= c.yomcst.rtt:
                        zsolqa[NCLDQS, NCLDQL] = zsolqa[NCLDQS, NCLDQL] + zrainaut
                        zsolqa[NCLDQS, NCLDQL] = zsolqa[NCLDQS, NCLDQL] + zrainacc
                        zsolqa[NCLDQL, NCLDQS] = zsolqa[NCLDQL, NCLDQS] - zrainaut
                        zsolqa[NCLDQL, NCLDQS] = zsolqa[NCLDQL, NCLDQS] - zrainacc
                    else:
                        zsolqa[NCLDQR, NCLDQL] = zsolqa[NCLDQR, NCLDQL] + zrainaut
                        zsolqa[NCLDQR, NCLDQL] = zsolqa[NCLDQR, NCLDQL] + zrainacc
                        zsolqa[NCLDQL, NCLDQR] = zsolqa[NCLDQL, NCLDQR] - zrainaut
                        zsolqa[NCLDQL, NCLDQR] = zsolqa[NCLDQL, NCLDQR] - zrainacc
                # on IWARMRAIN

            # on ZLIQCLD > ZEPSEC

            # ----------------------------------------------------------------------
            # RIMING - COLLECTION OF CLOUD LIQUID DROPS BY SNOW AND ICE
            #      only active if T<0degC and supercooled liquid water is present
            #      AND if not Sundquist autoconversion (as this includes riming)
            # ----------------------------------------------------------------------
            if iwarmrain > 1:

                if ztp1[jk] <= c.yomcst.rtt and zliqcld > zepsec:

                    # Fallspeed air density correction
                    zfallcorr = (c.yrecldp.rdensref / zrho) ** 0.4

                    # ------------------------------------------------------------------
                    # Riming of snow by cloud water - implicit in lwc
                    # ------------------------------------------------------------------
                    if zsnowcld > zepsec and zcovptot > 0.01:

                        # Calculate riming term
                        # Factor of liq water taken out because implicit
                        zsnowrime = (
                            0.3
                            * zcovptot
                            * s.ptsphy
                            * c.yrecldp.rcl_const7s
                            * zfallcorr
                            * (zrho * zsnowcld * c.yrecldp.rcl_const1s)
                            ** c.yrecldp.rcl_const8s
                        )

                        # Limit snow riming term
                        zsnowrime = min(zsnowrime, 1)

                        zsolqb[NCLDQS, NCLDQL] = zsolqb[NCLDQS, NCLDQL] + zsnowrime

                    # ------------------------------------------------------------------
                    # Riming of ice by cloud water - implicit in lwc
                    # NOT YET ACTIVE
                    # ------------------------------------------------------------------
                    #      IF (ZICECLD(JL)>ZEPSEC .AND. ZA(JL,JK)>0.01_JPRB) THEN
                    #
                    #        ! Calculate riming term
                    #        ! Factor of liq water taken out because implicit
                    #        ZSNOWRIME(JL) = ZA(JL,JK)*PTSPHY*RCL_CONST7S*ZFALLCORR &
                    #     &                  *(ZRHO(JL)*ZICECLD(JL)*RCL_CONST1S)**RCL_CONST8S
                    #
                    #        ! Limit ice riming term
                    #        ZSNOWRIME(JL)=MIN(ZSNOWRIME(JL),1.0_JPRB)
                    #
                    #        ZSOLQB(JL,NCLDQI,NCLDQL) = ZSOLQB(JL,NCLDQI,NCLDQL) + ZSNOWRIME(JL)
                    #
                    #      ENDIF
            # on IWARMRAIN > 1

            # ----------------------------------------------------------------------
            # 4.4a  MELTING OF SNOW and ICE
            #       with new implicit solver this also has to treat snow or ice
            #       precipitating from the level above... i.e. local ice AND flux.
            #       in situ ice and snow: could arise from LS advection or warming
            #       falling ice and snow: arrives by precipitation process
            # ----------------------------------------------------------------------

            zicetot = zqxfg[NCLDQI] + zqxfg[NCLDQS]
            zmeltmax = 0

            # If there are frozen hydrometeors present and dry-bulb temperature > 0degC
            if zicetot > zepsec and ztp1[jk] > c.yomcst.rtt:

                # Calculate subsaturation
                zsubsat = max(zqsice[jk] - zqx[NCLDQV, jk], 0)

                # Calculate difference between dry-bulb (ZTP1) and the temperature
                # at which the wet-bulb=0degC (RTT-ZSUBSAT*....) using an approx.
                # Melting only occurs if the wet-bulb temperature >0
                # i.e. warming of ice particle due to melting > cooling
                # due to evaporation.
                ztdmtw0 = (
                    ztp1[jk]
                    - c.yomcst.rtt
                    - zsubsat
                    * (
                        ztw1
                        + ztw2 * (f.pap[block, jk, ji] - ztw3)
                        - ztw4 * (ztp1[jk] - ztw5)
                    )
                )
                # Not implicit yet...
                # Ensure ZCONS1 is positive so that ZMELTMAX=0 if ZTDMTW0<0
                zcons1 = abs((ptsphy * (1 + 0.5 * ztdmtw0)) / c.yrecldp.rtaumel)
                zmeltmax = max(ztdmtw0 * zcons1 * zrldcp, 0)

            # Loop over frozen hydrometeors (ice, snow)
            for jm in range(NCLV):
                if iphase(jm) == 2:
                    jn = imelt(jm)
                    if zmeltmax > zepsec and zicetot > zepsec:
                        # Apply melting in same proportion as frozen hydrometeor fractions
                        zalfa = zqxfg[jm] / zicetot
                        zmelt = min(zqxfg[jm], zalfa * zmeltmax)
                        # needed in first guess
                        # This implies that zqpretot has to be recalculated below
                        # since is not conserved here if ice falls and liquid doesn't
                        zqxfg[jm] = zqxfg[jm] - zmelt
                        zqxfg[jn] = zqxfg[jn] + zmelt
                        zsolqa[jn, jm] = zsolqa[jn, jm] + zmelt
                        zsolqa[jm, jn] = zsolqa[jm, jn] - zmelt

            # ----------------------------------------------------------------------
            #  4.4b  FREEZING of RAIN
            # ----------------------------------------------------------------------

            # If rain present
            if zqx[NCLDQR, jk] > zepsec:

                if ztp1[jk] <= c.yomcst.rtt and ztp1[jk - 1] > c.yomcst.rtt:
                    # Base of melting layer/top of refreezing layer so
                    # store rain/snow fraction for precip type diagnosis
                    # If mostly rain, then supercooled rain slow to freeze
                    # otherwise faster to freeze (snow or ice pellets)
                    zqpretot = max(zqx[NCLDQS, jk] + zqx[NCLDQR, jk], zepsec)
                    o.prainfrac_toprfz[block, jl] = zqx(jl, jk, ncldqr) / zqpretot
                    if o.prainfrac_toprfz[block, jl] > 0.8:
                        llrainliq = True
                    else:
                        llrainliq = False

                # If temperature less than zero
                if ztp1[jk] < c.yomcst.rtt:

                    if o.prainfrac_toprfz[block, jl] > 0.8:

                        # Majority of raindrops completely melted
                        # Refreezing is by slow heterogeneous freezing

                        # Slope of rain particle size distribution
                        zlambda = (
                            c.yrecldp.rcl_fac1 / ((zrho * zqx[NCLDQR, jk]))
                        ) ** c.yrecldp.rcl_fac2

                        # Calculate freezing rate based on Bigg(1953) and Wisner(1972)
                        ztemp = c.yrecldp.rcl_fzrab * (ztp1[jk] - rtt)
                        zfrz = (
                            ptsphy
                            * (c.yrecldp.rcl_const5r / zrho)
                            * (math.exp(ztemp) - 1)
                            * zlambda**c.yrecldp.rcl_const6r
                        )
                        zfrzmax = max(zfrz, 0)

                    else:
                        # Majority of raindrops only partially melted
                        # Refreeze with a shorter timescale (reverse of melting...for now)
                        zcons1 = abs(
                            (ptsphy * (1 + 0.5 * (rtt - ztp1[jk]))) / c.yrecldp.rtaumel
                        )
                        zfrzmax = max((rtt - ztp1[jk]) * zcons1 * zrldcp, 0)

                    if zfrzmax > zepsec:
                        zfrz = min(zqx[NCLDQR, jk], zfrzmax)
                        zsolqa[NCLDQS, NCLDQR] = zsolqa[NCLDQS, NCLDQR] + zfrz
                        zsolqa[NCLDQR, NCLDQS] = zsolqa[NCLDQR, NCLDQS] - zfrz

            # ----------------------------------------------------------------------
            #  4.4c  FREEZING of LIQUID
            # ----------------------------------------------------------------------
            #  not implicit yet...
            zfrzmax = max((c.yrecldp.rthomo - ztp1[jk]) * zrldcp, 0)

            jm = NCLDQL
            jn = imelt(jm)
            if zfrzmax > zepsec and zqxfg[jm] > zepsec:
                zfrz = min(zqxfg[jm], zfrzmax)
                zsolqa[jn, jm] = zsolqa[jn, jm] + zfrz
                zsolqa[jm, jn] = zsolqa[jm, jn] - zfrz

            #  !----------------------------------------------------------------------
            #  ! 4.5   EVAPORATION OF RAIN/SNOW
            #  !----------------------------------------------------------------------

            #  !----------------------------------------
            #  ! Rain evaporation scheme from Sundquist
            #  !----------------------------------------
            if ievaprain == 1:

                # Rain
                zzrh = c.yrecldp.rprecrhmax + (
                    (1 - c.yrecldp.rprecrhmax) * zcovpmax
                ) / max(zepsec, 1 - za[jk])
                zzrh = min(max(zzrh, c.yrecldp.rprecrhmax), 1)

                zqe = (zqx[NCLDQV, jk] - za[jk] * zqsliq[jk]) / max(zepsec, 1 - za[jk])
                # ---------------------------------------------
                # humidity in moistest ZCOVPCLR part of domain
                # ---------------------------------------------
                zqe = max(0, min(zqe, zqsliq[jk]))
                if (
                    zcovpclr > zepsec
                    and zqxfg[NCLDQR] > zepsec
                    and zqe < zzrh * zqsliq[jk]
                ):
                    # note: zpreclr is a rain flux
                    zpreclr = (zqxfg[NCLDQR] * zcovpclr) / sign(
                        max(abs(zcovptot * zdtgdp), zepsilon), zcovptot * zdtgdp
                    )

                    # --------------------------------------
                    # actual microphysics formula in zbeta
                    # --------------------------------------

                    zbeta1 = (
                        (
                            sqrt(pap[block, jk, jl] / paph[block, klev, jl])
                            / c.yrecldp.rvrfactor
                        )
                        * zpreclr
                    ) / max(zcovpclr, zepsec)

                    zbeta = c.yomcst.rg * c.yrecldp.rpecons * 0.5 * zbeta1**0.5777

                    zdenom = 1 + zbeta * s.ptsphy * zcorqsliq
                    zdpr = (
                        ((zcovpclr * zbeta * (zqsliq[jk] - zqe)) / zdenom) * zdp * zrg_r
                    )
                    zdpevap = zdpr * zdtgdp

                    # ---------------------------------------------------------
                    # add evaporation term to explicit sink.
                    # this has to be explicit since if treated in the implicit
                    # term evaporation can not reduce rain to zero and model
                    # produces small amounts of rainfall everywhere.
                    # ---------------------------------------------------------

                    # Evaporate rain
                    zevap = min(zdpevap, zqxfg[NCLDQR])

                    zsolqa[NCLDQV, NCLDQR] = zsolqa[NCLDQV, NCLDQR] + zevap
                    zsolqa[NCLDQR, NCLDQV] = zsolqa[NCLDQR, NCLDQV] - zevap

                    # -------------------------------------------------------------
                    # Reduce the total precip coverage proportional to evaporation
                    # to mimic the previous scheme which had a diagnostic
                    # 2-flux treatment, abandoned due to the new prognostic precip
                    # -------------------------------------------------------------
                    zcovptot = max(
                        c.yrecldp.rcovpmin,
                        zcovptot
                        - max(0, ((zcovptot - za[jk]) * zevap) / zqxfg[NCLDQR]),
                    )

                    # Update fg field
                    zqxfg[NCLDQR] = zqxfg[NCLDQR] - zevap

            # ---------------------------------------------------------
            #  Rain evaporation scheme based on Abel and Boutle (2013)
            # ---------------------------------------------------------
            elif ievaprain == 2:

                # -----------------------------------------------------------------------
                #  Calculate relative humidity limit for rain evaporation
                #  to avoid cloud formation and saturation of the grid box
                # -----------------------------------------------------------------------
                #  Limit RH for rain evaporation dependent on precipitation fraction
                zzrh = c.yrecldp.rprecrhmax + (
                    (1 - c.yrecldp.rprecrhmax) * zcovpmax
                ) / max(zepsec, 1 - za[jk])
                zzrh = min(max(zzrh, c.yrecldp.rprecrhmax), 1)

                # Critical relative humidity
                # ZRHC=RAMID
                # ZSIGK=PAP(JL,JK)/PAPH(JL,KLEV+1)
                # Increase RHcrit to 1.0 towards the surface (eta>0.8)
                # IF(ZSIGK > 0.8_JPRB) THEN
                #  ZRHC=RAMID+(1.0_JPRB-RAMID)*((ZSIGK-0.8_JPRB)/0.2_JPRB)**2
                # ENDIF
                # ZZRH = MIN(ZRHC,ZZRH)

                # Further limit RH for rain evaporation to 80. (RHcrit in free troposphere)
                zzrh = min(0.8, zzrh)

                zqe = max(0, min(zqx[NCLDQV, jk], zqsliq[jk]))

                if (
                    zcovpclr > zepsec
                    and zqxfg[NCLDQR] > zepsec
                    and zqe < zzrh * zqsliq[jk]
                ):

                    # -------------------------------------------
                    # Abel and Boutle (2012) evaporation
                    # -------------------------------------------
                    # Calculate local precipitation (kg/kg)
                    zpreclr = zqxfg[NCLDQR] / zcovptot

                    # Fallspeed air density correction
                    zfallcorr = (c.yrecldp.rdensref / zrho) ** 0.4

                    # Saturation vapour pressure with respect to liquid phase
                    zesatliq = (c.yomcst.rv / c.yomcst.rd) * foeeliq(ztp1[jk])

                    # Slope of particle size distribution
                    zlambda = (
                        c.yrecldp.rcl_fac1 / ((zrho * zpreclr))
                    ) ** c.yrecldp.rcl_fac2  # zpreclr=kg/kg

                    zevap_denom = (
                        c.yrecldp.rcl_cdenom1 * zesatliq
                        - c.yrecldp.rcl_cdenom2 * ztp1[jk] * zesatliq
                        + c.yrecldp.rcl_cdenom3 * ztp1[jk] ** 3 * f.pap[block, jk, jl]
                    )

                    # Temperature dependent conductivity
                    zcorr2 = ((ztp1[jk] / 273.0) ** 1.5 * 393.0) / (ztp1[jk] + 120.0)
                    zka = c.yrecldp.rcl_ka273 * zcorr2

                    zsubsat = max(zzrh * zqsliq[jk] - zqe, 0)

                    zbeta = (
                        (0.5 / zqsliq[jk])
                        * ztp1[jk] ** 2
                        * zesatliq
                        * c.yrecldp.rcl_const1r
                        * (zcorr2 / zevap_denom)
                        * (
                            0.78 / (zlambda**c.yrecldp.rcl_const4r)
                            + (c.yrecldp.rcl_const2r * (zrho * zfallcorr) ** 0.5)
                            / ((zcorr2**0.5 * zlambda**c.yrecldp.rcl_const3r))
                        )
                    )

                    zdenom = 1.0 + zbeta * s.ptsphy  # *zcorqsliq(jl)
                    zdpevap = (zcovpclr * zbeta * s.ptsphy * zsubsat) / zdenom

                    # ---------------------------------------------------------
                    # Add evaporation term to explicit sink.
                    # this has to be explicit since if treated in the implicit
                    # term evaporation can not reduce rain to zero and model
                    # produces small amounts of rainfall everywhere.
                    # ---------------------------------------------------------

                    # Limit rain evaporation
                    zevap = min(zdpevap, zqxfg[NCLDQR])

                    zsolqa[NCLDQV, NCLDQR] = zsolqa[NCLDQV, NCLDQR] + zevap
                    zsolqa[NCLDQR, NCLDQV] = zsolqa[NCLDQR, NCLDQV] - zevap

                    # -------------------------------------------------------------
                    # Reduce the total precip coverage proportional to evaporation
                    # to mimic the previous scheme which had a diagnostic
                    # 2-flux treatment, abandoned due to the new prognostic precip
                    # -------------------------------------------------------------
                    zcovptot = max(
                        c.yrecldp.rcovpmin,
                        zcovptot
                        - max(0.0, ((zcovptot - za[jk]) * zevap) / zqxfg[NCLDQR]),
                    )

                    # Update fg field
                    zqxfg[NCLDQR] = zqxfg[NCLDQR] - zevap

            # on IEVAPRAIN

            # ----------------------------------------------------------------------
            # 4.5   EVAPORATION OF SNOW
            # ----------------------------------------------------------------------
            # Snow
            if ievapsnow == 1:

                zzrh = c.yrecldp.rprecrhmax + (
                    (1.0 - c.yrecldp.rprecrhmax) * zcovpmax
                ) / max(zepsec, 1.0 - za[jk])
                zzrh = min(max(zzrh, c.yrecldp.rprecrhmax), 1.0)
                zqe = (zqx[NCLDQV, jk] - za[jk] * zqsice[jk]) / max(
                    zepsec, 1.0 - za[jk]
                )

                # ---------------------------------------------
                # humidity in moistest ZCOVPCLR part of domain
                # ---------------------------------------------
                zqe = max(0.0, min(zqe, zqsice[jk]))
                if (
                    zcovpclr > zepsec
                    and zqxfg[NCLDQS] > zepsec
                    and zqe < zzrh * zqsice[jk]
                ):

                    # note: zpreclr is a rain flux a
                    zpreclr = (zqxfg[NCLDQS] * zcovpclr) / math.copysign(
                        max(abs(zcovptot * zdtgdp), zepsilon), zcovptot * zdtgdp
                    )

                    # --------------------------------------
                    # actual microphysics formula in zbeta
                    # --------------------------------------

                    zbeta1 = (
                        (
                            math.sqrt(f.pap[block, jk, jl] / f.paph[block, klev, jl])
                            / c.yrecldp.rvrfactor
                        )
                        * zpreclr
                    ) / max(zcovpclr, zepsec)

                    zbeta = c.yomcst.rg * c.yrecldp.rpecons * zbeta1**0.5777

                    zdenom = 1.0 + zbeta * s.ptsphy * zcorqsice
                    zdpr = (
                        ((zcovpclr * zbeta * (zqsice[jk] - zqe)) / zdenom) * zdp * zrg_r
                    )
                    zdpevap = zdpr * zdtgdp

                    # ---------------------------------------------------------
                    # add evaporation term to explicit sink.
                    # this has to be explicit since if treated in the implicit
                    # term evaporation can not reduce snow to zero and model
                    # produces small amounts of snowfall everywhere.
                    # ---------------------------------------------------------

                    # Evaporate snow
                    zevap = min(zdpevap, zqxfg[NCLDQS])

                    zsolqa[NCLDQV, NCLDQS] = zsolqa[NCLDQV, NCLDQS] + zevap
                    zsolqa[NCLDQS, NCLDQV] = zsolqa[NCLDQS, NCLDQV] - zevap

                    # -------------------------------------------------------------
                    # Reduce the total precip coverage proportional to evaporation
                    # to mimic the previous scheme which had a diagnostic
                    # 2-flux treatment, abandoned due to the new prognostic precip
                    # -------------------------------------------------------------
                    zcovptot = max(
                        c.yrecldp.rcovpmin,
                        zcovptot
                        - max(0.0, ((zcovptot - za[jk]) * zevap) / zqxfg[NCLDQS]),
                    )

                    # Update first guess field
                    zqxfg[NCLDQS] = zqxfg[NCLDQS] - zevap

            elif ievapsnow == 2:

                # -----------------------------------------------------------------------
                # Calculate relative humidity limit for snow evaporation
                # -----------------------------------------------------------------------
                zzrh = c.yrecldp.rprecrhmax + (
                    (1.0 - c.yrecldp.rprecrhmax) * zcovpmax
                ) / max(zepsec, 1.0 - za[jk])
                zzrh = min(max(zzrh, c.yrecldp.rprecrhmax), 1.0)
                zqe = (zqx[NCLDQV, jk] - za[jk] * zqsice[jk]) / max(
                    zepsec, 1.0 - za[jk]
                )

                # ---------------------------------------------
                # humidity in moistest ZCOVPCLR part of domain
                # ---------------------------------------------
                zqe = max(0.0, min(zqe, zqsice[jk]))
                if (
                    zcovpclr > zepsec
                    and zqx[NCLDQS, jk] > zepsec
                    and zqe < zzrh * zqsice[jk]
                ):

                    # Calculate local precipitation (kg/kg)
                    zpreclr = zqx[NCLDQS, jk] / zcovptot
                    zvpice = (foeeice(ztp1[jk]) * c.yomcst.rv) / c.yomcst.rd

                    # Particle size distribution
                    # ZTCG increases Ni with colder temperatures - essentially a
                    # Fletcher or Meyers scheme?
                    ztcg = 1.0  # v1 exp(rcl_x3i*(273.15-ztp1(jl,jk))/8.18)
                    # ZFACX1I modification is based on Andrew Barrett's results
                    zfacx1s = 1.0  # v1 (zice0/1.e-5)**0.627

                    zaplusb = (
                        c.yrecldp.rcl_apb1 * zvpice
                        - c.yrecldp.rcl_apb2 * zvpice * ztp1[jk]
                        + f.pap[block, jk, jl] * c.yrecldp.rcl_apb3 * ztp1[jk] ** 3
                    )
                    zcorrfac = (1.0 / zrho) ** 0.5
                    zcorrfac2 = ((ztp1[jk] / 273.0) ** 1.5) * (
                        393.0 / (ztp1[jk] + 120.0)
                    )

                    zpr02 = (zrho * zpreclr * c.yrecldp.rcl_const1s) / (
                        (ztcg * zfacx1s)
                    )

                    zterm1 = (
                        (zqsice[jk] - zqe)
                        * ztp1[jk] ** 2
                        * zvpice
                        * zcorrfac2
                        * ztcg
                        * c.yrecldp.rcl_const2s
                        * zfacx1s
                    ) / ((zrho * zaplusb * zqsice[jk]))
                    zterm2 = (
                        0.65 * c.yrecldp.rcl_const6s * zpr02**c.yrecldp.rcl_const4s
                        + (
                            c.yrecldp.rcl_const3s
                            * zcorrfac**0.5
                            * zrho**0.5
                            * zpr02**c.yrecldp.rcl_const5s
                        )
                        / zcorrfac2**0.5
                    )

                    zdpevap = max(zcovpclr * zterm1 * zterm2 * s.ptsphy, 0.0)

                    # --------------------------------------------------------------------
                    # Limit evaporation to snow amount
                    # --------------------------------------------------------------------
                    zevap = min(zdpevap, zevaplimice)
                    zevap = min(zevap, zqx[NCLDQS, jk])

                    zsolqa[NCLDQV, NCLDQS] = zsolqa[NCLDQV, NCLDQS] + zevap
                    zsolqa[NCLDQS, NCLDQV] = zsolqa[NCLDQS, NCLDQV] - zevap

                    # -------------------------------------------------------------
                    # Reduce the total precip coverage proportional to evaporation
                    # to mimic the previous scheme which had a diagnostic
                    # 2-flux treatment, abandoned due to the new prognostic precip
                    # -------------------------------------------------------------
                    zcovptot = max(
                        c.yrecldp.rcovpmin,
                        zcovptot
                        - max(0.0, ((zcovptot - za[jk]) * zevap) / zqx[NCLDQS, jk]),
                    )

                    # Update first guess field
                    zqxfg[NCLDQS] = zqxfg[NCLDQS] - zevap
            # on IEVAPSNOW

            # --------------------------------------
            # Evaporate small precipitation amounts
            # --------------------------------------
            for jm in range(NCLV):
                if llfall(jm) and zqxfg[jm] < c.yrecldp.rlmin:
                    zsolqa[NCLDQV, jm] = zsolqa[NCLDQV, jm] + zqxfg[jm]
                    zsolqa[jm, NCLDQV] = zsolqa[jm, NCLDQV] - zqxfg[jm]

            #######################################################################
            #            5.0  *** SOLVERS FOR A AND L ***
            # now use an implicit solution rather than exact solution
            # solver is forward in time, upstream difference for advection
            #######################################################################

            # ---------------------------
            # 5.1 solver for cloud cover
            # ---------------------------
            zanew = (za[jk] + zsolac) / (1 + zsolab)
            zanew = min(zanew, 1)
            if zanew < c.yrecldp.ramin:
                zanew = 0
            zda = zanew - zaorig[jk]
            # ---------------------------------
            # variables needed for next level
            # ---------------------------------
            zanewm1 = zanew

            # --------------------------------
            # 5.2 solver for the microphysics
            # --------------------------------

            # --------------------------------------------------------------
            # Truncate explicit sinks to avoid negatives
            # Note: Species are treated in the order in which they run out
            # since the clipping will alter the balance for the other vars
            # --------------------------------------------------------------
            llindex3 = np.empty((NCLV, NCLV), dtype=bool)
            llindex3[:, :] = False
            zsinksum = np.empty(NCLV, dtype=dtype)
            zsinksum[:] = 0

            # ----------------------------
            # collect sink terms and mark
            # ----------------------------
            for jm in range(NCLV):
                for jn in range(NCLV):
                    zsinksum[jm] = zsinksum[jm] - zsolqa[jm, jn]  # +ve total is bad

            # ---------------------------------------
            # calculate overshoot and scaling factor
            # ---------------------------------------
            zratio = np.zeros(NCLV, dtype=dtype)
            for jm in range(NCLV):
                zmax = max(zqx[jm, jk], zepsec)
                zrat = max(zsinksum[jm], zmax)
                zratio[jm] = zmax / zrat

            # --------------------------------------------
            # scale the sink terms, in the correct order,
            # recalculating the scale factor each time
            # --------------------------------------------
            zsinksum[:] = 0

            # ----------------
            # recalculate sum
            # ----------------
            for jm in range(NCLV):
                psum_solqa = 0.0
                for jn in range(NCLV):
                    psum_solqa = psum_solqa + zsolqa[jm, jn]

                # DISABLED: ZSINKSUM(JL,JM)=ZSINKSUM(JL,JM)-SUM(ZSOLQA(JL,JM,1:NCLV))
                zsinksum[jm] = zsinksum[jm] - psum_solqa

                # recalculate scaling factor
                zmm = max(zqx[jm, jk], zepsec)
                zrr = max(zsinksum[jm], zmm)
                zratio[jm] = zmm / zrr

                # scale
                zzratio = zratio[jm]
                for jn in range(NCLV):
                    if zsolqa[jm, jn] < 0:
                        zsolqa[jm, jn] = zsolqa[jm, jn] * zzratio
                        zsolqa[jn, jm] = zsolqa[jn, jm] * zzratio

            # --------------------------------------------------------------
            # 5.2.2 Solver
            # ------------------------

            # ------------------------
            # set the LHS of equation
            # ------------------------
            zqlhs = np.empty((NCLV, NCLV), dtype=dtype)
            for jm in range(NCLV):
                for jn in range(NCLV):
                    # ----------------------------------------------
                    # diagonals: microphysical sink terms+transport
                    # ----------------------------------------------
                    if jn == jm:
                        zqlhs[jn, jm] = 1 + zfallsink[jm]
                        for jo in range(NCLV):
                            zqlhs[jn, jm] = zqlhs[jn, jm] + zsolqb[jo, jn]
                    # ------------------------------------------
                    # non-diagonals: microphysical source terms
                    # ------------------------------------------
                    else:
                        zqlhs[jn, jm] = -zsolqb[
                            jn, jm
                        ]  # here is the delta T - missing from doc.

            # ------------------------
            # set the RHS of equation
            # ------------------------
            zqxn = np.empty(NCLV, dtype=dtype)
            for jm in range(NCLV):
                # ---------------------------------
                # sum the explicit source and sink
                # ---------------------------------
                zexplicit = 0.0
                for jn in range(NCLV):
                    zexplicit = zexplicit + zsolqa[jm, jn]  # sum over middle index
                zqxn[jm] = zqx[jm, jk] + zexplicit

            # -----------------------------------
            # *** solve by LU decomposition: ***
            # -----------------------------------

            # Note: This fast way of solving NCLVxNCLV system
            #       assumes a good behaviour (i.e. non-zero diagonal
            #       terms with comparable orders) of the matrix stored
            #       in ZQLHS. For the moment this is the case but
            #       be aware to preserve it when doing eventual
            #       modifications.

            # Non pivoting recursive factorization
            for jn in range(NCLV - 1):
                # number of steps
                for jm in range(jn + 1, NCLV):
                    # row index
                    zqlhs[jm, jn] = zqlhs[jm, jn] / zqlhs[jn, jn]
                    for ik in range(jn + 1, NCLV):
                        # column index
                        zqlhs[jm, ik] = zqlhs[jm, ik] - zqlhs[jm, jn] * zqlhs[jn, ik]

            # Backsubstitution
            #  step 1
            for jn in range(1, NCLV):
                for jm in range(0, jn - 1):
                    zqxn[jn] = zqxn[jn] - zqlhs[jn, jm] * zqxn[jm]
            #  step 2
            zqxn[NCLV - 1] = zqxn[NCLV - 1] / zqlhs[NCLV - 1, NCLV - 1]
            for jn in reversed(range(NCLV - 1)):
                for jm in range(jn + 1, NCLV):
                    zqxn[jn] = zqxn[jn] - zqlhs[jn, jm] * zqxn[jm]
                zqxn[jn] = zqxn[jn] / zqlhs[jn, jn]

            # Ensure no small values (including negatives) remain in cloud variables nor
            # precipitation rates.
            # Evaporate l,i,r,s to water vapour. Latent heating taken into account below
            for jn in range(NCLV - 1):
                if zqxn[jn] < zepsec:
                    zqxn[NCLDQV] = zqxn[NCLDQV] + zqxn[jn]
                    zqxn[jn] = 0

            # --------------------------------
            # variables needed for next level
            # --------------------------------
            for jm in range(NCLV):
                zqxnm1[jm] = zqxn[jm]
                zqxn2d[jm, jk] = zqxn[jm]

            # ------------------------------------------------------------------------
            # 5.3 Precipitation/sedimentation fluxes to next level
            #     diagnostic precipitation fluxes
            #     It is this scaled flux that must be used for source to next layer
            # ------------------------------------------------------------------------

            for jm in range(NCLV):
                zpfplsx[jm, jk + 1] = zfallsink[jm] * zqxn[jm] * zrdtgdp

            # Ensure precipitation fraction is zero if no precipitation
            zqpretot = zpfplsx[NCLDQS, jk + 1] + zpfplsx[NCLDQR, jk + 1]
            if zqpretot < zepsec:
                zcovptot = 0

            #######################################################################
            #              6  *** UPDATE TENDANCIES ***
            #######################################################################

            # --------------------------------
            # 6.1 Temperature and CLV budgets
            # --------------------------------

            for jm in range(NCLV - 1):

                # calculate fluxes in and out of box for conservation of TL
                zfluxq = (
                    zpsupsatsrce[jm]
                    + zconvsrce[jm]
                    + zfallsrce[jm]
                    - (zfallsink[jm] + zconvsink[jm]) * zqxn[jm]
                )

                if iphase(jm) == 1:
                    f.tendency_loc_t[block, jk, jl] = (
                        f.tendency_loc_t[block, jk, jl]
                        + c.yoethf.ralvdcp * (zqxn[jm] - zqx[jm, jk] - zfluxq) * zqtmst
                    )

                elif iphase(jm) == 2:
                    f.tendency_loc_t[block, jk, jl] = (
                        f.tendency_loc_t[block, jk, jl]
                        + c.yoethf.ralsdcp * (zqxn[jm] - zqx[jm, jk] - zfluxq) * zqtmst
                    )

                # ----------------------------------------------------------------------
                # New prognostic tendencies - ice,liquid rain,snow
                # Note: CLV arrays use PCLV in calculation of tendency while humidity
                #       uses ZQX. This is due to clipping at start of cloudsc which
                #       include the tendency already in TENDENCY_LOC_T and TENDENCY_LOC_q. ZQX was reset
                # ----------------------------------------------------------------------
                f.tendency_loc_cld[block, jm, jk, jl] = (
                    f.tendency_loc_cld[block, jm, jk, jl]
                    + (zqxn[jm] - zqx0[jm, jk]) * zqtmst
                )

            # ----------------------
            # 6.2 Humidity budget
            # ----------------------
            f.tendency_loc_q[block, jk, jl] = (
                f.tendency_loc_q[block, jk, jl]
                + (zqxn[NCLDQV] - zqx[NCLDQV, jk]) * zqtmst
            )

            # -------------------
            # 6.3 cloud cover
            # -----------------------
            f.tendency_loc_a[block, jk, jl] = (
                f.tendency_loc_a[block, jk, jl] + zda * zqtmst
            )

            # --------------------------------------------------
            # Copy precipitation fraction into output variable
            # -------------------------------------------------
            o.pcovptot[block, jk, jl] = zcovptot

        # on vertical level JK
        # ----------------------------------------------------------------------
        #                       END OF VERTICAL LOOP
        # ----------------------------------------------------------------------

        #######################################################################
        #              8  *** FLUX/DIAGNOSTICS COMPUTATIONS ***
        #######################################################################

        # --------------------------------------------------------------------
        # Copy general precip arrays back into PFP arrays for GRIB archiving
        # Add rain and liquid fluxes, ice and snow fluxes
        # --------------------------------------------------------------------
        for jk in range(klev + 1):
            o.pfplsl[block, jk, jl] = zpfplsx[NCLDQR, jk] + zpfplsx[NCLDQL, jk]
            o.pfplsn[block, jk, jl] = zpfplsx[NCLDQS, jk] + zpfplsx[NCLDQI, jk]

        # --------
        # Fluxes:
        # --------
        o.pfsqlf[block, 1, jl] = 0
        o.pfsqif[block, 1, jl] = 0
        o.pfsqrf[block, 1, jl] = 0
        o.pfsqsf[block, 1, jl] = 0
        o.pfcqlng[block, 1, jl] = 0
        o.pfcqnng[block, 1, jl] = 0
        o.pfcqrng[block, 1, jl] = 0  # rain
        o.pfcqsng[block, 1, jl] = 0  # snow
        # fluxes due to turbulence
        o.pfsqltur[block, 1, jl] = 0
        o.pfsqitur[block, 1, jl] = 0

        for jk in range(klev):

            zgdph_r = (
                -zrg_r * (f.paph[block, jk + 1, jl] - f.paph[block, jk, jl]) * zqtmst
            )
            o.pfsqlf[block, jk + 1, jl] = o.pfsqlf[block, jk, jl]
            o.pfsqif[block, jk + 1, jl] = o.pfsqif[block, jk, jl]
            o.pfsqrf[block, jk + 1, jl] = o.pfsqlf[block, jk, jl]
            o.pfsqsf[block, jk + 1, jl] = o.pfsqif[block, jk, jl]
            o.pfcqlng[block, jk + 1, jl] = o.pfcqlng[block, jk, jl]
            o.pfcqnng[block, jk + 1, jl] = o.pfcqnng[block, jk, jl]
            o.pfcqrng[block, jk + 1, jl] = o.pfcqlng[block, jk, jl]
            o.pfcqsng[block, jk + 1, jl] = o.pfcqnng[block, jk, jl]
            o.pfsqltur[block, jk + 1, jl] = o.pfsqltur[block, jk, jl]
            o.pfsqitur[block, jk + 1, jl] = o.pfsqitur[block, jk, jl]

            zalfaw = zfoealfa[jk]

            # Liquid , LS scheme minus detrainment
            o.pfsqlf[block, jk + 1, jl] = (
                o.pfsqlf[block, jk + 1, jl]
                + (
                    zqxn2d[NCLDQL, jk]
                    - zqx0[NCLDQL, jk]
                    + f.pvfl[block, jk, jl] * s.ptsphy
                    - zalfaw * f.plude[block, jk, jl]
                )
                * zgdph_r
            )
            # liquid, negative numbers
            o.pfcqlng[block, jk + 1, jl] = (
                o.pfcqlng[block, jk + 1, jl] + zlneg[NCLDQL, jk] * zgdph_r
            )

            # liquid, vertical diffusion
            o.pfsqltur[block, jk + 1, jl] = (
                o.pfsqltur[block, jk + 1, jl]
                + f.pvfl[block, jk, jl] * s.ptsphy * zgdph_r
            )

            # Rain, LS scheme
            o.pfsqrf[block, jk + 1, jl] = (
                o.pfsqrf[block, jk + 1, jl]
                + (zqxn2d[NCLDQR, jk] - zqx0[NCLDQR, jk]) * zgdph_r
            )
            # rain, negative numbers
            o.pfcqrng[block, jk + 1, jl] = (
                o.pfcqrng[block, jk + 1, jl] + zlneg[NCLDQR, jk] * zgdph_r
            )

            # Ice , LS scheme minus detrainment
            o.pfsqif[block, jk + 1, jl] = (
                o.pfsqif[block, jk + 1, jl]
                + (
                    zqxn2d[NCLDQI, jk]
                    - zqx0[NCLDQI, jk]
                    + f.pvfi[block, jk, jl] * s.ptsphy
                    - (1 - zalfaw) * f.plude[block, jk, jl]
                )
                * zgdph_r
            )
            # ice, negative numbers
            o.pfcqnng[block, jk + 1, jl] = (
                o.pfcqnng[block, jk + 1, jl] + zlneg[NCLDQI, jk] * zgdph_r
            )

            # ice, vertical diffusion
            o.pfsqitur[block, jk + 1, jl] = (
                o.pfsqitur[block, jk + 1, jl]
                + f.pvfi[block, jk, jl] * s.ptsphy * zgdph_r
            )

            # snow, LS scheme
            o.pfsqsf[block, jk + 1, jl] = (
                o.pfsqsf[block, jk + 1, jl]
                + (zqxn2d[NCLDQS, jk] - zqx0[NCLDQS, jk]) * zgdph_r
            )
            # snow, negative numbers
            o.pfcqsng[block, jk + 1, jl] = (
                o.pfcqsng[block, jk + 1, jl] + zlneg[NCLDQS, jk] * zgdph_r
            )

        # -----------------------------------
        # enthalpy flux due to precipitation
        # -----------------------------------
        for jk in range(klev + 1):
            o.pfhpsl[block, jk, jl] = -c.yomcst.rlvtt * o.pfplsl[block, jk, jl]
            o.pfhpsn[block, jk, jl] = -c.yomcst.rlstt * o.pfplsn[block, jk, jl]
    return f, o
