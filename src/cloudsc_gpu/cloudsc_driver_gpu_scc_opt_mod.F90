! (C) Copyright 1988- ECMWF.
!
! This software is licensed under the terms of the Apache Licence Version 2.0
! which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
!
! In applying this licence, ECMWF does not waive the privileges and immunities
! granted to it by virtue of its status as an intergovernmental organisation
! nor does it submit to any jurisdiction.

MODULE CLOUDSC_DRIVER_GPU_SCC_OPT_MOD

  USE PARKIND1,ONLY:JPIM,JPRB
  USE YOMPHYDER,ONLY:STATE_TYPE
  USE YOECLDP,ONLY:NCLV,YRECLDP,TECLDP
  USE CLOUDSC_MPI_MOD,ONLY:NUMPROC,IRANK
  USE TIMER_MOD,ONLY:PERFORMANCE_TIMER,GET_THREAD_NUM

  USE CLOUDSC_GPU_SCC_OPT_MOD,ONLY:CLOUDSC_SCC_OPT
  USE CUDAFOR

  IMPLICIT NONE

CONTAINS

  SUBROUTINE CLOUDSC_DRIVER_GPU_SCC_OPT( &
     & NUMOMP,NPROMA,NLEV,NGPTOT,NGPBLKS,NGPTOTG,KFLDX,PTSPHY, &
     & PT,PQ, &
     & BUFFER_CML,BUFFER_TMP,BUFFER_LOC, &
     & PVFA,PVFL,PVFI,PDYNA,PDYNL,PDYNI, &
     & PHRSW,PHRLW, &
     & PVERVEL,PAP,PAPH, &
     & PLSM,LDCUM,KTYPE, &
     & PLU,PLUDE,PSNDE,PMFU,PMFD, &
     & PA, &
     & PCLV,PSUPSAT,&
     & PLCRIT_AER,PICRIT_AER,PRE_ICE, &
     & PCCN,PNICE,&
     & PCOVPTOT,PRAINFRAC_TOPRFZ, &
     & PFSQLF,PFSQIF,PFCQNNG,PFCQLNG, &
     & PFSQRF,PFSQSF,PFCQRNG,PFCQSNG, &
     & PFSQLTUR,PFSQITUR, &
     & PFPLSL,PFPLSN,PFHPSL,PFHPSN, &
     & LPERFTEST)
    ! Driver routine that invokes the optimized CLAW-based CLOUDSC GPU kernel

    INTEGER(KIND=JPIM)                                    :: NUMOMP,NPROMA,NLEV,NGPTOT,NGPBLKS,NGPTOTG
    INTEGER(KIND=JPIM)                                    :: KFLDX
    REAL(KIND=JPRB)                                       :: PTSPHY ! Physics timestep
    REAL(KIND=JPRB),INTENT(IN)    :: PT(NPROMA,NLEV,NGPBLKS) ! T at start of callpar
    REAL(KIND=JPRB),INTENT(IN)    :: PQ(NPROMA,NLEV,NGPBLKS) ! Q at start of callpar
    REAL(KIND=JPRB),INTENT(INOUT) :: BUFFER_CML(NPROMA,NLEV,3+NCLV,NGPBLKS) ! Storage buffer for TENDENCY_CML
    REAL(KIND=JPRB),INTENT(INOUT) :: BUFFER_TMP(NPROMA,NLEV,3+NCLV,NGPBLKS) ! Storage buffer for TENDENCY_TMP
    REAL(KIND=JPRB),INTENT(INOUT) :: BUFFER_LOC(NPROMA,NLEV,3+NCLV,NGPBLKS) ! Storage buffer for TENDENCY_LOC
    REAL(KIND=JPRB),INTENT(IN)    :: PVFA(NPROMA,NLEV,NGPBLKS) ! CC from VDF scheme
    REAL(KIND=JPRB),INTENT(IN)    :: PVFL(NPROMA,NLEV,NGPBLKS) ! Liq from VDF scheme
    REAL(KIND=JPRB),INTENT(IN)    :: PVFI(NPROMA,NLEV,NGPBLKS) ! Ice from VDF scheme
    REAL(KIND=JPRB),INTENT(IN)    :: PDYNA(NPROMA,NLEV,NGPBLKS) ! CC from Dynamics
    REAL(KIND=JPRB),INTENT(IN)    :: PDYNL(NPROMA,NLEV,NGPBLKS) ! Liq from Dynamics
    REAL(KIND=JPRB),INTENT(IN)    :: PDYNI(NPROMA,NLEV,NGPBLKS) ! Liq from Dynamics
    REAL(KIND=JPRB),INTENT(IN)    :: PHRSW(NPROMA,NLEV,NGPBLKS) ! Short-wave heating rate
    REAL(KIND=JPRB),INTENT(IN)    :: PHRLW(NPROMA,NLEV,NGPBLKS) ! Long-wave heating rate
    REAL(KIND=JPRB),INTENT(IN)    :: PVERVEL(NPROMA,NLEV,NGPBLKS) !Vertical velocity
    REAL(KIND=JPRB),INTENT(IN)    :: PAP(NPROMA,NLEV,NGPBLKS) ! Pressure on full levels
    REAL(KIND=JPRB),INTENT(IN)    :: PAPH(NPROMA,NLEV+1,NGPBLKS) ! Pressure on half levels
    REAL(KIND=JPRB),INTENT(IN)    :: PLSM(NPROMA,NGPBLKS) ! Land fraction (0-1)
    LOGICAL,INTENT(IN)            :: LDCUM(NPROMA,NGPBLKS) ! Convection active
    INTEGER(KIND=JPIM),INTENT(IN) :: KTYPE(NPROMA,NGPBLKS) ! Convection type 0,1,2
    REAL(KIND=JPRB),INTENT(IN)    :: PLU(NPROMA,NLEV,NGPBLKS) ! Conv. condensate
    REAL(KIND=JPRB),INTENT(INOUT) :: PLUDE(NPROMA,NLEV,NGPBLKS) ! Conv. detrained water
    REAL(KIND=JPRB),INTENT(IN)    :: PSNDE(NPROMA,NLEV,NGPBLKS) ! Conv. detrained snow
    REAL(KIND=JPRB),INTENT(IN)    :: PMFU(NPROMA,NLEV,NGPBLKS) ! Conv. mass flux up
    REAL(KIND=JPRB),INTENT(IN)    :: PMFD(NPROMA,NLEV,NGPBLKS) ! Conv. mass flux down
    REAL(KIND=JPRB),INTENT(IN)    :: PA(NPROMA,NLEV,NGPBLKS) ! Original Cloud fraction (t)
    REAL(KIND=JPRB),INTENT(IN)    :: PCLV(NPROMA,NLEV,NCLV,NGPBLKS)
    REAL(KIND=JPRB),INTENT(IN)    :: PSUPSAT(NPROMA,NLEV,NGPBLKS)
    REAL(KIND=JPRB),INTENT(IN)    :: PLCRIT_AER(NPROMA,NLEV,NGPBLKS)
    REAL(KIND=JPRB),INTENT(IN)    :: PICRIT_AER(NPROMA,NLEV,NGPBLKS)
    REAL(KIND=JPRB),INTENT(IN)    :: PRE_ICE(NPROMA,NLEV,NGPBLKS)
    REAL(KIND=JPRB),INTENT(IN)    :: PCCN(NPROMA,NLEV,NGPBLKS) ! liquid cloud condensation nuclei
    REAL(KIND=JPRB),INTENT(IN)    :: PNICE(NPROMA,NLEV,NGPBLKS) ! ice number concentration (cf. CCN)

    REAL(KIND=JPRB),INTENT(INOUT) :: PCOVPTOT(NPROMA,NLEV,NGPBLKS) ! Precip fraction
    REAL(KIND=JPRB),INTENT(OUT) :: PRAINFRAC_TOPRFZ(NPROMA,NGPBLKS)
    ! Flux diagnostics for DDH budget
    REAL(KIND=JPRB),INTENT(OUT) :: PFSQLF(NPROMA,NLEV+1,NGPBLKS) ! Flux of liquid
    REAL(KIND=JPRB),INTENT(OUT) :: PFSQIF(NPROMA,NLEV+1,NGPBLKS) ! Flux of ice
    REAL(KIND=JPRB),INTENT(OUT) :: PFCQLNG(NPROMA,NLEV+1,NGPBLKS) ! -ve corr for liq
    REAL(KIND=JPRB),INTENT(OUT) :: PFCQNNG(NPROMA,NLEV+1,NGPBLKS) ! -ve corr for ice
    REAL(KIND=JPRB),INTENT(OUT) :: PFSQRF(NPROMA,NLEV+1,NGPBLKS) ! Flux diagnostics
    REAL(KIND=JPRB),INTENT(OUT) :: PFSQSF(NPROMA,NLEV+1,NGPBLKS) !    for DDH, generic
    REAL(KIND=JPRB),INTENT(OUT) :: PFCQRNG(NPROMA,NLEV+1,NGPBLKS) ! rain
    REAL(KIND=JPRB),INTENT(OUT) :: PFCQSNG(NPROMA,NLEV+1,NGPBLKS) ! snow
    REAL(KIND=JPRB),INTENT(OUT) :: PFSQLTUR(NPROMA,NLEV+1,NGPBLKS) ! liquid flux due to VDF
    REAL(KIND=JPRB),INTENT(OUT) :: PFSQITUR(NPROMA,NLEV+1,NGPBLKS) ! ice flux due to VDF
    REAL(KIND=JPRB),INTENT(OUT) :: PFPLSL(NPROMA,NLEV+1,NGPBLKS) ! liq+rain sedim flux
    REAL(KIND=JPRB),INTENT(OUT) :: PFPLSN(NPROMA,NLEV+1,NGPBLKS) ! ice+snow sedim flux
    REAL(KIND=JPRB),INTENT(OUT) :: PFHPSL(NPROMA,NLEV+1,NGPBLKS) ! Enthalpy flux for liq
    REAL(KIND=JPRB),INTENT(OUT) :: PFHPSN(NPROMA,NLEV+1,NGPBLKS) ! ice number concentration (cf. CCN)

    ! Local declarations of promoted temporaries
    INTEGER(KIND=JPIM) :: JL

    REAL(KIND=JPRB) :: TMP1(NPROMA,NLEV+1,1,NGPBLKS)
    REAL(KIND=JPRB) :: TMP2(NPROMA,NLEV+1,1,NGPBLKS)

    INTEGER(KIND=JPIM) :: JKGLO,IBL,ICEND
    TYPE(PERFORMANCE_TIMER) :: TIMER
    INTEGER(KIND=JPIM) :: TID ! thread id from 0 .. NUMOMP - 1

    ! Local copy of cloud parameters for offload
    TYPE(TECLDP) :: LOCAL_YRECLDP

    TYPE(CUDAEVENT) :: START, STOP
    INTEGER :: NREPS, JREP, ISTAT
    REAL :: TIME2
    LOGICAL :: LPERFTEST

    ISTAT = CUDAEVENTCREATE(START)
    ISTAT = CUDAEVENTCREATE(STOP)

    NREPS = 1
    IF (LPERFTEST) NREPS = 30

    NGPBLKS=(NGPTOT/NPROMA)+MIN(MOD(NGPTOT,NPROMA),1)
1003 format(5x,'NUMPROC=',i0,', NUMOMP=',i0,', NGPTOTG=',i0,', NPROMA=',i0,', NGPBLKS=',i0)
    if(irank==0) then
      write(0,1003) NUMPROC,NUMOMP,NGPTOTG,NPROMA,NGPBLKS
    endif

    TMP1(:,:,:,:) = 0
    TMP2(:,:,:,:) = 0

    ! Global timer for the parallel region
    CALL TIMER%START(NUMOMP)

    ! Workaround for PGI / OpenACC oddities:
    ! Create a local copy of the parameter struct to ensure they get
    ! moved to the device the in ``acc data`` clause below
    LOCAL_YRECLDP=YRECLDP

!$acc data &
!$acc copyin( &
!$acc   pt,pq,buffer_cml,buffer_tmp,pvfa, &
!$acc   pvfl,pvfi,pdyna,pdynl,pdyni,phrsw,phrlw,pvervel, &
!$acc   pap,paph,plsm,ldcum,ktype,plu,psnde, &
!$acc   pmfu,pmfd,pa,pclv,psupsat,plcrit_aer,picrit_aer, &
!$acc   pre_ice,pccn,pnice, local_yrecldp) &
!$acc copy( &
!$acc   buffer_loc,plude,pcovptot,prainfrac_toprfz,tmp1,tmp2) &
!$acc copyout( &
!$acc   pfsqlf,pfsqif,pfcqnng, &
!$acc   pfcqlng ,pfsqrf,pfsqsf,pfcqrng,pfcqsng,pfsqltur, &
!$acc   pfsqitur,pfplsl,pfplsn,pfhpsl,pfhpsn)

    ! Local timer for each thread
    TID=GET_THREAD_NUM()
    CALL TIMER%THREAD_START(TID)

    DO JREP=1,NREPS
      IF (JREP == 11) ISTAT = CUDAEVENTRECORD(START,0)

      CALL CLOUDSC_SCC_OPT &
        (1,NPROMA,NLEV,NGPBLKS,PTSPHY, &
         PT(:,:,:),PQ(:,:,:), &
         BUFFER_TMP(:,:,1,:),BUFFER_TMP(:,:,3,:),BUFFER_TMP(:,:,2,:),BUFFER_TMP(:,:,4:8,:), &
         BUFFER_LOC(:,:,1,:),BUFFER_LOC(:,:,3,:),BUFFER_LOC(:,:,2,:),BUFFER_LOC(:,:,4:8,:), &
         PVFA(:,:,:),PVFL(:,:,:),PVFI(:,:,:),PDYNA(:,:,:),PDYNL(:,:,:),PDYNI(:,:,:), &
         PHRSW(:,:,:),PHRLW(:,:,:), &
         PVERVEL(:,:,:),PAP(:,:,:),PAPH(:,:,:), &
         PLSM(:,:),LDCUM(:,:),KTYPE(:,:), &
         PLU(:,:,:),PLUDE(:,:,:),PSNDE(:,:,:),PMFU(:,:,:),PMFD(:,:,:), &
         !---prognostic fields
         PA(:,:,:),PCLV(:,:,:,:),PSUPSAT(:,:,:), &
         !-- arrays for aerosol-cloud interactions
         PLCRIT_AER(:,:,:),PICRIT_AER(:,:,:), &
         PRE_ICE(:,:,:), &
         PCCN(:,:,:),PNICE(:,:,:), &
         !---diagnostic output
         PCOVPTOT(:,:,:),PRAINFRAC_TOPRFZ(:,:), &
         !---resulting fluxes
         PFSQLF(:,:,:),PFSQIF(:,:,:),PFCQNNG(:,:,:),PFCQLNG(:,:,:), &
         PFSQRF(:,:,:),PFSQSF(:,:,:),PFCQRNG(:,:,:),PFCQSNG(:,:,:), &
         PFSQLTUR(:,:,:),PFSQITUR(:,:,:), &
         PFPLSL(:,:,:),PFPLSN(:,:,:),PFHPSL(:,:,:),PFHPSN(:,:,:), &
         TMP1(:,:,:,:),TMP2(:,:,:,:),&
         LOCAL_YRECLDP, &
         NGPTOT=NGPTOT,NPROMA=NPROMA)

    enddo

    ISTAT = CUDAEVENTRECORD(STOP,0)
    ISTAT = CUDADEVICESYNCHRONIZE()
    ISTAT = CUDAEVENTELAPSEDTIME(TIME2, START, STOP)
    TIME2 = TIME2 / (1.e3 * (nreps - 10))
    CALL TIMER%THREAD_END(TID)

!$acc end data

    CALL TIMER%END()

    IF (LPERFTEST) THEN
      PRINT "('ELAPSED TIME PER RUN [ms]:',F8.1)", TIME2*1000
      PRINT "('PERFORMANCE [MP/s]       :',F8.1)", NGPTOT * NLEV / TIME2 / 1e6
    ELSE
      ! On GPUs, adding block-level column totals is cumbersome and
      ! error prone, and of little value due to the large number of
      ! processing "thread teams". Instead we register the total here.
      CALL TIMER%THREAD_LOG(TID=TID, IGPC=NGPTOT)
      CALL TIMER%PRINT_PERFORMANCE(NPROMA, NGPBLKS, NGPTOT)
    ENDIF

    IF (.NOT. LPERFTEST) THEN
      OPEN(10, FILE="serialized_gpu_scc_opt.dat", status='replace', form='unformatted')
      WRITE(10) SHAPE(PLUDE), 0, PLUDE
      WRITE(10) SHAPE(PCOVPTOT), 0, PCOVPTOT
      WRITE(10) SHAPE(PRAINFRAC_TOPRFZ), 0, PRAINFRAC_TOPRFZ
      WRITE(10) SHAPE(PFSQLF), 0, PFSQLF
      WRITE(10) SHAPE(PFSQIF), 0, PFSQIF
      WRITE(10) SHAPE(PFCQLNG), 0, PFCQLNG
      WRITE(10) SHAPE(PFCQNNG), 0, PFCQNNG
      WRITE(10) SHAPE(PFSQRF), 0, PFSQRF
      WRITE(10) SHAPE(PFSQSF), 0, PFSQSF
      WRITE(10) SHAPE(PFCQRNG), 0, PFCQRNG
      WRITE(10) SHAPE(PFCQSNG), 0, PFCQSNG
      WRITE(10) SHAPE(PFSQLTUR), 0, PFSQLTUR
      WRITE(10) SHAPE(PFSQITUR), 0, PFSQITUR
      WRITE(10) SHAPE(PFPLSL), 0, PFPLSL
      WRITE(10) SHAPE(PFPLSN), 0, PFPLSN
      WRITE(10) SHAPE(PFHPSL), 0, PFHPSL
      WRITE(10) SHAPE(PFHPSN), 0, PFHPSN
      WRITE(10) SHAPE(BUFFER_TMP), 0, BUFFER_TMP
      WRITE(10) SHAPE(BUFFER_LOC), 0, BUFFER_LOC
      CLOSE(10)
    ENDIF

    ISTAT = CUDAEVENTDESTROY(START)
    ISTAT = CUDAEVENTDESTROY(STOP)

  ENDSUBROUTINE CLOUDSC_DRIVER_GPU_SCC_OPT

ENDMODULE CLOUDSC_DRIVER_GPU_SCC_OPT_MOD
