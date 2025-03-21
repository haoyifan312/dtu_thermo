      MODULE THERMCALC
      INTEGER, SAVE::  NC,NFMOD
      DOUBLE PRECISION, SAVE:: CU,CU1,CU2
      DOUBLE PRECISION, ALLOCATABLE:: AC(:),AC0(:),AC1(:),AC2(:)
      DOUBLE PRECISION, ALLOCATABLE:: TCR(:),PCR(:),OMG(:),TSQR(:),Q(:)
      DOUBLE PRECISION, ALLOCATABLE:: BC(:),CKMAT(:,:),AIJ2(:,:)
      PRIVATE
      PUBLIC INDATA,INDAT_FILE,THERMO,GETCRIT,VOLGEN,tERMO
      PUBLIC LUDEC,BACKSUBST
      PUBLIC INIT,FUGAC
      CONTAINS
      SUBROUTINE TEMSET(T)
C
C     TEMSET CALCULATES THE TEMPERATURE DEPENDENT PART OF THE EOS
C     PURE COMPONENT PARAMETERS, HERE SQRT (A/RT)
C
C     THE RESULTS, THE VECTORS AC0,AC1,AND AC2 REPRESENT
C     VALUE, 1ST DERIVATIVE AND 2ND DERIVATIVE OF SQRT (A/RT)
C
C     MODIFIED: NEW AC1,AC2 ARE OLD AC1,AC2 DIVIDED BY AC0
C
C
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      DATA TOLD/0.D0/
      IF (T.EQ.0) THEN
         TOLD = 0.D0
         RETURN
      ENDIF
      IF ( T.EQ.TOLD ) RETURN
      TOLD = T
      IF (T.EQ. 0) RETURN
      SQT = SQRT(T)
      SQTR = 1.D0/SQT
      T2R = 0.5D0/T
      T2RF = -3*T2R
      DO I = 1 , Nc
         ALF = TSQR(I)
         Q1 = AC(I)*(1.D0 + Q(I))*SQTR
         AC0(I) = Q1 - AC(I)*Q(I)*ALF
         AC0R = 1.D0/AC0(I)
         AC1(I) = -Q1*T2R*AC0R
         AC2(I) = T2RF*AC1(I)
      ENDDO
      DO I = 1,NC
         FF = AC0(I)+AC0(I)
         AIJ2(I:NC,I) = FF*(1.D0-CKMAT(I:NC,I))*AC0(I:NC)
         AIJ2(I,I+1:NC) = AIJ2(I+1:NC,I)
      ENDDO

      END subroutine

      SUBROUTINE THERMO_PHI (T,P,ZFEED,FUG)
    !DEC$ ATTRIBUTES DLLEXPORT :: THERMO_PHI
    !DEC$ ATTRIBUTES C :: THERMO_PHI
      implicit double precision (a-h,o-z)
      double precision zfeed(:),fug(:)
      
      call THERMO(TAN, PACK, ZFEED, FUG)
      
      end subroutine


      SUBROUTINE THERMO_DER (T,P,ZFEED,FUG,FUGT,FUGP,
     & FUGX,FLUID,FTYPE,ICON,AUX)
    !DEC$ ATTRIBUTES DLLEXPORT :: THERMO_DER
    !DEC$ ATTRIBUTES C :: THERMO_DER
      implicit double precision (a-h,o-z)
      double precision zfeed(:),fug(:)
      double precision::fugt(:),fugp(:),fugx(:,:),aux(:)
      CHARACTER:: ftype,FLUid
      integer:: icon
      
      call THERMO(T,P,ZFEED,FUG,FUGT,FUGP,
     & FUGX,FLUID,FTYPE,ICON,AUX)
      
      end subroutine
      
C
C     THERMODYNAMICS BLOCK
C
C
C     T:      (I):      Temperature (K)
C     P:      (I):      Pressure  (MPa)
C     ZFEED:  (I):      Feed (moles)
C     FUG:    (O):      Vector of log fugacity coefficients
C     FUGT:   (O):      Optional. T-derivative of FUG
C     FUGP:   (O):      Optional. P-derivative of FUG
C     FUGX:   (O):      Optional. Scaled matrix of n-derivatives of FUG
C     FLUID:  (I):      character,Optional. Desired fluid state.
C                       Possibilities are:
C                       Default (no argument):  Min G state
C                       'V' or 'V'  : Vapour
C                       'L' or 'l'  : Liquid
C     FTYPE:  (O):      Character.Calculated fluid type. Possibilities are
C                       'V' (Vapour) and 'L' (liquid).
C     ICON:   (I):      ONLY used to match old-style call
C     AUX:    (O):      Optional vector of additional results
C                       (1):   Z, the compressibility factor
C                       (2):   v dp/dv
C                       (3):   h resid
C                       (4):   s resid
C                       (5):   cp resid
C                       (6):   1/v dv/dt
C
C                       Arguments 3-6 are only calculated provided FUGT is
C                       present
C
      SUBROUTINE THERMO (T,P,ZFEED,FUG,FUGT,FUGP,
     & FUGX,FLUID,FTYPE,ICON,AUX)
      implicit double precision (a-h,o-z)
      double precision zfeed(:),fug(:)
      double precision, optional::fugt(:),fugp(:),fugx(:,:),aux(:)
      CHARACTER, optional:: ftype,FLUid
      integer,optional:: icon
      dimension X(nc)
      X = zfeed(:nc)/sum(zfeed(:nc))
      if (present(icon)) then
        nindex=0
        if (icon.ge.3) nindex=nindex+2
        if (icon.eq.2 .or. icon.gt.3) nindex=nindex+1
        nindex=-nindex
        else
        nindex=0
        if (present(fugp)) nindex=nindex+1
        if (present(fugt)) nindex=nindex+2
        if (present(fugx)) nindex=nindex+4
      endif
      CALL CUBGEN(icon,T,P,X,FuG,FugT,FugP,FugX,FLUid,ftype,aux)
      END subroutine
      
      
      SUBROUTINE CUBGEN(icon,T,P,X,FuG,FugT,FugP,FugX,FLU,ftyp,aux)
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      double precision X(nc),fug(nc)
      double precision, optional:: fugt(:),fugp(:),fugx(:,:),aux(:)
      integer,optional:: icon
      CHARACTER, OPTIONAL:: FTYP,FLU
C
C     PARAMETERS:
C
C     MT:     (I):      PHASE TYPE DESIRED
C                       1 = LIQ, -1 = VAP, 0 = MIN G PHASE
C     IC:     (O):      INDICATOR FOR PHASE TYPE; IC RETURNS
C                       1:   IF CALLED WITH MT = 1/-1 AND PHASE IS FOUND
C                       -1:  IF CALLED WITH MT = 1/-1 AND PHASE IS NOT FOUND
C                       2:   IF CALLED WITH MT = 0 AND MIN G PHASE IS LIQUID
C                       -2:  IF CALLED WITH MT = 0 AND MIN G PHASE IS VAPOUR
C     T:      (I):      TEMPERATURE (K)
C     P:      (I):      PRESSURE (ATM)
C     Z:      (O):      CALCULATED COMPRESSIBILITY FACTOR
C     XX:     (I):      MIXTURE TOTAL COMPOSITION
C     FG:     (O):      LOG FUGACITY COEFFICIENT VECTOR
C     FT:     (O):      T-DERIVATIVE OF FG
C     FP:     (O):      P-DERIVATIVE OF FG
C     FX:     (O):      SCALED COMPOSITION DERIVATIVE OF FG
C     AUX:    (O):      VARIOUS RESIDUAL PROPERTIES
C
C-----DUMMY VARIABLES
C-----LOCAL VARIABLES
      DIMENSION PD(NC),AD1(NC),ADT(NC)
      LOGICAL PSPEC
      MT=0
      nder = 0
      ntemp=0
      npres=0
      if (present(icon)) then
         if (icon.ge.2) nder=1
         if (icon.eq.2 .or. icon.gt.3) then
              ntemp=1
              npres=1
         endif
         if (icon.gt.4) ntemp=2
         else
         if (present(fugx)) nder=1
         if (present(fugt)) ntemp=1
         if (present(fugp)) npres=1
      endif
      IF (PRESENT(FLU)) then
         if (flu.eq.'l' .or. flu.eq.'L') then
           mt = 1
         elseif (flu.eq.'v' .or. flu.eq.'V') then
           mt=-1
         else
           mt=0
         endif
      endif
      CALL TEMSET(T)
C
C     ANEW CALLS MIXTURE PARAMETERS A AND B AND THEIR APPROPRIATE
C     DERIVATIVES
C
C     NTEMP = 0
      CALL ANEW(NTEMP,X,AD1,ADT,A,AT,ATT,B)
C
C     P = 0 IS TAKEN TO BE A SPECIAL CASE WHERE THE VOLUME (OR RATHER
C     V/R) IS THE INPUT AND THE CORRESPONDING P IS CALCULATED
C
      PSPEC = .TRUE.
      IF  (P .EQ. 0.D0) THEN
         PSPEC = .FALSE.
         V = ZC
         P = (1.D0/(V-B) - A/((V+Cu1*B)*(V+CU2*B)))*T
         ZC = P*V/T
      ENDIF
      TREC = 1.D0/T
      PREC = 1.D0/P
      BREC = 1.D0/B
      APT = A*P*TREC
      BPT = B*P*TREC
      IF (PSPEC) THEN
C
C     SOLVE CUBIC EOS
C
       CALL CUBIC(MT,APT,BPT,ZC)
C-------SECOND Z AND ENERGY DIFF. STORED IN AUX(1),AUX(2)
      ENDIF
      V = ZC*T*PREC
      if (present(aux)) aux(1)=ZC
      ICX = 1
      IF (V .GT. 3.D0*B) ICX = -1
C
C     PHASE TYPE RETURN INDICATOR
C
       IF (PRESENT(FTYP)) then
         if (icx.gt.0) then
            ftyp='L'
            else
            ftyp='V'
         endif
       endif
c       FTYP=IC
C------V = VOLUME/R
C
C     COEFFICIENTS FOR EOS
C
      S1 = 1./(V+CU1*B)
      S2 = 1./(V+CU2*B)
      P1 = P*TREC + A*S1*S2
      PA = -S1*S2
      PN = P1
      FAC = CU1*S1 + CU2*S2
      P2 = A*PA
      PB = P1*P1 - FAC*P2
C------ DERIVATIVE IS D(P/T) / D(V/R)
      DPDV = -P1*P1 - P2*(S1+S2)
      if (present(aux)) then
c
c     aux(8) v*dpdv
c
         if (size(aux).gt.1) AUX(2) = DPDV*T*V/P    !  returns v/p dp/dv
      endif
C
C     COEFFICIENTS FOR FUGACITY COEFFICIENT CALCULATION
C
      FN  = LOG(V*P1)
      XL2 = LOG(S1/S2)/(CU2-CU1)
      FA = -XL2*BREC
      F2 = -A*FA
      FF = FN - F2
      GB = -V*PA
      F2B = (A*GB-F2)*BREC
      FB = P1 - F2B
      FNB = P1
      FAB = -F2B/A
      GBB = -GB*FAC
      F2BB = (A*GBB-2.D0*F2B)*BREC
      FBB = P1*P1 - F2BB
      XLZ = LOG(ZC)
      FNP = FN - XLZ
C
C     ARE TEMPERATURE DERIVATIVES REQUIRED
C
      IF ( NTEMP.NE.0 ) THEN
         DFT = FA*AT
         HE = -T*DFT + Zc - 1.D0
c        HE = HE*T
C------EXCESS ENTHALPY/R STORED IN AUX(3)
         SE = -T*DFT - FF + XLZ
C------EXCESS ENTROPY/R STORED IN AUX(4)
         if (present(aux)) then
             if (size(aux).gt.2) auX(3) = HE          ! returs he/(RT)
             if (size(aux).gt.3) AUX(4) = SE
         endif
         PTR = PA*AT
         DPDT = P*TREC + T*PTR
         FTT = FA*ATT
         CP = -T*(T*FTT+2*DFT) - DPDT**2/DPDV - 1
C------EXCESS HEAT CAPACITY/R IN AUX(5)
c        DVDT = -DPDT/DPDV/T
         DVDT = -DPDT/DPDV
C---- AUX(6) IS PRESSURE DERIVATIVE OF RESID. ENTROPY
C-----AUX(7) IS PRESSURE DERIVATIVE OF ENTHALPY
         if (present(aux)) then
            if (size(aux).gt.4) AUX(5) = CP
            if (size(aux).gt.5) aux(6) = dvdt/v  ! returns t/v dv/dt
         endif
C
C     RESIDUAL CP REQUIRED
C
      ENDIF
C
C     LOG FUGACITY COEFFICIENTS
C
         FUG(:NC) = FNP + FA*AD1(:NC) + FB*BC(:NC)
         DPDVR = 1.D0/DPDV
         PD = (PN+PA*AD1(:nc)+PB*BC(:NC))*DPDVR
         IF (npres.gt.0)  FUGP(:NC) = -PD*TREC - PREC

C
C     COMPOSITION DERIVATIVES REQUIRED
C
           IF ( NDER.nE.0 ) THEN
              DO  I = 1 , Nc
                CC = 1.D0 + FNB*BC(I) + PN*PD(I)
                CA = FAB*BC(I) + PA*PD(I)
                CB = FNB + FAB*AD1(I) + FBB*BC(I) + PB*PD(I)
                FUGX(I:NC,I)=CC+CA*AD1(I:NC)+CB*BC(I:NC)+FA*AIJ2(I:NC,I)
                FUGX(I,I+1:NC)=FUGX(I+1:NC,I)
              ENDDO
           ENDIF
C
C     TEMPERATURE AND PRESSURE DERIVATIVES
C
           IF ( NTEMP.NE.0 ) THEN
              CB = FAB*AT
              DPDTT = DPDT*TREC
              FUGT(:NC) = TREC + CB*BC(:NC) + FA*ADT + DPDTT*PD
           ENDIF

      END SUBROUTINE




      SUBROUTINE CUBIC(MTYP,A,B,Z)
C
C     CUBIC EQUATION SOLVER ROUTINE
C
C     MTYP:   (I):      DESIRED PHASE TYPE 1=LIQ,-1=VAP,0=MIN. G
C     A:      (I):      EOS AP/T
C     B:      (I):      EOS BP/T
C     Z:      (O):      RETURNED COMPRESSIBILITY FACTOR
C
C------Z IS DESIRED ROOT; ZV2 IS THE OTHER ROOT; DELG IS G-DIFFERENCE
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      PARAMETER (THIRD = 1.D0/3)
      BCN = B*CU
      X2 = 1 - BCN
      X1 = A - B*(1+B+CU+2*BCN)
      X0 = B*(A-BCN*(1+B))
      Z = X2*THIRD
      F = ((Z-X2)*Z+X1)*Z - X0
      Z = B
      IF ( F.LT.0. .OR. B.GT.Z ) Z = Z + 1
C
C     IF B IS SMALL, CHECK FOR THE POSSIBILITY OF A ZERO PRESSURE
C     SOLUTION
C
      IF ( B.LT.1.D-5 ) THEN
         DD = X1*X1 - 4.D0*X2*X0
         IF ( DD.GT.0.D0 ) Z = B
      ENDIF
C
C     SET UP THIRD ORDER NEWTON FOR CUBIC EOS
C
 100  CONTINUE
      DF2 = 3.D0*Z - X2
      DF1 = Z*(DF2-X2) + X1
      DF1R = 1.D0/DF1
      F = ((Z-X2)*Z+X1)*Z - X0
      DZ = F*DF1R
      DZ = DZ*(1+DZ*DF2*DF1R)
      Z = Z - DZ
      IF ( ABS(DZ).GT.1.D-7 ) GOTO 100
C
C     CONVERGED; HOW MANY ROOTA ARE DESIRED ?
C
      IF ( MTYP*DF2.GE.0. ) THEN
C
C     FACTOR OUT FIRST ROOT AND CONSIDER REDUCED QUADRATIC
C
         E1 = Z - X2
         E0 = Z*E1 + X1
         D = E1*E1 - 4.D0*E0
C
C     D < 0 MEANS NO MORE ROOTS
C
         IF ( D.GE.0. ) THEN
C
C     GET REMAINING
C
            Z1 = .5D0*(ABS(E1)+SQRT(D))
            IF ( E1.GT.0. ) Z1 = -Z1
            IF ( Z.GT.Z1 ) Z1 = E0/Z1
C
C     REFINE BY A SINGLE NEWTON STEP
C
            DF2 = 3.D0*Z1 - X2
            DF1 = Z1*(DF2-X2) + X1
            F = ((Z1-X2)*Z1+X1)*Z1 - X0
            DF1R = 1.D0/DF1
            DZ = F*DF1R
            DZ = DZ*(1+DZ*DF2*DF1R)
            Z1 = Z1 - DZ
            IF ( Z1.GE.B ) THEN
               IF ( MTYP.EQ.0 ) THEN
C-----CALCULATE EXCESS GIBBS ENERGY DIFFERENCE FOR MULTIPLE SOLUTIONS
                  F = LOG((Z-B)/(Z1-B)) + A/(B*(CU2-CU1))
     &                *LOG((Z+CU2*B)*(Z1+CU1*B)/(Z+CU1*B)/(Z1+CU2*B) )
     &                 - (Z-Z1)
                  IF ( F.LT.0. ) Z=Z1
               ELSEIF ( MTYP.EQ.1 ) THEN
                  IF ( Z1.LT.Z ) Z = Z1
               ELSE
                  IF ( Z1.GT.Z ) Z = Z1
               ENDIF
            ENDIF
         ENDIF
      ENDIF
      END SUBROUTINE




      SUBROUTINE ANEW(NTEMP,X,AD1,ADT,A0,AT,att,B0)
C
C     ANEW IS AN INTERNAL SUBROUTINE FOR TERMO
C
C     X:      (I):      NORMALIZED COMPOSITION
C     A0:     (O):      MIXTURE A (A/RT)-PARAMETER
C     AT:     (O):      T-DERIVATIVE OF A0
C     ATT:    (O):      T-DERIVATIVE OF AT
C     B0:     (O):      MIXTURE B-PARAMETER
C     AD1:    (O):      COMPSOTION DERIVATIVE OF A0
C     ADT:    (O):      T-DERIVATIVE OF AD1
C
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      DIMENSION X(NC),AD1(NC),ADT(NC)
      DIMENSION Y(NC),BV(NC)

      B0 = DOT_PRODUCT(X,BC)
C------ HERE USED: SINGLE SUM IN B
      AD1 = MATMUL(X,AIJ2)
      A0  = 0.5D0*DOT_PRODUCT(X,AD1)
      IF (NTEMP.GT.0) THEN
         Y = AC1*X
         BV = MATMUL(Y,AIJ2)
         ADT = BV + AC1*AD1
         AT  = 0.5D0*DOT_PRODUCT(X,ADT)
         ATT=0.d0
         DO I = 1,NC
           ATT = ATT+Y(I)*BV(I)+X(I)*AD1(I)*AC2(I)
         ENDDO
      ENDIF
      END subroutine
C
C     MIXTURE PARAMETERS A AND B
C
C
      SUBROUTINE GETCRIT(I,TCRIT,PCRIT,OMEGA)
    !DEC$ ATTRIBUTES DLLEXPORT :: GETCRIT
    !DEC$ ATTRIBUTES C :: GETCRIT
C
C     GETCR RETURNS CRITICAL PROPERTIES AND COMPONENT INDICATORS
C
C     I       (I):      COMPONENT INDEX
C     TCRIT   (O):      COMPONENT CRITICAL T
C     PCRIT   (O):      COMPONENT CRITICAL P
C     OMEGA   (O):      COMPONENT ACENTRIC FACTOR
C     ITYP    (O):      COMPONENT TYPE:
C                       SET = 0 FOR HYDROCARBON, 1 FOR OTHERS
C                       (ONLY UED IN MULTIPHASE FLASH)
C
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      TCRIT = TCR(I)
      PCRIT = PCR(I)
      OMEGA = OMG(I)
      END SUBROUTINE
C
C
C
      SUBROUTINE INDAT_FILE(NCA,ICEQ,NUNIT)
C
C     PARAMETERS:
C
C     NCA     (O):      NO. OF COMPONENTS
C     ICEQ    (I):      EOS. 0 = SRK, 1 = PR, 2 = PR78
C     NUNIT   (I):      UNIT NO. FOR DATAFILE
C
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      CHARACTER*4, ALLOCATABLE:: NEW(:)
      READ (NUNIT,*) NC
      NCA = NC
      ALLOCATE (NEW(NC))
      IF (ALLOCATED(CKMAT)) THEN
        DEALLOCATE (AC,AC0,AC1,AC2,TCR,PCR,OMG,TSQR,Q,BC,CKMAT,AIJ2)
      ENDIF
      ALLOCATE (AC(NC),AC0(NC),AC1(NC),AC2(NC),TCR(NC),PCR(NC),OMG(NC),
     &        TSQR(NC),Q(NC),BC(NC),CKMAT(NC,NC),AIJ2(NC,NC))

      WRITE(*,40)
   40 FORMAT(' COMP   TCRIT     PCRIT  OMEGA   M-FACTOR')
   60 FORMAT(1X,A4,2F9.2,1X,F7.4,F9.4)
      DO I = 1,NC
           READ (NUNIT,*) NEW(I),TCR(I),PCR(I),OMG(I)
           TSQR(I)=1./SQRT(TCR(I))
C
C     EOS CHARACTERISTIC CONSTANTS
C
           BC(I)=CONB*TCR(I)/PCR(I)
           AC(I)=CONA*TCR(I)/SQRT(PCR(I))
C-----ICEQ=0: SRK
C-----ICEQ=1: PENG-ROBINSON
           IF (ICEQ.EQ.0) THEN
                Q(I)=.48+OMG(I)*(1.574-.176*OMG(I))
  		   ELSE IF (ICEQ.EQ.1) THEN
			   IF(OMG(I).LE. 0.49) THEN
				   Q(I)=.37464+OMG(I)*(1.54226-.26992*OMG(I))
               ELSE
				   Q(I)=.379642+OMG(I)*(1.48503-OMG(I)*(.16444+.01666*OMG(I)))
			   ENDIF
		   ENDIF
           WRITE(*,60) NEW(I),TCR(I),PCR(I),OMG(I),Q(I)
      ENDDO
      CU = MIN(ICEQ,1)
      CALL CONST(CONA,CONB)
C
C        INTERACTION COEFFICIENTS
C
      CKMAT=0
      DO I = 2,NC
           READ (NUNIT,*) CKMAT(:I-1,I)
           CKMAT(I,:I-1)=CKMAT(:I-1,I)
      ENDDO
      WRITE(*,90) NEW(:NC)
      DO J=1,NC
         WRITE(*,120) NEW(J),(CKMAT(:J-1,J))
      ENDDO
   90 FORMAT(/,' BINARY INTERACTION COEFFICIENTS ',//,(5X,15(A4,1X)))
  120 FORMAT(A4,15F5.2,/,(4X,15F5.2))
      END SUBROUTINE





      SUBROUTINE INDATA(NCA,ICEQ,LIST)
    !DEC$ ATTRIBUTES DLLEXPORT :: INDATA
    !DEC$ ATTRIBUTES C :: INDATA
C
C     A SMALL 'DATABASE ROUTINE' FOR PROPERTIES INPUT
C
C     NC:     (I):      NO. OF COMPONENT
C     ICEQ:   (I):      EOS; 0 = SRK, 1 = PRNG-ROBINSON
C     LIST:   (I):      VECTOR OF COMPONENT INDICES
C
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      PARAMETER (MAXC=15)
      DIMENSION LIST(*)
      CHARACTER*4 NAME(MAXC)
      character*8 new
      COMMON /NNAME / NEW(MAXC)
      REAL*4 T(MAXC),P(MAXC),OMEGA(MAXC),CKV(MAXC,MAXC)
C-----IDENTIFY HYDROCARBONS
      DATA NAME/
     &      ' C1 ',' C2 ',' C3 ',' IC4',' C4 ',
     &      ' IC5',' C5 ',' C6 ',' C7 ',' C8 ',
     &      ' C9 ',' H2O',' N2 ',' CO2',' H2S'/
C------CRITICAL TEMPERATURES (K) AND PRESSURES (ATM) (REID ET AL.)
      DATA T/
     &   190.6, 305.4, 369.8, 408.1, 425.2,
     &   460.4, 469.6, 507.4, 540.2, 568.8,
     &   594.6, 647.3, 126.2, 304.2, 373.2/
      DATA P/
     &     45.4, 48.2, 41.9, 36.0, 37.5,
     &     33.4, 33.3, 29.3, 27.0, 24.5,
     &     22.8,218.0, 33.5, 72.8, 88.2/
C------ACENTRIC FACTORS (REID ET AL.)
      DATA OMEGA/
     &     0.008, 0.098, 0.152, 0.176, 0.193,
     &     0.227, 0.251, 0.296, 0.351, 0.394,
     &     0.440, 0.344, 0.040, 0.225, 0.100/
C------NONZERO KIJ FOR SRK-EQUATION (REID ET AL., HEIDEMANN ET AL.)
      DATA (CKV( 1,J),J= 2,15)/10*0.,.45,.02,.12,.08/
      DATA (CKV( 2,J),J= 3,15)/ 9*0.,.45,.06,.15,.07/
      DATA (CKV( 3,J),J= 4,15)/ 8*0.,.53,.08,.15,.07/
      DATA (CKV( 4,J),J= 5,15)/ 7*0.,.52,.08,.15,.06/
      DATA (CKV( 5,J),J= 6,15)/ 6*0.,.52,.08,.15,.06/
      DATA (CKV( 6,J),J= 7,15)/ 5*0.,.50,.08,.15,.06/
      DATA (CKV( 7,J),J= 8,15)/ 4*0.,.50,.08,.15,.06/
      DATA (CKV( 8,J),J= 9,15)/ 3*0.,.50,.08,.15,.05/
      DATA (CKV( 9,J),J=10,15)/ 2*0.,.50,.08,.15,.04/
      DATA (CKV(10,J),J=11,15)/   0.,.50,.08,.15,.04/
      DATA (CKV(11,J),J=12,15)/      .50,.08,.15,.03/
      DATA (CKV(12,J),J=13,15)/                 3*0./
      DATA (CKV(13,J),J=14,15)/                 2*0./
      DATA  CKV(14,15)        /                  .12/
C---- GAS CONSTANT FOR ENTHALPY COEVERSION
      IF (ALLOCATED(CKMAT)) THEN
        DEALLOCATE (AC,AC0,AC1,AC2,TCR,PCR,OMG,TSQR,Q,BC,CKMAT,AIJ2)
      ENDIF
      nc=nca
      n=nca
      ALLOCATE (AC(NC),AC0(NC),AC1(NC),AC2(nC),TCR(NC),PCR(NC),OMG(NC),
     &        TSQR(NC),Q(NC),BC(NC),CKMAT(NC,NC),AIJ2(NC,NC))
      DO K=1,15
         CKV(K,K) = 0.
      ENDDO
      DO K=1,14
         DO I=K+1,15
              CKV(I,K) = CKV(K,I)
         ENDDO
      ENDDO
      CU=MIN(ICEQ,1)
C
C     GET EOS CONSTANTS
C
      CALL CONST(CONA,CONB)
C
C     RUN THROUGH LIST ELEMENTS TO SELECT COMPONENTS
C
      WRITE(*,40)
      DO I=1,N
         L=LIST(I)
         NEW(I)=NAME(L)
C
C     CRITICAL TEMPERATURE AND PRESSURE
C
         TCR(I)=T(L)
         TSQR(I)=1./SQRT(TCR(I))
         PCR(I)=P(L)*0.1013D0
C
C     EOS CHARACTERISTIC CONSTANTS
C
         BC(I)=CONB*TCR(I)/PCR(I)
         AC(I)=CONA*TCR(I)/SQRT(PCR(I))
         OM=OMEGA(L)
         OMG(I)=OM
         IF (ICEQ.EQ.0) THEN
                Q(I)=.48+OMG(I)*(1.574-.176*OMG(I))
         ELSE IF (ICEQ.EQ.1) THEN
			IF(OMG(I).LE. 0.49) THEN
				Q(I)=.37464+OMG(I)*(1.54226-.26992*OMG(I))
            ELSE
				Q(I)=.379642+OMG(I)*(1.48503-OMG(I)*(.16444+.01666*OMG(I)))
			ENDIF
		 ENDIF
         WRITE(*,60) NEW(I),TCR(I),PCR(I),omg(I),Q(I)
      ENDDO
   40 FORMAT(' COMP   TCRIT     PCRIT  OMEGA  M-FACTOR ')
   60 FORMAT(1X,A4,2F9.2,1X,F7.4,F9.4)
      WRITE(*,90) (NEW(K),K=1,N)
   90 FORMAT(/,' BINARY INTERACTION COEFFICIENTS ',//,5X,15(A4,1X))
      DO J=1,N
         L1=LIST(J)
         DO I=1,N
            L=LIST(I)
c           CKV(I,J)=CKV(L1,L)
c           CKV(J,I)=CKV(I,J)
            ckmat(i,j)=ckv(l1,l)
            ckmat(j,i)=ckv(l ,l1)


         ENDDO
         IF (J.GT.1) WRITE(*,120) NEW(J),CKmat(:J-1,J)
      ENDDO
  120 FORMAT(A4,15F5.2)
      CALL TEMSET(0.D0)
      END SUBROUTINE
      SUBROUTINE INIT(NCA,NOMA,LIST)
C
C     A SMALL 'DATABASE ROUTINE' FOR PROPERTIES INPUT
C
C     NC:     (I):      NO. OF COMPONENT
C     ICEQ:   (I):      EOS; 0 = SRK, 1 = PRNG-ROBINSON
C     LIST:   (I):      VECTOR OF COMPONENT INDICES
C
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      PARAMETER (MAXC=15)
      DIMENSION LIST(*)
      CHARACTER*4 NAME(MAXC)
      character*8 new
      COMMON /NNAME / NEW(MAXC)
      REAL*4 T(MAXC),P(MAXC),OMEGA(MAXC),CKV(MAXC,MAXC)
C-----IDENTIFY HYDROCARBONS
      DATA NAME/
     &      ' C1 ',' C2 ',' C3 ',' IC4',' C4 ',
     &      ' IC5',' C5 ',' C6 ',' C7 ',' C8 ',
     &      ' C9 ',' H2O',' N2 ',' CO2',' H2S'/
C------CRITICAL TEMPERATURES (K) AND PRESSURES (ATM) (REID ET AL.)
      DATA T/
     &   190.6, 305.4, 369.8, 408.1, 425.2,
     &   460.4, 469.6, 507.4, 540.2, 568.8,
     &   594.6, 647.3, 126.2, 304.2, 373.2/
      DATA P/
     &     45.4, 48.2, 41.9, 36.0, 37.5,
     &     33.4, 33.3, 29.3, 27.0, 24.5,
     &     22.8,218.0, 33.5, 72.8, 88.2/
C------ACENTRIC FACTORS (REID ET AL.)
      DATA OMEGA/
     &     0.008, 0.098, 0.152, 0.176, 0.193,
     &     0.227, 0.251, 0.296, 0.351, 0.394,
     &     0.440, 0.344, 0.040, 0.225, 0.100/
C------NONZERO KIJ FOR SRK-EQUATION (REID ET AL., HEIDEMANN ET AL.)
      DATA (CKV( 1,J),J= 2,15)/10*0.,.45,.02,.12,.08/
      DATA (CKV( 2,J),J= 3,15)/ 9*0.,.45,.06,.15,.07/
      DATA (CKV( 3,J),J= 4,15)/ 8*0.,.53,.08,.15,.07/
      DATA (CKV( 4,J),J= 5,15)/ 7*0.,.52,.08,.15,.06/
      DATA (CKV( 5,J),J= 6,15)/ 6*0.,.52,.08,.15,.06/
      DATA (CKV( 6,J),J= 7,15)/ 5*0.,.50,.08,.15,.06/
      DATA (CKV( 7,J),J= 8,15)/ 4*0.,.50,.08,.15,.06/
      DATA (CKV( 8,J),J= 9,15)/ 3*0.,.50,.08,.15,.05/
      DATA (CKV( 9,J),J=10,15)/ 2*0.,.50,.08,.15,.04/
      DATA (CKV(10,J),J=11,15)/   0.,.50,.08,.15,.04/
      DATA (CKV(11,J),J=12,15)/      .50,.08,.15,.03/
      DATA (CKV(12,J),J=13,15)/                 3*0./
      DATA (CKV(13,J),J=14,15)/                 2*0./
      DATA  CKV(14,15)        /                  .12/
C---- GAS CONSTANT FOR ENTHALPY COEVERSION
      IF (ALLOCATED(CKMAT)) THEN
        DEALLOCATE (AC,AC0,AC1,AC2,TCR,PCR,OMG,TSQR,Q,BC,CKMAT,AIJ2)
      ENDIF
      nc=nca
      n=nca
      NFMOD = NOMA
      NFMOD = MIN(MAX(NOMA,1),3)
      ICEQ=0
      ALLOCATE (AC(NC),AC0(NC),AC1(NC),AC2(nC),TCR(NC),PCR(NC),OMG(NC),
     &        TSQR(NC),Q(NC),BC(NC),CKMAT(NC,NC),AIJ2(NC,NC))

      DO K=1,15
         CKV(K,K) = 0.
      ENDDO
      DO K=1,14
         DO I=K+1,15
              CKV(I,K) = CKV(K,I)
         ENDDO
      ENDDO
      CU=MIN(ICEQ,1)
C
C     GET EOS CONSTANTS
C
      CALL CONST(CONA,CONB)
C
C     RUN THROUGH LIST ELEMENTS TO SELECT COMPONENTS
C
      WRITE(*,40)
      DO I=1,N
         L=LIST(I)
         NEW(I)=NAME(L)
C
C     CRITICAL TEMPERATURE AND PRESSURE
C
         TCR(I)=T(L)
         TSQR(I)=1./SQRT(TCR(I))
         PCR(I)=P(L)*0.1013D0
C
C     EOS CHARACTERISTIC CONSTANTS
C
         BC(I)=CONB*TCR(I)/PCR(I)
         AC(I)=CONA*TCR(I)/SQRT(PCR(I))
         OM=OMEGA(L)
         OMG(I)=OM
         IF (ICEQ.EQ.0) THEN
                Q(I)=.48+OMG(I)*(1.574-.176*OMG(I))
         ELSE IF (ICEQ.EQ.1) THEN
			IF(OMG(I).LE. 0.49) THEN
				Q(I)=.37464+OMG(I)*(1.54226-.26992*OMG(I))
            ELSE
				Q(I)=.379642+OMG(I)*(1.48503-OMG(I)*(.16444+.01666*OMG(I)))
			ENDIF
		 ENDIF
         WRITE(*,60) NEW(I),TCR(I),PCR(I),omg(I),Q(I)
      ENDDO
   40 FORMAT(' COMP   TCRIT     PCRIT  OMEGA  M-FACTOR ')
   60 FORMAT(1X,A4,2F9.2,1X,F7.4,F9.4)



      DO  I=1,NC
         DO J=I,NC
            IF (J.EQ. I) THEN
                CKMAT(I,J) = 0.D0
            ELSE
              CKMAT(I,J) = (J-I)*.02
              CKMAT(J,I) = -.7*CKMAT(I,J)
            ENDIF
         ENDDO
      ENDDO
      CALL TEMSET(0.D0)
      END SUBROUTINE

      SUBROUTINE FUGAC (T,P,XX,ZZ,FG)
      IMPLICIT REAL*8(A-H,O-Z)
      PARAMETER(MAXC=15,MC2=(MAXC*(MAXC+1))/2)
C-----DUMMY VARIABLES
      DIMENSION XX(NC),FG(NC)
C-----LOCAL VARIABLES
      DIMENSION X(NC),AD1(NC)
      CALL TEMSET(T)
      X = XX/SUM(XX)
      B = DOT_PRODUCT(X,BC)
      A = 0.D0
      DO I=1,NC
         AA1 = 0.D0
         DO J=1,NC
              IF (NFMOD .LE. 1) THEN
                   AIF = (CKMAT(I,J)+CKMAT(J,I))/2.D0
                   ELSEIF (NFMOD .EQ. 2) THEN
                   AIF = CKMAT(I,J)
                   ELSE
                   CP = 5.D-3*P
                   IF (CP .GT. .5) CP = .5
                   IF (I.EQ.J) CP = 0.
                   AIF = (CKMAT(I,J)+CKMAT(J,I))/2.D0 + CP
              ENDIF
              AA1 = AA1 + AC0(J)*X(J)*(1.D0-AIF)
         ENDDO
         AD1(I) = 2.D0*AC0(I)*AA1
         A = A + X(I)*AD1(I)
      ENDDO
      A = .5D0*A
      APT=A*P/T
      BPT=B*P/T
      CALL CUBIC(0,APT,BPT,Z)
      zz = z
C-------SECOND Z AND ENERGY DIFF. STORED IN AUX(1),AUX(2)
      V=Z*T/P
      S1=1./(V+CU1*B)
      S2=1./(V+CU2*B)
      P1=P/T+A*S1*S2
      PA=-S1*S2
C------ DERIVATIVE IS D(P/T) / D(V/R)
      XL1=LOG(V*P1)
      FN=XL1
      XL2=LOG(S1/S2)/(CU2-CU1)
      FA=-XL2/B
      F2=-A*FA
      GB=-V*PA
      F2B = (A*GB-F2)/B
      FB=P1-F2B
      XLZ=LOG(Z)
      FNP=FN-XLZ
      DO I=1,NC
           FG(I)=FNP+FA*AD1(I)+FB*BC(I)
      ENDDO
      END SUBROUTINE












      SUBROUTINE CONST(AC,BC)
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
C
C     BASIC EOS CONSTANTS FOR CUBIC EOS
C
C     C = 0:  REDLICH-KWONG
C     C = 1:  PENG-ROBINSON
C
         U = 1 + CU
         W = -CU
         D = U*U - 4*W
         CU2 = (U+SQRT(D))/2
         CU1 = W/CU2
         S1 = 1 + ((U+W)*(U+3)-W)/2
         S2 = 1 + U + W
         R = SQRT(S1*S1-S2*S2*S2)
         A = (S1+R)**(1.D0/3)
         B = S2/A
         X = A + B + 1
         BC = 1./(3*X+U-1)
         AC = BC*SQRT(X*(X*X-3*W)-U*W)
      END SUBROUTINE
      SUBROUTINE VOLGEN(T,V,P,DPDT,DPDV,X,FuG,FugT,FugV,FugX)
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      Double precision  X(nc),fug(nc)
      double precision, optional:: FUGT(:),FUGV(:),FUGX(:,:)
      DIMENSION AD1(nc),ADT(nc)
C
C-------parameters of volgen (crit. point calc)
C
C       t     temperature (k)                          (input)
C       v     volume/r    (k*mol/mpa)                  (input)
C       p     pressure    (mpa)                        (output)
C       dpdt  derivative of p   wrt. t                 (output)
C       dpdv  derivative of p   wrt. v                 (output)
C       x---  mixture mole numbers                     (input)
C       fg    logarithm of fugacity/mole number ratio  (output)
C       ft    t-derivative of fg (const. vol)          (output)
C       fv    vol-derivative of fg (const temp)        (output)
C       fx    comp-derivative of fg (const t. and v)   (output)
C---------------------------------------------------
C---  MODIFIED AND CORRECTED NOVEMBER 1992
C---
C---------------------------------------------------
      ntemp=0
      if (present(fugt)) ntemp=1
      CALL TEMSET(T)
C
C     sumn is total moles; volgen calculates in terms of overall moles
C     and overall volume
C
      SUMN = sum(x(:nc))
c
c     mixture parameters
c
      CALL ANEW(NTEMP,X,AD1,ADT,A,AT,ATT,B)
      BREC = 1.D0/B
      PN = 1.D0/(V-B)
      IERR =0
      IF (PN.LT. 0.D0) THEN
         IERR = 1
         RETURN
      ENDIF
      REP = SUMN*PN
      S1 = 1.D0/(V+Cu1*B)
      S2 = 1.D0/(V+Cu2*B)
      ATTR = A*S1*S2
      P = T*(REP-ATTR)
C
C   p=t*(sumn/(v-b)-a/(v+c1*b)/(v+c2*b))
C
C     terms like pa, pn, pb , fa, fb, fn, fab indicate the derivatives
C     of p and the reduced residual helmholz function with repsect to
C     total moles and the equation of state parameters
C
C     composition derivatives are the obtainable using the chain rule
C
      PA = -S1*S2
      FAC = Cu1*S1 + Cu2*S2
      P2 = A*PA
      PB = PN*REP - FAC*P2
      DPDV = -PN*REP - P2*(S1+S2)
      DPDV = DPDV*T
      XL2 = LOG(S1/S2)/(Cu2-Cu1)
      FA = -XL2*BREC
      F2 = -A*FA
      GB = -V*PA
      F2B = (A*GB-F2)*BREC
      FB = REP - F2B
      FNB = PN
      FAB = -F2B/A
      GBB = -GB*FAC
      F2BB = (A*GBB-2*F2B)*BREC
      FBB = REP*PN - F2BB
      DFDN = LOG(T*PN)
      DO  I = 1,Nc
            FuG(I) = DFDN + FA*AD1(I) + FB*BC(I)
      enddo
      if (present(fugv)) fugv(:nc)=-pn - pa*ad1(:nc)-pb*bc(:nc)
C
C  volume derivative of fugacity from composition deriv. of pressure
C
      if (present(fugx)) then
            DO I = 1,nc
               CC = FNB*BC(I)
               CA = FAB*BC(I)
               CB = FNB + FAB*AD1(I) + FBB*BC(I)
               fugx(i:nc,i)=cc+ca*ad1(i:nc)+cb*bc(i:nc)+fa*aij2(i:nc,i)
               fugx(i,i+1:nc)=fugx(i+1:nc,i)
            enddo
      ENDIF
      if (present(fugt))then
               CC = 1.D0/T
               CB = FAB*AT
               DPDT = PA*AT
               DPDT = t*dpdt+p/t
               DO 180 I = 1,Nc
                  FugT(:nc) = CC + CB*BC(:nc) + FA*ADT(:nc)
  180          CONTINUE
      ENDIF
      END subroutine
      subroutine termo(icon,ntyp,mtyp,t,p,zcomp,zz,fg,ft,fp,fx,aux)
      implicit double precision (a-h,o-z)
      dimension zz(:),fg(:),ft(:),fp(:),fx(:,:),aux(:)
      character flud,ftype
      if (ntyp.eq.1) then
         flud='L'
         elseif(ntyp.eq.0) then
         flud=' '
         else
         flud='V'
      endif
      call thermo(t,p,ZZ,fg,ft,fp,fx,flud,ftype,ICON,AUX)
      zcomp=aux(1)
      if (ntyp.eq.0) then
         if (ftype.eq.'L') then
           mtyp=2
         else
           mtyp=-2
         endif
      elseif (ntyp.eq.1) then
         if (ftype.eq.'L') then
           mtyp=1
         else
           mtyp=-1
         endif
      else
         if (ftype.eq.'V') then
           mtyp=1
         else
           mtyp=-1
         endif
      endif
      end subroutine

      SUBROUTINE LUDEC(N,INDX,A)
C
C
C   LU-DECOMPOSITION OF MATRIX A BY CROUT'S METHOD
C
C   ND:   ROW DIMENSION OF A IN CALLING PROGRAM
C   N:    ACTUAL SIZE OF A
C   INDX: PIVOT VECTOR FOR ROW INTERCHANGE DURING FACTORIZATION
C   A:    MATRIX TO FACTORIZE; ON EXIT: FACTORIZED MATRIX
C
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      DIMENSION A(:,:) , INDX(N)
C
C     SET INDX-VECTOR
C
      DO I = 1 , N
         INDX(I) = I
            DO K = I , N
               A(K,I) = A(K,I) -DOT_PRODUCT(A(:I-1,I),A(K,:I-1))
            ENDDO
C
C     PIVOT
C
            XMAX = ABS(A(I,I))
            IPIV = I
            DO K = I + 1 , N
              Y = ABS(A(K,I))
              IF ( Y.GT.XMAX ) THEN
                IPIV = K
                XMAX = Y
              ENDIF
            ENDDO
            IF ( IPIV.NE.I ) THEN
               INDX(I) = IPIV
               DO K = 1 , N
                  X = A(I,K)
                  A(I,K) = A(IPIV,K)
                  A(IPIV,K) = X
               ENDDO
            ENDIF
            A(I,I) = 1.D0/A(I,I)
            A(I+1:N,I) = A(I,I)*A(I+1:N,I)
            DO K = I + 1 , N
               A(I,K) = A(I,K) - DOT_PRODUCT(A(:I-1,K),A(I,:I-1))
            ENDDO
      ENDDO
      END SUBROUTINE
      SUBROUTINE BACKSUBST(N,INDX,A,V)
C
C     BACK-SUBSTITUTION FOR CROUT-BASED LU-ROUTINE
C
C     PLAIN FORTRAN VERSION
C
C     ND:       ROW DIMENSION OF A
C     N:        ACTUAL SIZE OF A
C     INDX:     PIVOT VECTOR CALCULATED IN LU
C     A:        FACTORIZED MATRIX
C     V:        RHS-VECTOR, ON EXIT SOLN. TO LINEAR EQNS.
C
      IMPLICIT REAL*8(A-H,O-Z)
      DIMENSION A(:,:) , V(N) , INDX(N)
C
C     REORDER
C
      DO I = 1 , N - 1
         IPIV = INDX(I)
         IF ( IPIV.NE.I ) THEN
            X = V(I)
            V(I) = V(IPIV)
            V(IPIV) = X
         ENDIF
      ENDDO
C
C     L-INVERS
C
      DO I = 2 , N
         V(I) = V(I) - DOT_PRODUCT(V(:I-1),A(I,:I-1))
      ENDDO
C
C     U-INVERS
C
      DO I = N , 1 , -1
         v(I)  =  ( V(I)  - DOT_PRODUCT(V(I+1:N),A(I,I+1:N)))*A(I,I)
      ENDDO
      END SUBROUTINE
      END MODULE
