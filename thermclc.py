#from numba import jit
import math
import numpy as np

# COMMON VARIABLES
MAXC=50
NC=1
NFMOD=1

TOLD=0.0

CU=0
CU1=0
CU2=0

AC = np.zeros(MAXC)
AC0 = np.zeros(MAXC)
AC1 = np.zeros(MAXC)
AC2 = np.zeros(MAXC)
TCR = np.zeros(MAXC)
PCR = np.zeros(MAXC)
OMG = np.zeros(MAXC)
TSQR= np.zeros(MAXC)
Q =  np.zeros(MAXC)
BC = np.zeros(MAXC)
CKMAT =  np.zeros((MAXC,MAXC))   
AIJ2 =  np.zeros((MAXC,MAXC))   

NAME = [" C1 ", " C2 ", " C3 ", " IC4", " C4 ",
        " IC5", " C5 ", " C6 ", " C7 ", " C8 ",
        " C9 ", " H2O", " N2 ", " CO2", " H2S"]
NAME = [n.rstrip(' ') for n in NAME]
NAME = [n.lstrip(' ') for n in NAME]

#@jit
def GETCRIT(i):
# GETCRIT RETURNS CRITICAL PROPERTIES AND COMPONENT INDICATORS
#     I       (I):      COMPONENT INDEX
#     TCRIT   (O):      COMPONENT CRITICAL T
#     PCRIT   (O):      COMPONENT CRITICAL P
#     OMEGA   (O):      COMPONENT ACENTRIC FACTOR
#     ITYP    (O):      COMPONENT TYPE:
#                       SET = 0 FOR HYDROCARBON, 1 FOR OTHERS
#                       (ONLY UED IN MULTIPHASE FLASH)
    global TCR, PCR, OMG
#
    TCRIT = TCR[i]
    PCRIT = PCR[i]
    OMEGA = OMG[i]
    return(TCRIT, PCRIT, OMEGA)

#@jit
def GETKIJ(i,j):
#
    global CKMAT
#
    KIJ = CKMAT[i,j]
    return(KIJ)

#@jit
def SET_THERM_GLOBAL():
#
    global NC
    global AC, AC0, AC1, AC2, TCR, PCR, OMG, TSQR, Q, BC, CKMAT, AIJ2
    AC = np.resize(AC, NC)
    AC0 = np.resize(AC0, NC)
    AC1 = np.resize(AC1, NC)
    AC2 = np.resize(AC2, NC)
    TCR = np.resize(TCR, NC)
    PCR = np.resize(PCR, NC)
    OMG = np.resize(OMG, NC)
    TSQR= np.resize(TSQR, NC)
    Q =  np.resize(Q, NC)
    BC = np.resize(BC, NC)
    CKMAT =  np.resize(CKMAT,(NC,NC))    
    AIJ2 =  np.resize(CKMAT,(NC,NC))    
       
#@jit
def CONST():
# BASIC EOS CONSTANTS FOR CUBIC EOS
# C = 0:  REDLICH-KWONG
# C = 1:  PENG-ROBINSON
    global CU, CU1, CU2
#    
    U = 1 + CU
    W = -CU
    D = U*U - 4*W
    CU2 = (U+math.sqrt(D))/2
    CU1 = W/CU2
    S1 = 1 + ((U+W)*(U+3)-W)/2
    S2 = 1 + U + W
    R = math.sqrt(S1*S1-S2*S2*S2)
    A = math.pow(S1+R,1/3)
    B = S2/A
    X = A + B + 1
    BC = 1./(3*X+U-1)
    AC = BC*math.sqrt(X*(X*X-3*W)-U*W)
    return(AC, BC)

#@jit
def INDATA(NCA,ICEQ,LIST):
#    
    global NC, CU
    global AC, TCR, PCR, OMG, TSQR, Q, BC, CKMAT
#    
    T = np.zeros(15)
    P = np.zeros(15)
    OMEGA = np.zeros(15)
    CKV = np.zeros((15,15))

# CRITICAL TEMPERATURES (K) AND PRESSURES (ATM) (REID ET AL.)
    T = np.array([190.6, 305.4, 369.8, 408.1, 425.2, \
                  460.4, 469.6, 507.4, 540.2, 568.8, \
                  594.6, 647.3, 126.2, 304.2, 373.2])
    P = np.array([45.4, 48.2, 41.9, 36.0, 37.5, \
                  33.4, 33.3, 29.3, 27.0, 24.5, \
                  22.8,218.0, 33.5, 72.8, 88.2]);
# ACENTRIC FACTORS (REID ET AL.)
    OMEGA = np.array([0.008, 0.098, 0.152, 0.176, 0.193, \
                      0.227, 0.251, 0.296, 0.351, 0.394, \
                      0.440, 0.344, 0.040, 0.225, 0.100]);
# NONZERO KIJ FOR SRK-EQUATION (REID ET AL., HEIDEMANN ET AL.)
# note the strange way of indexing and slicing in Python
    CKV[0,  11:15] = np.array([0.45, 0.02, 0.12, 0.08])
    CKV[1,  11:15] = np.array([0.45, 0.06, 0.15, 0.07])
    CKV[2,  11:15] = np.array([0.53, 0.08, 0.15, 0.07])
    CKV[3,  11:15] = np.array([0.52, 0.08, 0.15, 0.06])
    CKV[4,  11:15] = np.array([0.52, 0.08, 0.15, 0.06])
    CKV[5,  11:15] = np.array([0.50, 0.08, 0.15, 0.06])
    CKV[6,  11:15] = np.array([0.50, 0.08, 0.15, 0.06])
    CKV[7,  11:15] = np.array([0.50, 0.08, 0.15, 0.05])
    CKV[8,  11:15] = np.array([0.50, 0.08, 0.15, 0.04])
    CKV[9,  11:15] = np.array([0.50, 0.08, 0.15, 0.04])
    CKV[10, 11:15] = np.array([0.50, 0.08, 0.15, 0.03])
    CKV[11, 11:15] = np.array([0.00, 0.00, 0.00, 0.00])
    CKV[12, 11:15] = np.array([0.00, 0.00, 0.00, 0.00])
    CKV[13, 11:15] = np.array([0.00, 0.00, 0.00, 0.12])
#
    NC = NCA
    N = NCA
    SET_THERM_GLOBAL;
    NEW = ["NAME"]*NC
#    
    for k in range(0,14):
        CKV[k+1:15,k]=CKV[k,k+1:15]
    CU = min(ICEQ,1)
# GET EOS CONSTANTS
    [CONA, CONB] = CONST()
# RUN THROUGH LIST ELEMENTS TO SELECT COMPONENTS
    print(" COMP    TCRIT      PCRIT   OMEGA    M-FACTOR ")
    for i in range(0,N):
        l=LIST[i]-1 # note the weird index in Python
        NEW[i]=NAME[l]
# CRITICAL TEMPERATURE AND PRESSURE
        TCR[i]=T[l]
        TSQR[i]=1.0/math.sqrt(TCR[i])
        PCR[i]=P[l]*0.1013
# EOS CHARACTERISTIC CONSTANTS
        BC[i]=CONB*TCR[i]/PCR[i]
        AC[i]=CONA*TCR[i]/math.sqrt(PCR[i])
        OM=OMEGA[l]
        OMG[i]=OM
        if ICEQ==0:
            Q[i]=0.48+OMG[i]*(1.574-.176*OMG[i])
        else:
            if OMG[i]<0.49:
                Q[i]=0.37464+OMG[i]*(1.54226-0.26992*OMG[i])
            else:
                Q[i]=0.379642+OMG[i]*(1.48503-OMG[i]*(0.16444+0.01666*OMG[i]))
        print("%5s %9.2f %9.2f %8.4f %9.4f"% (NEW[i],TCR[i],PCR[i],OMG[i],Q[i]))
    print("\n  BINARY INTERACTION COEFFICIENTS \n     ")
    print("     ",end="")
    for i in range(0,N):
        print("%6s"% (NEW[i]), end='')
    print()    
    for j in range(0,N):
        l1=LIST[j]-1
        for i in range(0,N):
            l=LIST[i]-1
            CKMAT[i,j]=CKV[l1,l]
            CKMAT[j,i]=CKV[l,l1]
        if j>-1:
            print("%5s"% (NEW[j]), end='')
            for k in range(0,j):
                print("%5.2f "% (CKMAT[k,j]), end='')
            print()
    TEMSET(0.0);
    
    #print(T)
    #print(P)
    #print(OMEGA)
    #print(CKV)
    #print(NEW)
    #print(TCR)
    #print(PCR)
    #print(OMG)
    #print(CKMAT[0:N,0:N])
    


#@jit
def TEMSET(T):
# TEMSET CALCULATES THE TEMPERATURE DEPENDENT PART OF THE EOS
# PURE COMPONENT PARAMETERS, HERE SQRT (A/RT)
#
# THE RESULTS, THE VECTORS AC0,AC1,AND AC2 REPRESENT
# VALUE, 1ST DERIVATIVE AND 2ND DERIVATIVE OF SQRT (A/RT)
#
# MODIFIED: NEW AC1,AC2 ARE OLD AC1,AC2 DIVIDED BY AC0
#
    global TOLD
    global NC, TSQR, Q, AC, CKMAT, AC0, AC1, AC2, AIJ2
#
    if T==0 :
        TOLD = 0.0
        return None
    if T==TOLD :
        return None
    TOLD = T
    if T==0: # is this redundant? from Fortran
        return None
    SQT = math.sqrt(T)
    SQTR = 1.0/SQT
    T2R = 0.5/T
    T2RF = -3*T2R
    for i in range(0,NC):
        ALF = TSQR[i]
        Q1 = AC[i]*(1.0 + Q[i])*SQTR
        AC0[i] = Q1 - AC[i]*Q[i]*ALF
        AC0R = 1.0/AC0[i]
        AC1[i] = -Q1*T2R*AC0R
        AC2[i] = T2RF*AC1[i]
    for i in range (0,NC):
        FF = AC0[i]+AC0[i]
        AIJ2[i:NC,i] = FF*np.multiply(1.0-CKMAT[i:NC,i],AC0[i:NC])
        AIJ2[i,i+1:NC] = AIJ2[i+1:NC,i]
    #print(AC0[0:NC])
    #print(AC1[0:NC])
    #print(AC2[0:NC])
    #print(AIJ2[0:NC,0:NC])

#@jit    
def CUBIC(MTYP,A,B):
# CUBIC EQUATION SOLVER ROUTINE
#
# MTYP:   (I):      DESIRED PHASE TYPE 1=LIQ,-1=VAP,0=MIN. G
# A:      (I):      EOS AP/T
# B:      (I):      EOS BP/T
# Z:      (O):      RETURNED COMPRESSIBILITY FACTOR
    global CU, CU1, CU2
#
    BCN = B*CU
    X2 = 1 - BCN
    X1 = A - B*(1+B+CU+2*BCN)
    X0 = B*(A-BCN*(1+B))
    Z = X2*(1.0/3) #       PARAMETER (THIRD = 1.D0/3)
    F = ((Z-X2)*Z+X1)*Z - X0
    Z = B
    if F<0.0 or B>Z:
        Z = Z + 1;
# IF B IS SMALL, CHECK FOR THE POSSIBILITY OF A ZERO PRESSURE SOLUTION
    if B<1E-5:
        DD = X1*X1 - 4.0*X2*X0
        if DD>0.0:
            Z = B
# SET UP THIRD ORDER NEWTON FOR CUBIC EOS
    while True:
        DF2 = 3.0*Z - X2
        DF1 = Z*(DF2-X2) + X1
        DF1R = 1.0/DF1
        F = ((Z-X2)*Z+X1)*Z - X0
        DZ = F*DF1R
        DZ = DZ*(1+DZ*DF2*DF1R)
        Z = Z - DZ
        if abs(DZ)<=1E-7:
            break
# CONVERGED; HOW MANY ROOTS ARE DESIRED ?
    if MTYP*DF2>=0.0:
# FACTOR OUT FIRST ROOT AND CONSIDER REDUCED QUADRATIC
        E1 = Z - X2
        E0 = Z*E1 + X1
        D = E1*E1 - 4.0*E0
# D < 0 MEANS NO MORE ROOTS
        if D>=0.0:
# GET REMAINING
            Z1 = 0.5*(abs(E1)+math.sqrt(D));
            if E1>0.0:
                Z1 = -Z1;
            if Z>Z1:
                Z1 = E0/Z1;
# REFINE BY A SINGLE NEWTON STEP
            DF2 = 3.0*Z1 - X2;
            DF1 = Z1*(DF2-X2) + X1;
            F = ((Z1-X2)*Z1+X1)*Z1 - X0;
            DF1R = 1.0/DF1;
            DZ = F*DF1R;
            DZ = DZ*(1+DZ*DF2*DF1R);
            Z1 = Z1 - DZ;
            if Z1>=B:
                if MTYP==0:
# CALCULATE EXCESS GIBBS ENERGY DIFFERENCE FOR MULTIPLE SOLUTIONS
                    F = math.log((Z-B)/(Z1-B)) + A/(B*(CU2-CU1)) \
                        *math.log((Z+CU2*B)*(Z1+CU1*B)/(Z+CU1*B)/(Z1+CU2*B) ) \
                        - (Z-Z1);
                if F<0.0:
                    Z=Z1
            elif MTYP==1:
                if Z1<Z:
                    Z=Z1
            else:
                if Z1>Z:
                    Z=Z1
                    
    return(Z)


#@jit
def ANEW(NTEMP,X):
# MIXTURE PARAMETERS A AND B
# ANEW IS AN INTERNAL SUBROUTINE FOR TERMO
# X:      (I):      NORMALIZED COMPOSITION
# A0:     (O):      MIXTURE A (A/RT)-PARAMETER
# AT:     (O):      T-DERIVATIVE OF A0
# ATT:    (O):      T-DERIVATIVE OF AT
# B0:     (O):      MIXTURE B-PARAMETER
# AD1:    (O):      COMPSOTION DERIVATIVE OF A0
# ADT:    (O):      T-DERIVATIVE OF AD1
#
    global NC, BC, AIJ2, AC1, AC2;
# preallocate vectors only used in this function
    Y  = np.zeros(NC)
    BV = np.zeros(NC);
    AD1= np.zeros(NC);
    ADT= np.zeros(NC);

    B0 = np.dot(X[0:NC], BC[0:NC]) #       B0 = DOT_PRODUCT(X,BC)
    AD1[0:NC] = np.matmul(AIJ2[0:NC,0:NC],X[0:NC]) # MATMUL(X,AIJ2)
    A0 = 0.5*np.dot(X[0:NC],AD1[0:NC]) # 0.5D0*DOT_PRODUCT(X,AD1)
    if NTEMP>0:
        Y[0:NC] = np.multiply(AC1[0:NC],X[0:NC]) #Y = AC1*X
        BV[0:NC] = np.matmul(AIJ2[0:NC,0:NC],Y[0:NC]) # MATMUL(Y,AIJ2);
        ADT[0:NC] = BV[0:NC] + np.multiply(AC1[0:NC],AD1[0:NC])
        AT  = 0.5*np.dot(X[0:NC],ADT[0:NC]) # 0.5D0*DOT_PRODUCT(X,ADT)
        ATT = 0.0
        for i in range(0,NC):
            ATT = ATT+Y[i]*BV[i]+X[i]*AD1[i]*AC2[i]
    #print(AD1[0:NC])
    #print(ADT[0:NC])
    #print(A0)
    #print(AT)
    #print(ATT)
    #print(B0)
    return(AD1, ADT, A0, AT, ATT, B0)

#@jit
def THERMO(T, P, ZFEED, MT, ICON):
#
    global NC
#
    X = np.zeros(NC)
    X[0:NC] = ZFEED[0:NC]/sum(ZFEED[0:NC])
    (FUG, FUGT, FUGP, FUGX, AUX, FTYPE) = CUBGEN(ICON, MT, T, P, X)
    return(FUG, FUGT, FUGP, FUGX, AUX, FTYPE)


#@jit
def CUBGEN(ICON, MT, T, P, X) :
# CUBGEN(icon,T,P,X,FuG,FugT,FugP,FugX,FLU,ftyp,aux)
#
#     PARAMETERS:
#
#     MT:     (I):      PHASE TYPE DESIRED
#                       1 = LIQ, -1 = VAP, 0 = MIN G PHASE
#     IC:     (O):      INDICATOR FOR PHASE TYPE; IC RETURNS
#                       1:   IF CALLED WITH MT = 1/-1 AND PHASE IS FOUND
#                       -1:  IF CALLED WITH MT = 1/-1 AND PHASE IS NOT FOUND
#                       2:   IF CALLED WITH MT = 0 AND MIN G PHASE IS LIQUID
#                       -2:  IF CALLED WITH MT = 0 AND MIN G PHASE IS VAPOUR
#     T:      (I):      TEMPERATURE (K)
#     P:      (I):      PRESSURE (ATM)
#     Z:      (O):      CALCULATED COMPRESSIBILITY FACTOR
#     XX:     (I):      MIXTURE TOTAL COMPOSITION
#     FG:     (O):      LOG FUGACITY COEFFICIENT VECTOR
#     FT:     (O):      T-DERIVATIVE OF FG
#     FP:     (O):      P-DERIVATIVE OF FG
#     FX:     (O):      SCALED COMPOSITION DERIVATIVE OF FG
#     AUX:    (O):      VARIOUS RESIDUAL PROPERTIES
#
    global NC, CU1, CU2, BC, AIJ2
# preallocation
    PD  = np.zeros(NC)
    FUG = np.zeros(NC)
    FUGT= np.zeros(NC)
    FUGP= np.zeros(NC)
    FUGX= np.zeros((NC,NC))
    AUX = np.zeros(6)
#
    FTYPE = 0
    NDER = 0
    NTEMP = 0
    NPRES = 0
    NAUX = 1
    if ICON>=2:
        NDER=1
    if ICON==2 or ICON>3:
        NTEMP=1
        NPRES=1
    if ICON>4:
        NTEMP=2
    else:
        NDER=1
        NTEMP=1
        NPRES=1
    TEMSET(T)
# ANEW CALLS MIXTURE PARAMETERS A AND B AND THEIR APPROPRIATE DERIVATIVES
    (AD1, ADT, A, AT, ATT, B) = ANEW(NTEMP,X)
# P = 0 IS TAKEN TO BE A SPECIAL CASE WHERE THE VOLUME (OR RATHER
# V/R) IS THE INPUT AND THE CORRESPONDING P IS CALCULATED
    PSPEC = True
    if  P==0.0:
        PSPEC = False
        V = ZC # this can be wrong since matlab does not use ZC as input
        P = (1.0/(V-B) - A/((V+CU1*B)*(V+CU2*B)))*T
        ZC = P*V/T
    TREC = 1.0/T
    PREC = 1.0/P
    BREC = 1.0/B
    APT = A*P*TREC
    BPT = B*P*TREC
    if PSPEC: # SOLVE CUBIC EOS
        ZC = CUBIC(MT,APT,BPT) # SECOND Z AND ENERGY DIFF. STORED IN AUX(1),AUX(2)
    V = ZC*T*PREC; # V = VOLUME/R
    if NAUX!=0:
        AUX[1-1]=ZC
    ICX = 1
    if V>3.0*B:
        ICX = -1
# PHASE TYPE RETURN INDICATOR
    if ICX>0:
        FTYPE = +1 # liquid
    else:
        FTYPE = -1 # gas
# COEFFICIENTS FOR EOS
    S1 = 1.0/(V+CU1*B)
    S2 = 1.0/(V+CU2*B)
    P1 = P*TREC + A*S1*S2
    PA = -S1*S2
    PN = P1
    FAC = CU1*S1 + CU2*S2
    P2 = A*PA
    PB = P1*P1 - FAC*P2
    #print(V)
    #print(S1, S2, P1, PA, PN, FAC, P2, PB)    
# DERIVATIVE IS D(P/T) / D(V/R)
    DPDV = -P1*P1 - P2*(S1+S2);
    if NAUX!=0:
#c     AUX(10) = DPDV
#c     aux(8) v*dpdv
        AUX[2-1] = DPDV*T*V/P
# COEFFICIENTS FOR FUGACITY COEFFICIENT CALCULATION
    FN  = math.log(V*P1)
    XL2 = math.log(S1/S2)/(CU2-CU1)
    FA = -XL2*BREC
    F2 = -A*FA
    FF = FN - F2
    GB = -V*PA
    F2B = (A*GB-F2)*BREC
    FB = P1 - F2B
    FNB = P1
    FAB = -F2B/A
    GBB = -GB*FAC
    F2BB = (A*GBB-2.0*F2B)*BREC
    FBB = P1*P1 - F2BB
    XLZ = math.log(ZC)
    FNP = FN - XLZ
# ARE TEMPERATURE DERIVATIVES REQUIRED
    if NTEMP!=0:
        DFT = FA*AT
        HE = -T*DFT + ZC - 1.0
        HE = HE*T
        SE = -T*DFT - FF + XLZ
        if NAUX!=0:
            AUX[3-1] = HE # EXCESS ENTHALPY/R STORED IN AUX(3)
            AUX[4-1] = SE # EXCESS ENTROPY/R STORED IN AUX(4)
        PTR = PA*AT
        DPDT = P*TREC + T*PTR
        FTT = FA*ATT
        CP = -T*(T*FTT+2*DFT) - DPDT**2/DPDV - 1
# EXCESS HEAT CAPACITY/R IN AUX(5)
        DVDT = -DPDT/DPDV/T
# AUX(6) IS PRESSURE DERIVATIVE OF RESID. ENTROPY
# AUX(7) IS PRESSURE DERIVATIVE OF ENTHALPY
        if NAUX!=0:
            AUX[5-1] = CP
            AUX[6-1] = DVDT/V
# RESIDUAL CP REQUIRED
# LOG FUGACITY COEFFICIENTS
    FUG[0:NC] = FNP + FA*AD1[0:NC] + FB*BC[0:NC]
    DPDVR = 1.0/DPDV
    PD[0:NC] = (PN+PA*AD1[0:NC]+PB*BC[0:NC])*DPDVR
    if NPRES!=0:
        FUGP[0:NC] = -PD[0:NC]*TREC - PREC
# COMPOSITION DERIVATIVES REQUIRED
    if NDER!=0:
        for i in range(0,NC):
            CC = 1.0 + FNB*BC[i] + PN*PD[i];
            CA = FAB*BC[i] + PA*PD[i];
            CB = FNB + FAB*AD1[i] + FBB*BC[i] + PB*PD[i];
            FUGX[i:NC,i]=CC+CA*AD1[i:NC]+CB*BC[i:NC]+FA*AIJ2[i:NC,i];
            FUGX[i,i+1:NC]=FUGX[i+1:NC,i];
# TEMPERATURE AND PRESSURE DERIVATIVES
    if NTEMP!=0:
        CB = FAB*AT
        DPDTT = DPDT*TREC
        FUGT[0:NC] = TREC + CB*BC[0:NC] + FA*ADT[0:NC] + DPDTT*PD[0:NC]
    return(FUG, FUGT, FUGP, FUGX, AUX, FTYPE)


#@jit
def INIT(NCA,NOMA,LIST): # similar to INDATA but for 3 other models
#
# A SMALL 'DATABASE ROUTINE' FOR PROPERTIES INPUT
# NC:     (I):      NO. OF COMPONENT
# NOMA:   (I):      Model no.
# LIST:   (I):      VECTOR OF COMPONENT INDICES
# ICEQ set to 0 internally:      EOS; 0 = SRK, 1 = PRNG-ROBINSON
#
#    
    global NC, CU
    global AC, TCR, PCR, OMG, TSQR, Q, BC, CKMAT
    global NFMOD    
#    
    T = np.zeros(15)
    P = np.zeros(15)
    OMEGA = np.zeros(15)
    CKV = np.zeros((15,15))
# IDENTIFY HYDROCARBONS

# CRITICAL TEMPERATURES (K) AND PRESSURES (ATM) (REID ET AL.)
    T = np.array([190.6, 305.4, 369.8, 408.1, 425.2, \
                  460.4, 469.6, 507.4, 540.2, 568.8, \
                  594.6, 647.3, 126.2, 304.2, 373.2])
    P = np.array([45.4, 48.2, 41.9, 36.0, 37.5, \
                  33.4, 33.3, 29.3, 27.0, 24.5, \
                  22.8,218.0, 33.5, 72.8, 88.2]);
# ACENTRIC FACTORS (REID ET AL.)
    OMEGA = np.array([0.008, 0.098, 0.152, 0.176, 0.193, \
                      0.227, 0.251, 0.296, 0.351, 0.394, \
                      0.440, 0.344, 0.040, 0.225, 0.100]);
# NONZERO KIJ FOR SRK-EQUATION (REID ET AL., HEIDEMANN ET AL.)
# note the strange way of indexing and slicing in Python
    CKV[0,  11:15] = np.array([0.45, 0.02, 0.12, 0.08])
    CKV[1,  11:15] = np.array([0.45, 0.06, 0.15, 0.07])
    CKV[2,  11:15] = np.array([0.53, 0.08, 0.15, 0.07])
    CKV[3,  11:15] = np.array([0.52, 0.08, 0.15, 0.06])
    CKV[4,  11:15] = np.array([0.52, 0.08, 0.15, 0.06])
    CKV[5,  11:15] = np.array([0.50, 0.08, 0.15, 0.06])
    CKV[6,  11:15] = np.array([0.50, 0.08, 0.15, 0.06])
    CKV[7,  11:15] = np.array([0.50, 0.08, 0.15, 0.05])
    CKV[8,  11:15] = np.array([0.50, 0.08, 0.15, 0.04])
    CKV[9,  11:15] = np.array([0.50, 0.08, 0.15, 0.04])
    CKV[10, 11:15] = np.array([0.50, 0.08, 0.15, 0.03])
    CKV[11, 11:15] = np.array([0.00, 0.00, 0.00, 0.00])
    CKV[12, 11:15] = np.array([0.00, 0.00, 0.00, 0.00])
    CKV[13, 11:15] = np.array([0.00, 0.00, 0.00, 0.12])
    #
    ICEQ = 0
    NFMOD = NOMA # Model index
    #
    NC = NCA
    N = NCA
    SET_THERM_GLOBAL;
    NEW = ["NAME"]*NC
    #    
    for k in range(0,14):
        CKV[k+1:15,k]=CKV[k,k+1:15]
    CU = min(ICEQ,1)
# GET EOS CONSTANTS
    [CONA, CONB] = CONST()   
# RUN THROUGH LIST ELEMENTS TO SELECT COMPONENTS
    print(" COMP    TCRIT      PCRIT   OMEGA    M-FACTOR ")
    for i in range(0,N):
        l=LIST[i]-1 # note the weird index in Python
        NEW[i]=NAME[l]
# CRITICAL TEMPERATURE AND PRESSURE
        TCR[i]=T[l]
        TSQR[i]=1.0/math.sqrt(TCR[i])
        PCR[i]=P[l]*0.1013
# EOS CHARACTERISTIC CONSTANTS
        BC[i]=CONB*TCR[i]/PCR[i]
        AC[i]=CONA*TCR[i]/math.sqrt(PCR[i])
        OM=OMEGA[l]
        OMG[i]=OM
        if ICEQ==0:
            Q[i]=0.48+OMG[i]*(1.574-.176*OMG[i])
        else:
            if OMG[i]<0.49:
                Q[i]=0.37464+OMG[i]*(1.54226-0.26992*OMG[i])
            else:
                Q[i]=0.379642+OMG[i]*(1.48503-OMG[i]*(0.16444+0.01666*OMG[i]))
        print("%5s %9.2f %9.2f %8.4f %9.4f"% (NEW[i],TCR[i],PCR[i],OMG[i],Q[i]))       
    print("\n  BINARY INTERACTION COEFFICIENTS \n     ")
    print("     ",end="")
    for i in range(0,N):
        print("%6s "% (NEW[i]), end='')
    print()    
    for i in range(0,NC):
        for j in range(0,NC):
            if j==i:
                CKMAT[i,j] = 0.0
            else:
                CKMAT[i,j] = (j-i)*0.02
                CKMAT[j,i] = -0.7*CKMAT[i,j]
        if i>-1:
            print("%5s"% (NEW[i]), end='')
            for j in range(0,i):
                print("%5.2f "% (CKMAT[i,j]), end='')
            print()
    TEMSET(0.0);

#@jit
def FUGAC(T, P, XX):
# test thermal routine
    global NFMOD, NC, BC, CU1, CU2
    # preallocation
    AD1 = np.zeros(NC)
    FG = np.zeros(NC)
    #
    TEMSET(T)
    X = np.zeros(NC)
    X[0:NC]=XX[0:NC]/sum(XX[0:NC])
    B = np.dot(X[0:NC],BC[0:NC])
    A = 0.0
    for i in range(0,NC):
        AA1 = 0.0
        for j in range(0,NC):
            if NFMOD<=1:
                CIF = (CKMAT[i,j]+CKMAT[j,i])/2.0
            elif NFMOD==2:
                CIF = CKMAT[i,j]
            else:
                CP = 5E-3*P
                if CP>0.5:
                    CP = 0.5
                if i==j:
                    CP=0.0
                CIF = (CKMAT[i,j]+CKMAT[j,i])/2.0 + CP
            AA1 = AA1 + AC0[j]*X[j]*(1.0-CIF)
        AD1[i] = 2*AC0[i]*AA1
        A = A + X[i]*AD1[i]
    A = 0.5*A
    APT = A*P/T
    BPT = B*P/T
    Z = CUBIC(0,APT,BPT)
    ZZ = Z # for output
    V = Z*T/P
    S1 = 1.0/(V+CU1*B)
    S2 = 1.0/(V+CU2*B)
    P1 = P/T+A*S1*S2
    PA = -S1*S2
    # DERIVATIVE IS D(P/T) / D(V/R)
    XL1 = math.log(V*P1)
    FN = XL1
    XL2 = math.log(S1/S2)/(CU2-CU1)
    FA = -XL2/B
    F2 = -A*FA
    GB = -V*PA
    F2B = (A*GB-F2)/B
    FB = P1-F2B
    XLZ = math.log(Z)
    FNP = FN-XLZ
    for i in range(0,NC):
        FG[i]=FNP+FA*AD1[i]+FB*BC[i]
    return(FG, ZZ)