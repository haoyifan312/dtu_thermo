# Wei Yan, DTU, weya@kemi.dtu.dk
# August 10, 2020
# Test script for thermclc_shared.pyc
import time
import numpy as np
import thermclc as th

LIST = np.array([1, 2, 3, 5, 7, 8, 13]).astype(int)
NC = np.size(LIST)
ICEQ = 1
z = np.array([94.3, 2.7, 0.74, 0.49, 0.27, 0.10, 1.40])
z = z/sum(z)

th.INDATA(NC,ICEQ,LIST)

Tc = np.zeros(NC) #Critical temperatures [K]
Pc = np.zeros(NC) #Critical pressures [MPa]
omega = np.zeros(NC) #Acentric factors
kij = np.zeros((NC,NC)) #Binary interaction parameters

for i in range(0,NC): # writing the critical parameters in vectors.
    (Tc[i], Pc[i], omega[i]) = th.GETCRIT(i) # this is to get critical properties
    for j in range(i+1,NC):
        kij[i,j] = th.GETKIJ(i,j) # this is to get kij
        kij[j,i] = kij[i,j]

T = float(input("Enter the temperature in K ")) # K
P = float(input('Enter the Pressure in MPa ')) # MPa
        
(FUG, FUGT, FUGP, FUGX, AUX, FTYPE) = th.THERMO(T,P,z,0,5) # this is to call the THERMO subroutine to get all the properties

print("FUG")
print(FUG)

print("FUGT")
print(FUGT)

print("FUGP")
print(FUGP)

print("FUGX")
print(FUGX)

print("AUX")
print(AUX)

print("FTYPE")
print(FTYPE)
