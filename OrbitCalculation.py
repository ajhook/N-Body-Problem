import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint

G = 6.67e-11
AU = 1.4959787e11
km = 1000
M0 = 1.988e30
ME = 5.97226e24

Ma = 0.146 * M0
Mb = 5.6 * ME
Mc = 1.272 * ME

#Orbital Periods
Tb = 24.7 * 24 * 3600
Tc = 3.8 * 24 * 3600

#Orbital Radius
rb = 0.0875 * AU
rc = 0.02507 * AU
re = 0.075 * AU
#Positions
xAi, yAi = 0,0
xBi, yBi = -rb, 0
xCi, yCi = 0, rc
xEi, yEi = 0, -0.07*AU

#Velocities
vxAi, vyAi = 0, 0
vxBi, vyBi = 0, 2* np.pi * (rb/Tb)
vxCi, vyCi = 2* np.pi * (rc/Tc), 0


threeBodyConditions = [xAi, yAi, xBi, yBi, xCi, yCi, vxAi, vyAi, vxBi, vyBi, vxCi, vyCi]
fourBodyConditions = [xAi, yAi, xBi, yBi, xCi, yCi, xEi, yEi, vxAi, vyAi, vxBi, vyBi, vxCi, vyCi, vxEi, vyEi]

def threeBodySystem(state, t, G, Ma, Mb, Mc):
    xA, yA, xB, yB, xC, yC, vxA, vyA, vxB, vyB, vxC, vyC = state


    #Distances
    r_AB = np.sqrt((xB-xA)**2 + (yB-yA)**2)
    r_AC = np.sqrt((xC-xA)**2 + (yC-yA)**2)
    r_BC = np.sqrt((xC-xB)**2 + (yC-yB)**2)

    #Velocities
    dxA, dyA = vxA, vyA
    dxB, dyB = vxB, vyB
    dxC, dyC = vxC, vyC

    #Accelerations
    dvxA = (G * (Mb/r_AB**3) * (xB - xA)) + (G * (Mc/r_AC**3) * (xC - xA))
    dvyA = (G * (Mb/r_AB**3) * (yB - yA)) + (G * (Mc/r_AC**3) * (yC - yA))

    dvxB = (G * (Ma/r_AB**3) * (xA - xB)) + (G * (Mc/r_BC**3) * (xC - xB))
    dvyB = (G * (Ma/r_AB**3) * (yA - yB)) + (G * (Mc/r_BC**3) * (yC - yB))

    dvxC = (G * (Ma/r_AC**3) * (xA - xC)) + (G * (Mb/r_BC**3) * (xB-xC))
    dvyC = (G * (Ma/r_AC**3) * (yA - yC)) + (G * (Mb/r_BC**3) * (yB-yC))

    return [dxA, dyA, dxB, dyB, dxC, dyC, dvxA, dvyA, dvxB, dvyB, dvxC, dvyC]
def fourBodySystem(state, t, G, Ma, Mb, Mc, ME,):
    xA, yA, xB, yB, xC, yC, xE, yE, vxA, vyA, vxB, vyB, vxC, vyC, vxE, vyE = state

    # Distances
    r_AB = np.sqrt((xB - xA) ** 2 + (yB - yA) ** 2)
    r_AC = np.sqrt((xC - xA) ** 2 + (yC - yA) ** 2)
    r_AE = np.sqrt((xE - xA) ** 2 + (yE - yA) ** 2)
    r_BC = np.sqrt((xC - xB) ** 2 + (yC - yB) ** 2)
    r_BE = np.sqrt((xE - xB) ** 2 + (yE - yB) ** 2)
    r_CE = np.sqrt((xE - xC) ** 2 + (yE - yC) ** 2)

    # Velocities
    dxA, dyA = vxA, vyA
    dxB, dyB = vxB, vyB
    dxC, dyC = vxC, vyC
    dxE, dyE = vxE, vyE

    # Accelerations
    dvxA = (G * (Mb / r_AB ** 3) * (xB - xA)) + (G * (Mc / r_AC ** 3) * (xC - xA)) + (G * (ME / r_AE ** 3) * (xE - xA))
    dvyA = (G * (Mb / r_AB ** 3) * (yB - yA)) + (G * (Mc / r_AC ** 3) * (yC - yA)) + (G * (ME / r_AE ** 3) * (yE - yA))

    dvxB = (G * (Ma / r_AB ** 3) * (xA - xB)) + (G * (Mc / r_BC ** 3) * (xC - xB)) + (G * (ME / r_BE ** 3) * (xE - xB))
    dvyB = (G * (Ma / r_AB ** 3) * (yA - yB)) + (G * (Mc / r_BC ** 3) * (yC - yB)) + (G * (ME / r_BE ** 3) * (yE - yB))

    dvxC = (G * (Ma / r_AC ** 3) * (xA - xC)) + (G * (Mb / r_BC ** 3) * (xB - xC)) + (G * (ME / r_CE ** 3) * (xE - xC))
    dvyC = (G * (Ma / r_AC ** 3) * (yA - yC)) + (G * (Mb / r_BC ** 3) * (yB - yC)) + (G * (ME / r_CE ** 3) * (yE - yC))

    dvxE = (G * (Ma / r_AE ** 3) * (xA - xE)) + (G * (Mb / r_BE ** 3) * (xB - xE)) + (G * (Mc / r_CE ** 3) * (xC - xE))
    dvyE = (G * (Ma / r_AE ** 3) * (yA - yE)) + (G * (Mb / r_BE ** 3) * (yB - yE)) + (G * (Mc / r_CE ** 3) * (yC - yE))

    return [dxA, dyA, dxB, dyB, dxC, dyC, dxE, dyE, dvxA, dvyA, dvxB, dvyB, dvxC, dvyC, dvxE, dvyE]

#Calculating the eccentricity of an orbit given rc
def EccentricityR(initial_Conditions, rc, t, G, Ma, Mb, Mc ):

    xCi, yCi = 0, rc
    xvCi, vyCi = -2*np.pi*(rc/Tc), 0
    state = (
            initial_Conditions[0:4] # xA, xB, yA, yB
            + [xCi] + [yCi]
            + initial_Conditions[6:10] # vxA, vyA, vxB, vyB
            + [xvCi] + [vyCi]
    )

    solution = odeint(threeBodySystem, state, t, args=(G, Ma, Mb, Mc))

    #extract variable
    xC,yC = solution[:,4], solution[:,5]
    distance = np.sqrt(xC**2 + yC**2)

    rmax, rmin = np.max(distance), np.min(distance)
    eccentricityR = (rmax-rmin)/(rmax + rmin)
    print(f'Radius: {rc/AU}, ' f' Eccentricity: {eccentricityR}')

    return eccentricityR

def EccentricityV(initial_Conditions, ve, t, G, Ma, Mb, Mc, ME ):

    xEi, yEi = -re, 0
    xvEi, vyEi = 0, -ve
    state = (
            initial_Conditions[0:6] # xA, xB, yA, yB
            + [xEi] + [yEi]
            + initial_Conditions[8:12] # vxA, vyA, vxB, vyB
            + [xvEi] + [vyEi]
    )

    solution = odeint(fourBodySystem, state, t, args=(G, Ma, Mb, Mc, ME))

    #extract variable
    xE,yE = solution[:,6], solution[:,7]
    distance = np.sqrt(xE**2 + yE**2)

    rmax, rmin = np.max(distance), np.min(distance)
    eccentricityV = (rmax-rmin)/(rmax + rmin)
    print(f'Radius: {rc/AU}, ' f' Eccentricity: {eccentricityV}')

    return eccentricityV

#Create time array for one orbit of LHS-1140-c around its star
t = np.linspace(0, Tc, 1000)

#Solve ODE's for each system
fourBodySol = odeint(fourBodySystem, fourBodyConditions, t, args=(G, Ma, Mb, Mc, ME))
threeBodySol = odeint(threeBodySystem, threeBodyConditions, t, args=(G, Ma, Mb, Mc))

rvalues = np.linspace(0.0028*AU, 0.06*AU, 1000)
eccentricitiesR = []

for r in rvalues:
    eR= EccentricityR(threeBodyConditions,r,t,G,Ma,Mb,Mc)
    eccentricitiesR.append(eR)

r = rvalues[np.argmin(eccentricitiesR)]
print(f'Radius: {r/AU}, ' f'Min Eccentricity: {np.min(eccentricitiesR)}')

plt.figure(figsize=(10, 6))
plt.plot(rvalues / AU, eccentricitiesR, label="Eccentricity vs Radius")
plt.xlabel("Orbital Radius (AU)")
plt.ylabel("Eccentricity")
plt.title("Eccentricity vs Orbital Radius")
plt.grid(True)
plt.legend()
plt.show()