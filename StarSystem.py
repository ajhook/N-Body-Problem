import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint

#Define Constants
G = 6.67e-11
AU = 1.4959787e11
km = 1000
M0 = 1.988e30
ME = 5.97226e24

Ma = 0.146 * M0
Mb = 5.6 * ME
Mc = 1.91 * ME

#Orbital Periods
Tb = 24.7 * 24 * 3600
Tc = 3.8 * 24 * 3600

#Orbital Radius
rb = 0.0875 * AU
rc = 0.02507 * AU

#Positions
xAi, yAi = 0,0
xBi, yBi = -rb, 0
xCi, yCi = 0, rc
xEi, yEi = 0, -0.075*AU

#Velocities
vxAi, vyAi = 0, 0
vxBi, vyBi = 0, 2* np.pi * (rb/Tb)
vxCi, vyCi = 2* np.pi * (rc/Tc), 0
vxEi, vyEi = 41541.5, 0

initialConditions = [xAi, yAi, xBi, yBi, xCi, yCi, xEi, yEi, vxAi, vyAi, vxBi, vyBi, vxCi, vyCi, vxEi, vyEi]

def calculateEnergy(state, Ma, Mb, Mc, ME):
    xA, yA, xB, yB, xC, yC, xE, yE, vxA, vyA, vxB, vyB, vxC, vyC, vxE, vyE = state

    k1 = 0.5 * Ma * (vxA ** 2 + vyA ** 2)
    k2 = 0.5 * Mb * (vxB ** 2 + vyB ** 2)
    k3 = 0.5 * Mc * (vxC ** 2 + vyC ** 2)
    k4 = 0.5 * ME * (vxE ** 2 + vyE ** 2)
    K = k1 + k2 + k3 + k4

    r_AB = np.sqrt((xB - xA) ** 2 + (yB - yA) ** 2)
    r_AC = np.sqrt((xC - xA) ** 2 + (yC - yA) ** 2)
    r_AE = np.sqrt((xE - xA) ** 2 + (yE - yA) ** 2)
    r_BC = np.sqrt((xC - xB) ** 2 + (yC - yB) ** 2)
    r_BE = np.sqrt((xE - xB) ** 2 + (yE - yB) ** 2)
    r_CE = np.sqrt((xE - xC) ** 2 + (yE - yC) ** 2)

    U = -G*(((Ma * Mb) / r_AB) + ((Mb*Mc)/r_BC) + ((Ma*Mc)/r_AC) + ((Ma*ME)/r_AE) + ((Mb*ME)/r_BE) + ((Mb*ME)/r_CE))

    return K + U

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

t = np.linspace(0, 5*Tb, 1000)
T = t/(24*3600)
sol = odeint(fourBodySystem, initialConditions, t, args=(G, Ma, Mb, Mc, ME))

E = np.array([calculateEnergy(state,Ma,Mb,Mc,ME) for state in sol])
dE = abs((E - E[0])/E[0])


xA = sol[:, 0]/AU
yA = sol[:, 1]/AU
xB = sol[:, 2]/AU
yB = sol[:, 3]/AU
xC = sol[:, 4]/AU
yC = sol[:, 5]/AU
xE = sol[:, 6]/AU
yE = sol[:, 7]/AU

r_AB = np.sqrt((xB - xA) ** 2 + (yB - yA) ** 2)
r_AE = np.sqrt((xE - xA) ** 2 + (yE - yA) ** 2)
r_AC = np.sqrt((xC - xA) ** 2 + (yC - yA) ** 2)

plt.figure(figsize=(10, 10))
plt.plot(0,0, 'o', label=f'LHS-1140-A', color='orange')
plt.plot(xB, yB, label=f'LHS-1140-B', color='r')
plt.plot(xC, yC, label=f'LHS-1140-C', color='g')
plt.plot(xE, yE, label=f'Earth', color ='b')
plt.xlim(-0.1,0.1)
plt.ylim(-0.1,0.1)
plt.title("Orbital Paths of the LHS-1140 Star System", size=18)
plt.xlabel("X(AU)", size=18)
plt.ylabel("Y(AU)", size=18)
plt.legend()
plt.grid()

plt.figure(2)
plt.plot(T, dE, label="Relative Energy Deviation")
plt.xlabel("Time (days)")
plt.ylabel(r"$\Delta E / E_0$")
plt.title("Energy Conservation")
plt.xlim(0,125)
plt.grid()

plt.figure(3)
plt.plot(t,r_AB,color='r', label='LHS-1140-B' )
plt.plot(t,r_AE,color='b', label='Earth')
plt.plot(t,r_AC,color='g', label='LHS-1140-C')
plt.ylim(0,0.12)
plt.xlim(0,6)
plt.fill_between(t,0.0616,0.0943,alpha=0.2,color='g')
plt.grid()
plt.legend()

plt.show()