import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint

G = 6.67e-11
AU = 1.496e11
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
rb = 0.0946 * AU
rc = 0.02734 * AU

#Habitable Zone
innerHZ = 0.06*AU
outerHZ = 0.087*AU

#Positions
xAi, yAi = 0,0
xBi, yBi = rb, 0
xCi, yCi = -rc, 0
xEi, yEi = 50000*AU, 0

#Velocities
vxAi, vyAi = 0, 0
vxBi, vyBi = 0, 2* np.pi * (rb/Tb)
vxCi, vyCi = 0, -2* np.pi * (rc/Tc)
vxEi, vyEi = 0,0

initialConditions = [xAi, yAi, xBi, yBi, xCi, yCi, xEi, yEi, vxAi, vyAi, vxBi, vyBi, vxCi, vyCi, vxEi, vyEi]

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

t = np.linspace(0, 10*Tb, 1000000)
sol = odeint(fourBodySystem, initialConditions, t, args=(G, Ma, Mb, Mc, ME))

xA = sol[:, 0]/AU
yA = sol[:, 1]/AU
xB = sol[:, 2]/AU
yB = sol[:, 3]/AU
xC = sol[:, 4]/AU
yC = sol[:, 5]/AU
xE = sol[:, 6]/AU
yE = sol[:, 7]/AU

plt.figure(figsize=(10, 10))
plt.plot(xA, yA, label=f'LHS-1140-A', color='r')
plt.plot(xB, yB, label=f'LHS-1140-B', color='b')
plt.plot(xC, yC, label=f'LHS-1140-C', color='g')
plt.plot(xE, yE, label=f'Earth', color ='y')
plt.xlim(-0.2,0.2)
plt.ylim(-0.2,0.2)
plt.xlabel("X(AU)")
plt.ylabel("Y(AU)")
plt.legend()
plt.grid()
plt.show()