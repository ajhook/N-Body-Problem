#Import Libraries
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

#Constants (dimensionless units)
G = 1
m1,m2,m3 = 3, 4, 5


#Initial Conditions
#Postions
x1i, y1i = 1, 3
x2i, y2i = -2, -1
x3i, y3i = 1,-1


#Velocities
vx1i, vy1i = 0, 0
vx2i, vy2i = 0, 0
vx3i, vy3i = 0, 0

initialConditions = [x1i, y1i, x2i, y2i, x3i, y3i, vx1i, vy1i, vx2i, vy2i, vx3i, vy3i]

#Define Functions
def calculateEnergy(state, m1, m2, m3):
    x1, y1, x2, y2, x3, y3, vx1, vy1, vx2, vy2, vx3, vy3 = state

    k1 = 0.5 * m1 * (vx1 ** 2 + vy1 ** 2)
    k2 = 0.5 * m2 * (vx2 ** 2 + vy2 ** 2)
    k3 = 0.5 * m3 * (vx3 ** 2 + vy3 ** 2)
    K = k1 + k2 + k3

    r_12 = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    r_13 = np.sqrt((x3 - x1) ** 2 + (y3 - y1) ** 2)
    r_23 = np.sqrt((x3 - x2) ** 2 + (y3 - y2) ** 2)

    U = -G*(((m1 * m2) / r_12) + ((m2*m3)/ r_23) + ((m1*m3)/r_13))

    return K + U
def threeBodySystem(state, t, G, m1, m2, m3):
    x1, y1, x2, y2, x3, y3, vx1, vy1, vx2, vy2, vx3, vy3 = state


    #Distances
    r_12 = np.sqrt((x2-x1)**2 + (y2-y1)**2)
    r_13 = np.sqrt((x3-x1)**2 + (y3-y1)**2)
    r_23 = np.sqrt((x3-x2)**2 + (y3-y2)**2)

    #Velocities
    dx1, dy1 = vx1, vy1
    dx2, dy2 = vx2, vy2
    dx3, dy3 = vx3, vy3

    #Accelerations
    dvx1 = (G * (m2/r_12**3) * (x2 - x1)) + (G * (m3/r_13**3) * (x3 - x1))
    dvy1 = (G * (m2/r_12**3) * (y2 - y1)) + (G * (m3/r_13**3) * (y3 - y1))

    dvx2 = (G * (m1/r_12**3) * (x1 - x2)) + (G * (m3/r_23**3) * (x3 - x2))
    dvy2 = (G * (m1/r_12**3) * (y1 - y2)) + (G * (m3/r_23**3) * (y3 - y2))

    dvx3 = (G * (m1/r_13**3) * (x1 - x3)) + (G * (m2/r_23**3) * (x2-x3))
    dvy3 = (G * (m1/r_13**3) * (y1 - y3)) + (G * (m2/r_23**3) * (y2-y3))

    return [dx1, dy1, dx2, dy2, dx3, dy3, dvx1, dvy1, dvx2, dvy2, dvx3, dvy3]

time = np.linspace(0, 10, 10000)

sol = odeint(threeBodySystem, initialConditions, time, args = (G, m1, m2, m3))

E = np.array([calculateEnergy(state,m1,m2,m3) for state in sol])
dE = abs((E - E[0])/E[0])

x1 = sol[:, 0]
y1 = sol[:, 1]
x2 = sol[:, 2]
y2 = sol[:, 3]
x3 = sol[:, 4]
y3 = sol[:, 5]

plt.figure(figsize=(10, 10))
plt.plot(x1, y1, ':', label=f'm1={m1}', alpha=0.7)  # dotted line for m1
plt.plot(x2, y2, '--', label=f'm2={m2}', alpha=0.7)  # dashed line for m2
plt.plot(x3, y3, '-', label=f'm3={m3}', alpha=0.7)  # solid line for m3

# Plot initial positions
plt.figure(1)
plt.title("Burrau's Problem",size=20)
plt.xlabel("x",size=20)
plt.ylabel("y",size=20)
plt.plot([x1[0]], [y1[0]], 'ko')
plt.plot([x2[0]], [y2[0]], 'ko')
plt.plot([x3[0]], [y3[0]], 'ko')
plt.legend()
plt.grid()

plt.figure(figsize=(10,7))
plt.plot(time, dE, label="Relative Energy Deviation")
plt.xlabel("Time",size=20)
plt.ylabel(r"$\Delta E / E_0$", size=20)
plt.title("Energy Conservation",size=20)
plt.grid()
plt.show()
