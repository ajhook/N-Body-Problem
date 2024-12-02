#Import Libraries
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

#Constants
G = 6.67e-11
AU = 1.496e11
km = 1000
pc = 3.0857e16
M0 = 1.989e30

#Masses
m1 = 3 * M0
m2 = 4 * M0
m3 = 5 * M0


#Initial Conditions
#Postions
x1i, y1i = 1*pc, 3*pc
x2i, y2i = -2*pc, -1*pc
x3i, y3i = 1*pc,-1*pc


#Velocities
vx1i, vy1i = 0, 0
vx2i, vy2i = 0, 0
vx3i, vy3i = 0, 0

initialConditions = [x1i, y1i, x2i, y2i, x3i, y3i, vx1i, vy1i, vx2i, vy2i, vx3i, vy3i]

#Define Function
def threeBodySystem(state, G, t, m1, m2, m3):
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

time = np.linspace(0,2*365.25*24*60*60, 100000)

solution = odeint(threeBodySystem, initialConditions, time, args = (G, m1, m2, m3))

x1 = solution[:, 0] / AU
y1 = solution[:, 1] / AU
x2 = solution[:, 2] / AU
y2 = solution[:, 3] / AU
x3 = solution[:, 4] / AU
y3 = solution[:, 5] / AU

plt.figure(3)
plt.plot(x1, y1, label="M1")
plt.plot(x2, y2, label="M2")
plt.plot(x3, y3, label="M3")
plt.legend()
plt.grid()
plt.show()