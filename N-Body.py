#Import Libraries
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

#Constants
G = 6.67e-11
AU = 1.496e11
km = 1000
x1i = -0.5 * AU
x2i = 0.5 * AU
y1i = 0
y2i = 0
vx1i = 0
vx2i = 0
vy1i = -15 *km
vy2i = 15 *km


m2 = 1.989e30
m1 = m2

#Define functions
def calculateEnergy(state,m1,m2):
    x1, x2, y1, y2, vx1, vx2, vy1, vy2 = state

    k1 = 0.5 * m1 * (vx1**2 + vy1**2)
    k2 = 0.5 * m2 * (vx2**2 + vy2**2)
    K = k1+k2

    r = np.sqrt((x2i - x1i) ** 2 + (y2i - y1i) ** 2)

    U = ((-G) * m1 * m2) / r


    return K+U

def testCase(state, t, G, m1, m2):
    x1, x2, y1, y2, vx1, vx2, vy1, vy2 = state

    r = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    dx1 = vx1
    dx2 = vx2
    dy1 = vy1
    dy2 = vy2

    dvx1 = G * (m2 / (r ** 3)) * (x2 - x1)
    dvx2 = G * (m1 / (r ** 3)) * (x1 - x2)
    dvy1 = G * (m2 / (r ** 3)) * (y2 - y1)
    dvy2 = G * (m1 / (r ** 3)) * (y1 - y2)

    return [dx1, dx2, dy1, dy2, dvx1, dvx2, dvy1, dvy2]



initial = [x1i, x2i, y1i, y2i, vx1i, vx2i, vy1i, vy2i]

t = np.linspace(0, 5 * 365.25 * 24 * 60 * 60, 1000) #Create time array
T = t/365*24*60*60
sol = odeint(testCase, initial, t, args=(G, m1, m2)) #Solve ODE's

E = np.array([calculateEnergy(state,m1,m2) for state in sol])
dE = (E - E[0])/E[0]

#Extract values on Astronomical Units
x1 = sol[:,0]/AU
x2 = sol[:,1]/AU
y1 = sol[:,2]/AU
y2 = sol[:,3]/AU

#Plot Graphs
plt.figure(1,figsize=(10,10))
plt.axis('equal')
plt.plot(x1,y1)
plt.plot(x2,y2)
plt.figure(2,figsize=(12,9))
plt.plot(T,x1)
plt.plot(T,x2)
plt.figure(3)
plt.plot(T,dE)
plt.show()

