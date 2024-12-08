import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

#Constants
G = 6.67e-11
AU = 1.496e11
km = 1000
Years = 365.25 * 24 * 3600
m1 = m2 = 1.989e30

#Initial Positions
x1i = -0.5 * AU
x2i = 0.5 * AU
y1i = 0
y2i = 0

#Initial Velocities
vx1i = 0
vx2i = 0
vy1i = -15 *km
vy2i = 15 *km

#Initial Array
initialConditions = [x1i, x2i, y1i, y2i, vx1i, vx2i, vy1i, vy2i]

#Define functions
def calculateEnergy(state,m1,m2):
    x1, x2, y1, y2, vx1, vx2, vy1, vy2 = state

    #Kinetic Energy for each body
    k1 = 0.5 * m1 * (vx1**2 + vy1**2)
    k2 = 0.5 * m2 * (vx2**2 + vy2**2)
    K = k1+k2

    r = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    #Potential Energy for system
    U = ((-G) * m1 * m2) / r

    return K+U #Return total energy of system

def testCase(state, t, G, m1, m2):
    x1, x2, y1, y2, vx1, vx2, vy1, vy2 = state

    #Distance
    r = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    #Velocity
    dx1 = vx1
    dx2 = vx2
    dy1 = vy1
    dy2 = vy2

    #Acceleration
    dvx1 = G * (m2 / (r ** 3)) * (x2 - x1)
    dvx2 = G * (m1 / (r ** 3)) * (x1 - x2)
    dvy1 = G * (m2 / (r ** 3)) * (y2 - y1)
    dvy2 = G * (m1 / (r ** 3)) * (y1 - y2)

    return [dx1, dx2, dy1, dy2, dvx1, dvx2, dvy1, dvy2]

t = np.linspace(0, 5 * Years, 1000) #Create time array
T = t/Years
sol = odeint(testCase, initialConditions, t, args=(G, m1, m2)) #Solve ODE's

#Calculate relative energy change
E = np.array([calculateEnergy(state,m1,m2) for state in sol])
dE = abs((E - E[0])/E[0])/ 1e-6

#Extract values in terms of AU
x1 = sol[:,0]/AU
x2 = sol[:,1]/AU
y1 = sol[:,2]/AU
y2 = sol[:,3]/AU

#Plot Graphs
#Orbital paths
plt.figure(1,figsize=(10,10))
plt.ylim(-0.4,0.4)
plt.xlabel("X (AU)",size=18)
plt.ylabel("Y (AU)", size=18)
plt.title("Orbital Paths", size=18)
plt.axis('equal')
plt.plot(x1,y1,label="m1", color='black')
plt.plot(x2,y2, label="m2", color='r')
plt.legend(fontsize=18)
plt.grid()
plt.tick_params(axis='both', which='major', labelsize=14)

#X displacement from origin
plt.figure(2,figsize=(12,9))
plt.title("Time Evolution of X-Position", size = 18)
plt.xlabel("Time (Years)",size=18)
plt.ylabel("X (AU)",size=18)
plt.plot(T,x1, color='black', label="m1")
plt.plot(T,x2, color='r', label="m2")
plt.legend(fontsize=18, loc="upper right")
plt.grid()
plt.tick_params(axis='both', which='major', labelsize=14)

#Relative Energy Change
plt.figure(3,(12,9))
plt.title("Relative Energy Change", size = 18)
plt.xlabel("Time (Years)", size=18)
plt.ylabel(r"$\Delta E / E_0$  ($1 \times 10^{-6}$)", size=18)
plt.plot(T,dE)
plt.grid()
plt.tick_params(axis='both', labelsize=16)
plt.show()
