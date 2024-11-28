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

    r = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

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

t = np.linspace(0, 5 * 365.25 * 24 * 60 * 60, 100) #Create time array
T = t/365*24*60*60
sol = odeint(testCase, initial, t, args=(G, m1, m2)) #Solve ODE's

E = np.array([calculateEnergy(state,m1,m2) for state in sol])
dE = (E - E[0])/abs(E[0])

#Extract values
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

#%%
# Constants scientific study
M0 = 1.989e30
MA = 2.1 * M0
MB = M0
ME = 5.972e24

# Initial Conditions
#Postions
xAi,yAi = -5*AU,0
xBi, yBi = 20 * AU, 0
xEi, yEi = -50 * AU, 20 * AU


#Velocities
vxAi, vyAi = 200*km,500*km
vxBi, vyBi = -200*km, 550 *km
vxEi, vyEi = -500*km, -500*km

initialConditions = [xAi, yAi, xBi, yBi, xEi, yEi, vxAi, vyAi, vxBi, vyBi, vxEi, vyEi]

#Define Function
def threeBodySystem(state, G, t, MA, MB, ME):
    xA, yA, xB, yB, xE, yE, vxA, vyA, vxB, vyB, vxE, vyE = state

    #Distances
    r_AB = np.sqrt((xB-xA)**2 + (yB-yA)**2)
    r_AE = np.sqrt((xE-xA)**2 + (yE-yA)**2)
    r_BE = np.sqrt((xE-xB)**2 + (yE-yB)**2)

    #Velocities
    dxA, dyA = vxA, vyA
    dxB, dyB = vxB, vyB
    dxE, dyE = vxE, vyE

    #Accelerations
    dvxA = (G * (MB/r_AB**3) * (xB - xA)) + (G * (ME/r_AE**3) * (xE - xA))
    dvyA = (G * (MB/r_AB**3) * (yB - yA)) + (G * (ME/r_AE**3) * (yE - yA))

    dvxB = (G * (MA/r_AB**3) * (xA - xB)) + (G * (ME/r_BE**3) * (xE - xB))
    dvyB = (G * (MA/r_AB**3) * (yA - yB)) + (G * (ME/r_BE**3) * (yE - yB))

    dvxE = (G * (MA/r_AE**3) * (xA - xE)) + (G * (MB/r_BE**3) * (xB-xE))
    dvyE = (G * (MA/r_AE**3) * (yA - yE)) + (G * (MB/r_BE**3) * (yB-yE))

    return [dxA, dxB, dxE, dyA, dyB, dyE, dvxA, dvyA, dvxB, dvyB, dvxE, dvyE]

time = np.linspace(0,200*365.25*24*60*60, 100000)

solution = odeint(threeBodySystem, initialConditions, t, args = (G, MA, MB, ME))

xA = solution[:, 0] / AU
yA = solution[:, 1] / AU
xB = solution[:, 2] / AU
yB = solution[:, 3] / AU
xE = solution[:, 4] / AU
yE = solution[:, 5] / AU

plt.figure(3)
plt.plot(xA, yA, label="Sirius A")
plt.plot(xB, yB, label="Sirius B")
plt.plot(xE, yE, label="Earth")
plt.legend()
plt.grid()
plt.show()