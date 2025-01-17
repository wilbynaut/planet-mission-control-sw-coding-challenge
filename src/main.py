
# Standard Library imports
import argparse
import json
import math
# from numba import njit
import numpy as np
import scipy

# Other imports
from astropy.time import Time, TimeDelta

# Project imports
import constants

def Defaults():
 return {
    "forceModel": {
        "drag": {
            "densityRef": 3.206e-4,
            "heightRef": 60e3,
            "scaleHeight": 7.714e3
        }
    },
    "propagation": {
        "dt": 1.0,
        "finalAltitude": 100,
        "radiusEarth": 6.3781363e3 
    }
}

def wrapTo2Pi(angle):
    """Keeps an angle within 2 Pi radians"""
    twoPi = 2 * math.pi

    newAngle = angle

    if (angle > twoPi):
        newAngle = angle % twoPi
    elif (angle < 0):
        newAngle = angle % (-twoPi)

    return newAngle

class PropagationControls():
    def __init__(self, prop):
        defaultProp = Defaults()['propagation']

        if 'dt' in prop:
            self.dt = TimeDelta(prop['dt'], format='sec')
        else:
            self.dt = TimeDelta(defaultProp['dt'], format='sec')

        if 'finalAltitude' in prop:
            self.finalAltitude = prop['finalAltitude']
        else:
            self.finalAltitude = defaultProp['finalAltitude']

        if 'radiusEarth' in prop:
            self.radiusEarth = prop['radiusEarth']
        else:
            self.radiusEarth = defaultProp['radiusEarth']
        
        for key in prop:
            if key not in defaultProp:
                print(f'Warning: key "{key}" is not an accepted input to "propagation" input')

class ForceModel():
    def __init__(self, force):
        self.drag = DragModel(force['drag'])

class DragModel():
    def __init__(self, drag):
        defaultDrag = Defaults()['forceModel']['drag']

        if 'densityRef' in drag:
            self.densityRef = (drag['densityRef'])
        else:
            self.densityRef = (defaultDrag['densityRef'])
        self.densityRef * 1e9   # Convert from mkg/m^3 to kg/km^3
        
        if 'heightRef' in drag:
            self.heightRef = (drag['heightRef'])
        else:
            self.heightRef = (defaultDrag['heightRef'])
        self.heightRef * 1e-3   # Convert from m to km
        
        if 'scaleHeight' in drag:
            self.scaleHeight = (drag['scaleHeight'])
        else:
            self.scaleHeight = (defaultDrag['scaleHeight'])
        self.scaleHeight * 1e-3   # Convert from m to km

        for key in drag:
            if key not in defaultDrag:
                print(f'Warning: key "{key}" is not an accepted input to "drag" input')
    
    def accel(self, velocity, alt, dragCoeff, dragArea, mass):
        """Method to compute acceleration due to drag"""
        rho = self.density(alt)

        a = -0.5 * rho * (dragCoeff * dragArea / mass) * velocity**2
        return a
    
    def density(self, alt):
        """Method to compute the local atmospheric density [kg / km^3]"""
        rho = self.densityRef * np.exp(-(alt - self.heightRef) / self.scaleHeight)
        return rho

class Classical():
    """State defined with classical orbital elements"""
    def __init__(self, initState):
        self.a = initState['sma']
        self.ecc = initState['ecc']
        self.inc = math.radians(initState['inc'])
        self.raan = math.radians(initState['raan'])
        self.argp = math.radians(initState['argp'])
        self.nu = math.radians(initState['trueAnom'])
    
    def semiLatRect(self):
        """Calculates the semi-latus rectum of the orbit"""
        p = self.a * (1 - self.ecc**2)
        return p
    
    def radius(self):
        """Calculates the current orbital radius"""
        r = self.semiLatRect() / (1 + self.ecc * math.cos(self.nu))
        return r
    
    def velocity(self):
        """Calculates velocity Based on energy: V^2/2 -mu/r = -mu/(2*a)"""
        v = math.sqrt(2 * (constants.MU / self.radius() - constants.MU / (2*self.a)))
        return v
    
    def altitude(self):
        """Calculates the orbit's altitude above the Earth's surface"""
        alt = self.radius() - constants.RAD_E
        return alt
    
    def angMomentum(self):
        """Calculates the magnitude of the orbit's angular momentum"""
        h = math.sqrt(constants.MU * self.semiLatRect())
        return h
    
    def stateVector(self):
        return [self.a, self.ecc, self.inc, self.raan, self.argp, self.nu]
    
    def setState(self, x):
        self.a = x[0]
        self.ecc = x[1]
        self.inc = wrapTo2Pi(x[2])
        self.raan = wrapTo2Pi(x[3])
        self.argp = wrapTo2Pi(x[4])
        self.nu = wrapTo2Pi(x[5])

class Thruster():
    """Class to define a thruster"""
    def __init__(self, sc):
        self.thrust = sc['thrust']
        self.Isp = sc['Isp']
        self.massFlowRate = self.thrust / (self.Isp * constants.G0)

class Spacecraft():
    """Class with all information relative to a spacecraft and its current state"""
    def __init__(self, sc, scNum):
        if 'satName' in sc:
            self.name = sc['satName']
        else:
            self.name = "sat" + str(scNum)
        self.classical = Classical(sc['initialState'])
        self.epoch = Time(sc['initialState']['epoch'], format='isot', scale='utc')
        self.dragArea = sc['dragArea'] * 1e-6       # Conver m^2 to km^2
        self.dragCoeff = sc['dragCoeff']
        self.massDry = sc['massDry']
        self.massProp = sc['massProp']
        self.thruster = Thruster(sc)

        self.fuelDepletedFlag = False
        self.fuelDepletedEpoch = self.epoch
    
    def totalMass(self):
        return self.massDry + self.massProp
    
    def setEpoch(self, epoch):
        self.epoch = epoch
    
    def setMassProp(self, mass):
        self.massProp = mass
    
    def checkIfFuelDepleted(self):
        if self.massProp <= 0.0:
            self.fuelDepletedFlag = True
            self.fuelDepletedEpoch = self.epoch
    
class Propagator():
    """A class to propagate a spacecraft with force model and propagation controls"""
    def __init__(self, force, propCtrls, spacecraft, burnStart):
        self.force = force
        self.propCtrls = propCtrls
        self.spacecraft = spacecraft
        self.burnStart = burnStart

    def dragAccel(self):
        """Calculates tangential acceleration due to drag [km/s^2]"""
        v = self.spacecraft.classical.velocity()
        mass = self.spacecraft.totalMass()
        aDrag = self.force.drag.accel(v, self.spacecraft.classical.altitude(), self.spacecraft.dragCoeff, self.spacecraft.dragArea, mass)

        return aDrag
    
    def thrustAccel(self):
        """Calculates acceleration due to thrust [km/s^2]"""
        if (self.spacecraft.epoch >= self.burnStart) and (self.spacecraft.massProp > 0.0):
            # If we are passed the burn start epoch and have remaining fuel, apply thrust, convert from [m/s^2] to [km/s^2]
            aThrust = -(self.spacecraft.thruster.thrust / (self.spacecraft.massDry + self.spacecraft.massProp)) * 1e-3
            dmProp = -self.spacecraft.thruster.massFlowRate
        else:
             # If the burn has not started, or we are out of fuel, do not apply thrust
            aThrust = 0.0
            dmProp = 0.0
        
        return aThrust, dmProp
    
    def GaussPlanEq(self, t, state):
        # Extract state values
        a = state[0]
        e = state[1]
        i = state[2]
        nu = state[5]

        # Precompute reused values
        aR = 0.0    # Radial perturbing acceleration
        aN = 0.0    # Normal perturbing acceleration
        aTangDrag = self.dragAccel()
        aTangThrust, dmPropdt = self.thrustAccel()
        aTang = aTangDrag + aTangThrust
        p = self.spacecraft.classical.semiLatRect()
        r = self.spacecraft.classical.radius()
        h = self.spacecraft.classical.angMomentum()

        dadt = (2*a**2/h) * (e*math.sin(nu)*aR + (p/r)*aTang)
        dedt = (p/h) * (math.sin(nu)*aR + (math.cos(nu) + (e + math.cos(nu))/(1 + e*math.cos(nu))) * aTang)
        didt = 0.0      # Only from normal acceleration, not present in force model (calculation speed enhancement)
        draandt = 0.0   # Only from normal acceleration, not present in force model (calculation speed enhancement)
        dargpdt = (1/(h*e)) * (-p*math.cos(nu)*aR + (p+r)*math.sin(nu)*aTang) - draandt*math.cos(i)
        dnudt = h/r**2 - dargpdt - draandt*math.cos(i)

        return [dadt, dedt, didt, draandt, dargpdt, dnudt, dmPropdt]

    def propagate(self):
        """Propagates the spacecraft forward by a time step"""
        # Propagate by one time step
        tspan = [0, self.propCtrls.dt.value]
        # Extract the state, removing units
        state = [x for x in self.spacecraft.classical.stateVector()]
        state.append(self.spacecraft.massProp)
        sol = scipy.integrate.solve_ivp(self.GaussPlanEq, tspan, state)

        # Update state, epoch, and mass based on solution from this propagation time step
        newState = sol.y[0:6,-1]
        endTimeDeltaInSeconds = sol.t[-1]
        newMass = sol.y[-1,-1]
        self.spacecraft.classical.setState(newState)
        endEpoch = self.spacecraft.epoch + TimeDelta(endTimeDeltaInSeconds, format='sec')
        self.spacecraft.setEpoch(endEpoch)
        self.spacecraft.setMassProp(newMass)

        # TODO: determine when fuel has been depleted
        if (not self.spacecraft.fuelDepletedFlag):
            self.spacecraft.checkIfFuelDepleted()

def Main(inputFile):
    # Handles error if input file not found here
    with open(inputFile, 'r') as file:
        inpData = json.load(file)
    
    # Extracts propagation controls from input file
    propCtrls = PropagationControls(inpData['propagation'])

    # Extracts drag force model settings
    force = ForceModel(inpData['forceModel'])

    # Sets up each spacecraft
    spacecrafts = []
    scNum = 0
    for sc in inpData['spacecraft']:
        spacecrafts.append(Spacecraft(sc, scNum))
        scNum += 1
    
    # Sets up each propagator
    # TODO: wrap the burn start time in optiimizer
    burnStartAfterEpochInMinutes = 0.1
    propagators = []
    for sc in spacecrafts:
        burnEpoch = sc.epoch + TimeDelta(burnStartAfterEpochInMinutes * 60, format='sec')       # Converts minutes offset to seconds offset
        propagators.append(Propagator(force, propCtrls, sc, burnEpoch))
    
    # Propagate the propagators in serial
    for prop in propagators:
        while prop.spacecraft.classical.altitude() > prop.propCtrls.finalAltitude:
            prop.propagate()
            
        print(f'Epoch after crossing altitude threshold: {prop.spacecraft.epoch}')
        print(f'Current altitude: {prop.spacecraft.classical.altitude()}')
        print(f'fuel afterwards: {prop.spacecraft.massProp}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputFile', type=str, required=True, help="Path to input JSON file")
    args = parser.parse_args()

    Main(args.inputFile)
