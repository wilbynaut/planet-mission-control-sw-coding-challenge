
# Standard Library imports
import argparse
# import datetime
import json
import math
# from numba import njit
import numpy as np
import scipy

# Other imports
from astropy import units as u
from astropy.time import Time, TimeDelta

import constants

def Defaults():
 return {
    "forceModel": {
        "drag": {
            "densityRef": 3.206e-4,
            "heightRef": 60e3,
            "scaleHeight": 7.714e3
        },
        "muEarth": 3.986004e14 
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
            self.finalAltitude = prop['finalAltitude'] * u.km
        else:
            self.finalAltitude = defaultProp['finalAltitude'] * u.km

        if 'radiusEarth' in prop:
            self.radiusEarth = prop['radiusEarth'] * u.km
        else:
            self.radiusEarth = defaultProp['radiusEarth'] * u.km
        
        for key in prop:
            if key not in defaultProp:
                print(f'Warning: key "{key}" is not an accepted input to "propagation" input')

class ForceModel():
    def __init__(self, force):
        self.drag = DragModel(force['drag'])
        self.muEarth = force['muEarth']

class DragModel():
    def __init__(self, drag):
        defaultDrag = Defaults()['forceModel']['drag']

        if 'densityRef' in drag:
            self.densityRef = (drag['densityRef']) * u.kg / u.m**3
        else:
            self.densityRef = (defaultDrag['densityRef']) * u.kg / u.m**3
        self.densityRef.to(u.kg / u.km**3)
        
        if 'heightRef' in drag:
            self.heightRef = (drag['heightRef']) * u.m
        else:
            self.heightRef = (defaultDrag['heightRef']) * u.m
        self.heightRef.to(u.km)
        
        if 'scaleHeight' in drag:
            self.scaleHeight = (drag['scaleHeight']) * u.m
        else:
            self.scaleHeight = (defaultDrag['scaleHeight']) * u.m
        self.scaleHeight.to(u.km)
    
    def accel(self, velocity, alt, dragCoeff, dragArea, mass):
        """Method to compute acceleration due to drag"""
        rho = self.density(alt)

        a = -0.5 * rho * dragCoeff * dragArea / mass * (velocity * u.km / u.s)**2
        return a
    
    def density(self, alt):
        """Method to compute the local atmospheric density [kg / km^3]"""
        rho = self.densityRef * np.exp(-(alt - self.heightRef) / self.scaleHeight)
        return rho.to(u.kg / u.km**3)

class Classical():
    """State defined with classical orbital elements"""
    def __init__(self, initState):
        self.a = initState['sma'] << u.km
        self.ecc = initState['eccentricity'] << u.one
        self.inc = math.radians(initState['inclination']) << u.rad
        self.raan = math.radians(initState['raan']) << u.rad
        self.argp = math.radians(initState['argp']) << u.rad
        self.nu = math.radians(initState['trueAnom']) << u.rad
    
    def semiLatRect(self):
        """Calculates the semi-latus rectum of the orbit"""
        p = self.a * (1 - self.ecc**2)
        return p
    
    def radius(self):
        """Calculates the current orbital radius"""
        r = self.semiLatRect() / (1 + self.ecc * math.cos(self.nu.value))
        return r
    
    def velocity(self):
        """Calculates velocity Based on energy: V^2/2 -mu/r = -mu/(2*a)"""
        v = math.sqrt(2 * (constants.MU / self.radius() - constants.MU / (2*self.a)).value)
        return v
    
    def altitude(self):
        """Calculates the orbit's altitude above the Earth's surface"""
        alt = self.radius() - constants.RAD_E
        return alt
    
    def angMomentum(self):
        """Calculates the magnitude of the orbit's angular momentum"""
        h = math.sqrt(constants.MU.value * self.semiLatRect().value)
        return h
    
    def stateVector(self):
        return [self.a, self.ecc, self.inc, self.raan, self.argp, self.nu]
    
    def setState(self, x):
        self.a = x[0] << u.km
        self.ecc = x[1] << u.one
        self.inc = wrapTo2Pi(x[2]) << u.rad
        self.raan = wrapTo2Pi(x[3]) << u.rad
        self.argp = wrapTo2Pi(x[4]) << u.rad
        self.nu = wrapTo2Pi(x[5]) << u.rad

class Thruster():
    """Class to define a thruster"""
    def __init__(self, sc):
        self.thrust = sc['thrust'] << u.kg * u.m / u.s**2
        self.Isp = sc['Isp'] << u.s
        self.massFlowRate = self.thrust / (self.Isp * constants.G0)

class Spacecraft():
    """Class with all information relative to a spacecraft and its current state"""
    def __init__(self, sc):
        self.name = sc['satName']
        self.classical = Classical(sc['initialState'])
        self.epoch = Time(sc['initialState']['epoch'], format='isot', scale='utc')
        self.dragArea = (sc['dragArea'] * u.m**2).to(u.km**2)
        self.dragCoeff = sc['dragCoeff'] << u.one
        self.massDry = sc['massDry'] << u.kg
        self.massProp = sc['massPropRemaining'] << u.kg
        self.thruster = Thruster(sc)
    
    def totalMass(self):
        return self.massDry + self.massProp
    
    def setEpoch(self, epoch):
        self.epoch = epoch
    
    def setMassProp(self, mass):
        self.massProp = mass << u.kg
    
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

        return aDrag.value
    
    def thrustAccel(self):
        """Calculates acceleration due to thrust [km/s^2]"""
        if (self.spacecraft.epoch >= self.burnStart) and (self.spacecraft.massProp > 0.0):
            # If we are passed the burn start epoch and have remaining fuel, apply thrust, convert from [m/s^2] to [km/s^2]
            aThrust = -(self.spacecraft.thruster.thrust / (self.spacecraft.massDry + self.spacecraft.massProp)).to(u.km / u.s**2)
            dmProp = -self.spacecraft.thruster.massFlowRate
        else:
             # If the burn has not started, or we are out of fuel, do not apply thrust
            aThrust = 0.0 << u.km / u.s
            dmProp = 0.0 << u.kg / u.s
        
        return aThrust.value, dmProp.value
    
    def GaussPlanEq(self, t, state):
        # Extract state values
        a = state[0]
        e = state[1]
        i = state[2]
        raan = state[3]
        argp = state[4]
        nu = state[5]
        mProp = state[6]

        # Precompute reused values
        aR = 0.0    # Radial perturbing acceleration
        aN = 0.0    # Normal perturbing acceleration
        aTangDrag = self.dragAccel()
        aTangThrust, dmPropdt = self.thrustAccel()
        # aTang = aTangThrust
        # aTang = aTangDrag
        aTang = aTangDrag + aTangThrust
        p = self.spacecraft.classical.semiLatRect().value
        r = self.spacecraft.classical.radius().value
        h = self.spacecraft.classical.angMomentum()

        dadt = (2*a**2/h) * (e*math.sin(nu)*aR + p/r*aTang)
        dedt = (p/h) * (math.sin(nu)*aR + (math.cos(nu) + (e + math.cos(nu))/(1 + e*math.cos(nu))) * aTang)
        didt = 0.0      # Only from normal acceleration, not present in force model
        draandt = 0.0   # Only from normal acceleration, not present in force model
        dargpdt = (1/(h*e)) * (-p*math.cos(nu)*aR + (p+r)*math.sin(nu)*aTang) - draandt*math.cos(i)
        dnudt = h/r**2 - dargpdt - draandt*math.cos(i)

        return [dadt, dedt, didt, draandt, dargpdt, dnudt, dmPropdt]

    def propagate(self):
        """Propagates the spacecraft forward by a time step"""
        # Propagate by one time step
        tspan = [0, self.propCtrls.dt.value]
        # Extract the state, removing units
        state = [x.value for x in self.spacecraft.classical.stateVector()]
        state.append(self.spacecraft.massProp.value)
        sol = scipy.integrate.solve_ivp(self.GaussPlanEq, tspan, state)

        # Update state and epoch based on solution from this priopagation time step
        self.spacecraft.classical.setState(sol.y[0:6,-1])
        endEpoch = self.spacecraft.epoch + TimeDelta(sol.t[-1], format='sec')
        self.spacecraft.setEpoch(endEpoch)
        self.spacecraft.setMassProp(sol.y[-1,-1])

        print(f'current mass prop: {self.spacecraft.massProp}')

        # TODO: determine when fuel has been depleted

        # if (self.spacecraft.epoch > self.burnStart) and (self.spacecraft.massProp > 0.0):
        #     # Apply appropriate mass propellant loss if burn is active
        #     massLoss = (self.spacecraft.thruster.massFlowRate * self.propCtrls.dt)
        #     self.spacecraft.massProp -= massLoss

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
    for sc in inpData['spacecraft']:
        spacecrafts.append(Spacecraft(sc))
    
    # Sets up each propagator
    minsAfterEpoch = 1
    propagators = []
    for sc in spacecrafts:
        # TODO: 
        burnEpoch = sc.epoch + minsAfterEpoch * u.min
        propagators.append(Propagator(force, propCtrls, sc, burnEpoch))
    
    # Propagate the propagators in serial
    for prop in propagators:
        while prop.spacecraft.classical.altitude() > prop.propCtrls.finalAltitude:
            prop.propagate()
            
            print(f'Epoch after propagation call within prop.propagate: {prop.spacecraft.epoch}')
            print(f'Current altitude: {prop.spacecraft.classical.altitude()}')
        
        print(f'fuel afterwards: {prop.spacecraft.massProp}')
    
    print(f'thrust: {prop.spacecraft.thruster.thrust}')
    print(f'Isp: {prop.spacecraft.thruster.Isp}')
    print(f'mass flow rate: {prop.spacecraft.thruster.massFlowRate}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputFile', type=str, required=True, help="Path to input JSON file")
    args = parser.parse_args()

    Main(args.inputFile)
