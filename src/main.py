""" Program to optimize burn time to deorbit a spacecraft.

An input JSON file is used to define the initial condition of a spacecraft
and potential overrides to defaults for propagation controls and force
model. The time to initiate a deorbit burn will be optimized such 
that the spacecraft reaches a target altitude when the spacecraft runs
out of fuel, or for the least fuel remaining if it does not deplete the fuel.
"""

# Standard imports
import argparse
import json
import math
import numpy as np
import os
import scipy

# Other imports
from astropy.time import Time, TimeDelta

# Project imports
import constants

def Defaults():
    """Get the default settings that can be overridden in the input JSON file.
    
    Returns
    -------
    dict:
        dict of the default settings.
    """

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
    """Keeps an angle within 0 and 2 Pi radians.

    Parameters
    ----------
    angle: float
        The angle in radians to wrap

    Returns
    -------
    float
        The input angle wrapped between 0 and 2 pi
    """

    twoPi = 2 * math.pi

    newAngle = angle

    if (angle > twoPi):
        newAngle = angle % twoPi
    elif (angle < 0):
        newAngle = angle % (-twoPi)

    return newAngle

class PropagationControls():
    """ A class to represent propagation control settings.

    Attributes
    ----------
    dt : float
        Propagation time step [s]
    finalAltitude : float
        Altitude to target [km]
    """
    
    def __init__(self, prop):
        """
        Parameters
        ----------
        prop : dict
            Dictionary of propagation controls
        """
        defaultProp = Defaults()['propagation']

        if 'dt' in prop:
            self.dt = TimeDelta(prop['dt'], format='sec')
        else:
            self.dt = TimeDelta(defaultProp['dt'], format='sec')

        if 'finalAltitude' in prop:
            self.finalAltitude = prop['finalAltitude']
        else:
            self.finalAltitude = defaultProp['finalAltitude']
        
        for key in prop:
            if key not in defaultProp:
                print(f'Warning: key "{key}" is not an accepted input to "propagation" input')

class ForceModel():
    """ A class to represent the force model used in propagation.

    Attributes
    ----------
    drag: DragModel
        The drag model used
    """

    def __init__(self, force):
        """
        Parameters
        ----------
        force : dict
            Dict containing force model controls
        """

        if 'drag' in force:
            self.drag = DragModel(force['drag'])
        else:
            self.drag = DragModel({})

class DragModel():
    """ A class to represent an exponential drag model.

    Attributes
    ----------
    densityRef: float
        Reference density for the exponential drag model [kg/km^3]
    heightRef: float
        Height at which the reference density is reported [km]
    scaleHeight: float
        Scale height for the exponential model [km]

    Methods
    -------
    accel:
        Acceleration applied to the spacecraft at its current state
    density:
        Density at the current altitude
    """
    def __init__(self, drag):
        """
        Parameters
        ----------
        drag: dict
            A dictionary representing drag model settings
        """

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
        """Compute acceleration due to drag.

        Parameters
        ----------
        velocity : float
            Spacecraft's current velocity [km/s]
        alt : float
            Spacecraft's current altitude [km]
        dragCoeff : float
            Spacecraft's drag coefficient
        dragArea : float
            Spacecraft's drag area [km^2]
        mass : float
            Spacecraft's current mass [kg]

        Returns
        -------
        float :
            Acceleration [km/s]
        """

        rho = self.density(alt)

        a = -0.5 * rho * (dragCoeff * dragArea / mass) * velocity**2
        return a
    
    def density(self, alt):
        """Compute the local atmospheric density.
        density = (reference density) * exp((height - reference height) / scale height)

        Parameters
        ----------
        alt : float
            Spacecraft's current altitude [km]

        Returns
        -------
        float : 
            Density at current altitude [kg/km^3]
        """
        rho = self.densityRef * np.exp(-(alt - self.heightRef) / self.scaleHeight)
        return rho

class Classical():
    """ A class to represent classical orbital elements.

    Attributes
    ----------
    a : float
        smei-major axis [km]
    ecc : float
        eccentricity
    inc : float 
        inclination [rad]
    raan : float 
        right ascension of the ascending node [rad]
    argp : float 
        argument of pergiee [rad]
    nu : float 
        true anomaly [rad]
    
    Methods
    -------
    semiLatRect : 
        Calculates the orbit's semi-latus rectum
    radius : 
        Calculates the orbit's radius
    velocity :
        Calculates the orbit's velocity
    altitude :
        Calculates the orbit's altitude above the Earth's surface
    angMomentum : 
        Calculates the orbit's angular momentum magnitude
    stateVector :
        Returns the orbit's state as an array
    setState :
        Sets the state based on an array of orbital elements
    """

    def __init__(self, initState):
        """
        ----------
        initState : dict
            Dict containing the classical orbital elements
        """
        self.a = initState['sma']
        self.ecc = initState['ecc']
        self.inc = math.radians(initState['inc'])
        self.raan = math.radians(initState['raan'])
        self.argp = math.radians(initState['argp'])
        self.nu = math.radians(initState['trueAnom'])
    
    def semiLatRect(self):
        """Calculates the semi-latus rectum of the orbit.

        Returns
        -------
        float
            Semi-latus rectum
        """

        p = self.a * (1 - self.ecc**2)
        return p
    
    def radius(self):
        """Calculates the current orbital radius

        Returns
        -------
        float
            Orbital radius
        """

        r = self.semiLatRect() / (1 + self.ecc * math.cos(self.nu))
        return r
    
    def velocity(self):
        """Calculates velocity Based on energy: V^2/2 -mu/r = -mu/(2*a)

        Returns
        -------
        float
            Orbital velocity
        """

        v = math.sqrt(2 * (constants.MU / self.radius() - constants.MU / (2*self.a)))
        return v
    
    def altitude(self):
        """Calculates the orbit's altitude above the Earth's surface

        Returns
        -------
        float
            Orbital altitude
        """

        alt = self.radius() - constants.RAD_E
        return alt
    
    def angMomentum(self):
        """Calculates the magnitude of the orbit's angular momentum

        Returns
        -------
        float
            Angular momentum magnitude
        """

        h = math.sqrt(constants.MU * self.semiLatRect())
        return h
    
    def stateVector(self):
        """_summary_

        Returns
        -------
        array:
            Array of the current orbital elements
        """
        return [self.a, self.ecc, self.inc, self.raan, self.argp, self.nu]
    
    def setState(self, x):
        """_summary_

        Parameters
        ----------
        x : array
            Array of the orbital elements
        """
        self.a = x[0]
        self.ecc = x[1]
        self.inc = wrapTo2Pi(x[2])
        self.raan = wrapTo2Pi(x[3])
        self.argp = wrapTo2Pi(x[4])
        self.nu = wrapTo2Pi(x[5])

class Thruster():
    """ A class to define a thruster.

    Attributes
    ----------
    thrust : float
        thruster's thrust [N]
    Isp : float
        Specific impulse [s]
    massFlowRate : float
        Mass flow rate of the thruster [kg/s]
    """

    def __init__(self, thruster):
        """
        Parameters
        ----------
        thruster : dict
            Dict of the thruster parameters
        """
        self.thrust = thruster['thrust']
        self.Isp = thruster['Isp']
        self.massFlowRate = self.thrust / (self.Isp * constants.G0)

class Spacecraft():
    """ Class with all information related to a spacecraft and its current state.

    Attributes
    ----------
    satName : str
        Spacecraft's name
    classical : Classical
        Classical orbital elements representing state
    dragArea : float
        Drag area [km^2]
    dragCoeff : float
        Drag coefficient
    massDry : float
        Dry mass of the spacecraft [kg]
    massProp : float
        Propellant mass of the spacecraft [kg]
    thruster : Thruster
        Thruster to execute burns
    fuelDepletedFlag : bool
        Flag indicating if the fuel has been depleted
    fuelDepletedEpoch : astropy.Time
        Epoch when the spacecraft ran out of fuel
    initialEpoch : astropy.Time
        The initial epoch of the spacecraft
    ephems : array
        Array of tuples of the epoch, mass, and state at each time step
    
    Methods
    -------
    totalMass :
        Gets the total mass of the spacecraft
    setEpoch :
        Sets a new eopch for the spacecraft
    setMassProp :
        Sets the new mass propellant remaining
    checkIfFuelDepleted :
        Checks if fuel has been depleted, and updates depleted epoch if it is
    writeEphem :
        Writes the ephemerides to a file
    writeOutputMetrics :
        Writes the optimization metrics to a file
    """

    def __init__(self, sc, scNum):
        """
        Parameters
        ----------
        sc : dict
            Dict of the spaeccraft parameters
        scNum : int
            Number of spacecraft defined in the input
        """
        if 'satName' in sc:
            self.satName = sc['satName']
        else:
            self.satName = "sat" + str(scNum)
        self.classical = Classical(sc['initialState'])
        self.epoch = Time(sc['initialState']['epoch'], format='isot', scale='utc')
        self.dragArea = sc['dragArea'] * 1e-6       # Convert m^2 to km^2
        self.dragCoeff = sc['dragCoeff']
        self.massDry = sc['massDry']
        self.massProp = sc['massProp']
        self.thruster = Thruster(sc)

        self.fuelDepletedFlag = False
        self.fuelDepletedEpoch = self.epoch
        self.initialEpoch = self.epoch
        self.ephems = []
    
    def totalMass(self):
        """Gets the total mass of the spacecraft

        Returns
        -------
        float
            total mass of the spacecraft [kg]
        """
        return self.massDry + self.massProp
    
    def setEpoch(self, epoch):
        """Sets a new eopch for the spacecraft

        Parameters
        ----------
        epoch : astropy.Time
            New epoch of the spacecraft
        """
        self.epoch = epoch
    
    def setMassProp(self, mass):
        """Sets the new mass of propellant remaining

        Parameters
        ----------
        mass : float
            mass of propellant remaining [kg]
        """
        self.massProp = mass
    
    def checkIfFuelDepleted(self):
        """Checks if fuel has been depleted, and updates depleted epoch if it is
        """
        if self.massProp <= 0.0:
            self.fuelDepletedFlag = True
            self.fuelDepletedEpoch = self.epoch
    
    def writeEphem(self, outputDir):
        """Writes the ephemerides to a file

        Parameters
        ----------
        outputDir : str
            Location to store the file
        """
        fileName = self.satName + '.ephem'
        outFile = os.path.join(outputDir, fileName)
        header = 'Epoch\t\t\t\t\tMass [kg]\t\t\taltitude [km]\tsma [km]\t\t eccentricity\t\t\tinclination [rad]\traan [rad]\t\t\targp [rad]\t\t\ttrueAnom[rad]\n'
        with open(outFile, 'w') as f:
            f.write(header)
            for eph in self.ephems:
                epoch, mass, altitude, elems = eph
                ephem = f'{epoch} {mass} {altitude} {elems[0]} {elems[1]} {elems[2]} {elems[3]} {elems[4]} {elems[5]}\n'
                f.write(ephem)

    def writeOutputMetrics(self, outputDir, optimalBurnStart):
        """Writes the optimization metrics to a file

        Parameters
        ----------
        outputDir : str
            Location to store the file
        optimalBurnStart : float
            Optimal burn time found [minutes]
        """
        fileName = self.satName + '_optimal_metrics.txt'
        outFile = os.path.join(outputDir, fileName)
        with open(outFile, 'w') as f:
            f.write(f'Optimal burn start time from epoch: {optimalBurnStart} minutes\n')
            f.write(f'Altitude threshold cross epoch: {self.epoch}\n')
            f.write(f'Fuel remaining at altitude threshold cross: {self.massProp}\n')
            if self.fuelDepletedFlag:
                f.write(f'Fuel depletion epoch: {self.fuelDepletedEpoch}\n')
            else:
                f.write('Fuel was not depleted before crossing the target altitude\n')
    
class Propagator():
    """Class to handle propagation of a spacecraft

    Attributes
    -------
    force : ForceModel
        Force model settings to use
    propCtrls : PropagationControls
        Propagation control settings
    spacecraft : Spacecraft
        Spacecraft to propagate
    burnStart : astropy.Time
        Epoch at which to initiate the de-orbit burn
    storeEphem : bool
        Indicates if ephemeris should be stored for the spacecraft during propagation
    
    Methods
    -------
    dragAccel :
        Calculates the acceleration due to drag
    thrustAccel :
        Calculates acceleration due to thrust
    GaussPlanEq :
        Gauss Planetary Equations to determine rate of change of the spacecraft
    propagate :
        Propagates the spacecraft forward by a time step
    """
    # A class to propagate a spacecraft with force model and propagation controls

    def __init__(self, force, propCtrls, spacecraft, burnStart, storeEphem):
        """
        Parameters
        ----------
        force : ForceModel
            Force model settings to use
        propCtrls : PropagationControls
            Propagation control settings
        spacecraft : Spacecraft
            Spacecraft to propagate
        burnStart : astropy.Time
            Epoch at which to initiate the de-orbit burn
        storeEphem : bool
            Indicates if ephemeris should be stored for the spacecraft during propagation
        """
        self.force = force
        self.propCtrls = propCtrls
        self.spacecraft = spacecraft
        self.burnStart = burnStart
        self.storeEphem = storeEphem

    def dragAccel(self):
        """Calculates the acceleration due to drag.

        Returns
        -------
        float :
            Acceleration due to drag
        """
        # Calculates tangential acceleration due to drag [km/s^2]

        v = self.spacecraft.classical.velocity()
        mass = self.spacecraft.totalMass()
        aDrag = self.force.drag.accel(v, self.spacecraft.classical.altitude(), self.spacecraft.dragCoeff, self.spacecraft.dragArea, mass)

        return aDrag
    
    def thrustAccel(self):
        """Calculates acceleration due to thrust.

        Returns
        -------
        float, float
            Tuple of thrust acceleration and mass loss rate
        """
    
        # Calculates acceleration due to thrust [km/s^2]

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
        """Gauss Planetary Equations to determine rate of change of the spacecraft.

        Parameters
        ----------
        t : float
            current time offset from the beginning of a propagation step
        state : array
            Current spacecraft state

        Returns
        -------
        array
            Rate of change of the spacecraft state
        """
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
        """Propagates the spacecraft forward by a time step

        Sets the new spacecraft epoch, mass, and state based on solving Gauss Planetary Equations
        """
        
        # Propagate by one time step
        tspan = [0, self.propCtrls.dt.value]
        # Extract the state
        state = [x for x in self.spacecraft.classical.stateVector()]
        state.append(self.spacecraft.massProp)
        # Propagate to the time bounds
        sol = scipy.integrate.solve_ivp(self.GaussPlanEq, tspan, state)

        # Update state, epoch, and mass based on solution from this propagation time step
        newState = sol.y[0:6,-1]
        endTimeDeltaInSeconds = sol.t[-1]
        newMass = sol.y[-1,-1]
        self.spacecraft.classical.setState(newState)
        endEpoch = self.spacecraft.epoch + TimeDelta(endTimeDeltaInSeconds, format='sec')
        self.spacecraft.setEpoch(endEpoch)
        self.spacecraft.setMassProp(newMass)

        # Store ephemeris records
        if self.storeEphem:
            self.spacecraft.ephems.append((self.spacecraft.epoch, self.spacecraft.massProp, self.spacecraft.classical.altitude(), newState))

        # Determine when fuel has been depleted
        if (not self.spacecraft.fuelDepletedFlag):
            self.spacecraft.checkIfFuelDepleted()

def makePropagatorAndPropagate(burnStartAfterEpochInMinutes, force, propCtrls, scDict, scNum, storeEphem):
    """Creates a propagator and propagates until the target altitude is reached.

    Parameters
    ----------
    burnStartAfterEpochInMinutes : float
        How long after the initial epoch to begin the de-orbit burn
    force : ForceModel
        Force model settings to use
    propCtrls : PropagationControls
        Propagation control settings
    scDict : dict
        Dict of the spacecraft settings to propagate
    scNum : int
        Number of spacecraft defined in the input
    storeEphem : bool
        Indicates if ephemeris should be stored for the spacecraft during propagation

    Returns
    -------
    Propagator
        Propagator at the end of propagation
    """

    sc = Spacecraft(scDict, scNum)
    burnEpoch = sc.epoch + TimeDelta(burnStartAfterEpochInMinutes * 60, format='sec')       # Converts minutes offset to seconds offset
    prop = Propagator(force, propCtrls, sc, burnEpoch, storeEphem)

    while prop.spacecraft.classical.altitude() > prop.propCtrls.finalAltitude:
        prop.propagate()
        
    return prop
    
def optimizeFunction(burnStartAfterEpochInMinutes, force, propCtrls, scDict, scNum):
    """_summary_

    Parameters
    ----------
    burnStartAfterEpochInMinutes : float
        How long after the initial epoch to begin the de-orbit burn
    force : ForceModel
        Force model settings to use
    propCtrls : PropagationControls
        Propagation control settings
    scDict : dict
        Dict of the spacecraft settings to propagate
    scNum : int
        Number of spacecraft defined in the input

    Returns
    -------
    float
        Cost function evaluated
    """

    storeEphem = False
    prop = makePropagatorAndPropagate(burnStartAfterEpochInMinutes, force, propCtrls, scDict, scNum, storeEphem)

    if prop.spacecraft.fuelDepletedFlag:
        # If the spacecraft ran out of fuel, target the run out time close to the altitude threshold
        timeDepletionToThresholdInSecs = (prop.spacecraft.fuelDepletedEpoch - prop.spacecraft.initialEpoch).to_value('sec').value
        fuelRemaining = 0.0
    else:
        # If the spacecraft did not run out of fuel, optimize for least fuel remaining (translates to quickest de-orbit time)
        timeDepletionToThresholdInSecs = 0.0
        fuelRemaining = prop.spacecraft.massProp
    
    # Will optimize for closest time to deplete fuel prior to crossing altitude threshold, or for the minimum fuel remaining
    toMinimize = timeDepletionToThresholdInSecs + fuelRemaining
    
    return toMinimize

def Main(inputFile):
    """Sets up optimization of each spacecraft in the input JSON file.

    Parameters
    ----------
    inputFile : str
        Location of the input file
    """
    outputDir = 'results'
    os.makedirs(outputDir, exist_ok=True)

    # Reads in the input file
    with open(inputFile, 'r') as file:
        inpData = json.load(file)
    
    # Extracts propagation controls from input file
    if 'propagation' in inpData:
        propCtrls = PropagationControls(inpData['propagation'])
    else:
        propCtrls = PropagationControls({})

    # Extracts drag force model settings
    if 'forceModel' in inpData:
        force = ForceModel(inpData['forceModel'])
    else:
        force = ForceModel({})

    # Sets up each spacecraft propagation
    for i,scDict in enumerate(inpData['spacecraft']):
        # A bit annoying, must re-create the spacecraft each optimization step because of Python copying. 
        # There are other work-arounds, but this is good enough for now.
        scNum = i
        if 'satName' not in scDict:
            scDict['satName'] = "sat" + str(scNum)
        burnStartAfterEpochInMinutes = 1
        initGuess = [burnStartAfterEpochInMinutes]  # Initial guess to initialize the optimizer
        bounds = [(0, 50)]                          # Bounds on the burn start offset to search (0 - 50 minutes)
        args = (force, propCtrls, scDict, scNum)    # Extra arguments needed for the cost function

        # Optimize using a Nelder-Mead minimizer
        scMin = scipy.optimize.minimize(optimizeFunction, initGuess, args=args, bounds=bounds, method='Nelder-Mead')

        # Extract the optimal solution
        optimalBurnStart = scMin['x'][0]

        # Propagate the optimal spacecraft again to get more metrics
        # Could be avoided using a custom optimizer, but the used method is written in C and will ultimately be faster
        storeEphem = True
        optimalProp = makePropagatorAndPropagate(optimalBurnStart, force, propCtrls, scDict, scNum, storeEphem)
        
        optimalProp.spacecraft.writeEphem(outputDir)
        optimalProp.spacecraft.writeOutputMetrics(outputDir, optimalBurnStart)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputFile', type=str, required=True, help="Path to input JSON file")
    args = parser.parse_args()

    Main(args.inputFile)
