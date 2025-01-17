# planet-mission-control-sw-coding-challenge
Coding challenge for Software Orbits R&amp;D Engineer at Planet


TODO: requirements.txt
make note that removing astropy units significantly imrpoved runtime, but is potentially more prone to human errors in coding

# Running the program

To run this program from the main directory on Linux, input the command:

`python src/main.py --inputFile <path/to/input.json>`

The main program will take in an input JSON file with configuration settings for the spacecraft initial conditions, propagation controls, and force model.

# Input JSON File
## Minimal Request
An example input file is located in `inputs/test_simple.json`. This file shows a basic run of the program with a single spacecraft. All of the fields (except spacecraft name) in this file are required, and represent a minimal request.

The only required field is the `spacecraft` field. This is an  array of spacecraft to propagate.

Each spacecraft contains the following inputs:

- satName: The name of the satellite. If no name is provided, satellites will recieve a numbered name based on the input order.
- initialState: The initial Classical orbital element state of the satellite
    - epoch: The epoch at which the initial state is defined as an ISOT string
    - sma: semi major axis [ km ]
    - ecc: eccentricity
    - inc: inclination [ deg ]
    - raan: right ascension of the ascending node [ deg ]
    - argp: argument of perigee [ deg ]
    - trueAnom: true anomaly [ deg ]
- dragArea: The area of the satellite exposed to drag [ m<sup>2</sup> ]
- dragCoeff: The drag coefficient of the satellite
- massDry: The dry mass of the satellite [ kg ]
- massProp: The mass opf the satellite's propellant [ kg ]
- thrust: The thrust of the thruster [ N ]
- Isp: The specific impulse of the thruster [ s ]

## Verbose Request
A verbose example input file is located in `inputs/test_verbose.json`. This file shows which other settings are exposed to the user. This file also shows the default values used when not included in the JSON file by a user.

The following fields are exposed to the user:

- forceModel
    - drag: controls for an exponential drag model
        - densityRef: Reference density at a certain altitude [ kg / m<sup>3</sup> ]
        - heightRef: Height at which the reference density is reported [ m ]
        - scaleHeight: Scale height for the exponential drag model [ m ]
- propagation: controls for propagation
    - dt: Propagation time step [ s ]
    - finalAltitude: Target altitude to end propagation [ km ]
    - radiusEarth: Radius of the Earth [ km ]