#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
brownian() implements one dimensional Brownian motion (i.e. the Wiener process).
"""

# File: sensorsim.py

from math import sqrt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from scipy.stats import norm

import argparse
import igrow_pb.location_pb2 as location_pb2
import igrow_pb.observation_pb2 as observation_pb2
import matplotlib.pyplot as plt
import numpy as np
import requests
import time
import uuid

DEFAULT_URL = 'http://localhost:9998/observations'

current_time_ms = lambda: int(round(time.time() * 1000))

def brownian(x0, n, dt, delta, out=None):
    """
    Generate an instance of Brownian motion (i.e. the Wiener process):

        X(t) = X(0) + N(0, delta**2 * t; 0, t)

    where N(a,b; t0, t1) is a normally distributed random variable with mean a and
    variance b.  The parameters t0 and t1 make explicit the statistical
    independence of N on different time intervals; that is, if [t0, t1) and
    [t2, t3) are disjoint intervals, then N(a, b; t0, t1) and N(a, b; t2, t3)
    are independent.

    Written as an iteration scheme,

        X(t + dt) = X(t) + N(0, delta**2 * dt; t, t+dt)


    If `x0` is an array (or array-like), each value in `x0` is treated as
    an initial condition, and the value returned is a numpy array with one
    more dimension than `x0`.

    x0 : float or numpy array (or something that can be converted to a numpy array
         using numpy.asarray(x0)).
        The initial condition(s) (i.e. position(s)) of the Brownian motion.
    n : int
        The number of steps to take.
    dt : float
        The time step.
    delta : float
        delta determines the "speed" of the Brownian motion.  The random variable
        of the position at time t, X(t), has a normal distribution whose mean is
        the position at time t=0 and whose variance is delta**2*t.
    out : numpy array or None
        If `out` is not None, it specifies the array in which to put the
        result.  If `out` is None, a new numpy array is created and returned.

    A numpy array of floats with shape `x0.shape + (n,)`.

    Note that the initial value `x0` is not included in the returned array.
    """

    x0 = np.asarray(x0)

    # For each element of x0, generate a sample of n numbers from a
    # normal distribution.
    r = norm.rvs(size=x0.shape + (n,), scale=delta*sqrt(dt))

    # If `out` was not given, create an output array.
    if out is None:
        out = np.empty(r.shape)

    # This computes the Brownian motion by forming the cumulative sum of
    # the random samples.
    np.cumsum(r, axis=-1, out=out)

    # Add the initial condition.
    out += np.expand_dims(x0, axis=-1)

    return out


def main(args):
    # The Wiener process parameter.
    delta = 2
    # Total time.
    T = 10.0
    # Number of steps.
    N = 20
    # Time step size
    dt = T/N
    # Number of realizations to generate.
    m = 20
    # Create an empty array to store the realizations.
    x = np.empty((m,N+1))
    # Initial values of x.
    x[:, 0] = 15.0

    brownian(x[:,0], N, dt, delta, out=x[:,1:])

    t = np.linspace(0.0, N*dt, N+1)

    fig = plt.figure()  # a new figure window
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
    
    for k in range(m):
        ax.plot(t, x[k])
    ax.set_xlabel('t', fontsize=16)
    ax.set_ylabel('x', fontsize=16)
    ax.grid(True)

    canvas.print_figure('Brownian Motion')

    sensor_id = uuid.uuid4()
    
    while True:
        observations = observation_pb2.Observations()

        for i in range(N):
            observation = observations.observations.add()
            observation.type = observation_pb2.Observation.ENVIRONMENTAL_SENSOR
            observation.sensorId = str(sensor_id)
            observation.timestamp = current_time_ms()
            observation.mode = observation_pb2.Observation.ACTIVE
            
            observation.location.latitude = 50939970
            observation.location.longitude = -1415058
            observation.location.altitude = 0
            observation.location.hdop = 0
            observation.location.vdop = 0

            observation.envSensorObservation.temperature = x[0, i]
            observation.envSensorObservation.humidity = x[1, i]
            observation.envSensorObservation.irradiance = x[2, i]

            time.sleep(5)  # simulate the sensor sampling behaviour
             
        print(observations)
        r = requests.post(DEFAULT_URL, data=observations.SerializeToString())
        time.sleep(5)

        r.raise_for_status()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simulate sensor data and write '
                                     'to observation service.',
                                     epilog='''Example:
                                    
$ python sensorsim.py --sandbox
''')

    parser.add_argument('--url', action='store',
                        dest='url',
                        help='Override default URL')
    parser.add_argument('--sandbox', action='store_false',
                        dest='prod',
                        help='Use the iGrow sandbox environment rather than production')

    parser.add_argument('--version', action='version', version='%(prog)s 0.1')

    parser.set_defaults(url=DEFAULT_URL)
    parser.set_defaults(prod=True)
    
    args = parser.parse_args()

    main(args)


