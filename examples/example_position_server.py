"""
============================
Position sending server
============================

Simple application that starts a server that provides a stream of random positions.

"""

import pyigtl  # pylint: disable=import-error

from time import sleep
import numpy as np
from math import sin

server = pyigtl.OpenIGTLinkServer(port=18944)

timestep = 0
while True:
    if not server.is_connected():
        # Wait for client to connect
        sleep(0.1)
        continue

    # Generate data
    timestep += 1
    point = np.random.random(size=3)
    quaternions = np.random.random(size=4)
    position_message = pyigtl.PositionMessage(point, quaternions, device_name='Position')
    server.send_message(position_message, wait=True)
    # Since we wait until the message is actually sent, the message queue will not be flooded
