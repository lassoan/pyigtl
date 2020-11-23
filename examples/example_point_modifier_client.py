"""
============================
Point modifier
============================

Projects received points to a plane and sends it back as another points device.

"""

import numpy as np
import pyigtl  # pylint: disable=import-error

client = pyigtl.OpenIGTLinkClient(host="127.0.0.1", port=18945)
while True:
    # Get point list from "F" device
    input_message = client.wait_for_message("F")
    # print(input_message)

    # Project points to XY plane
    newPositions = np.array(input_message.positions)
    newPositions[:, 2] = 0.0

    # Send updated positions and color as "F-modified" device
    output_message = pyigtl.PointMessage(
        device_name='F-modified', positions=newPositions,
        names=input_message.names, rgba_colors=[255, 0, 0, 255])
    client.send_message(output_message)
