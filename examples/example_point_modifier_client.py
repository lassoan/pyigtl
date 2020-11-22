import numpy as np
from pyigtl.comm import OpenIGTLinkClient
from pyigtl.messages import PointMessage

client = OpenIGTLinkClient(host="127.0.0.1", port=18945)
while True:
    # Get point list from "F" device
    input_message = client.wait_for_message("F")
    # print(input_message)

    # Project points to XY plane
    newPositions = np.array(input_message.positions)
    newPositions[:,2] = 0.0

    # Send updated positions and color as "F-modified" device
    output_message = PointMessage(device_name='F-modified', positions=newPositions,
        names=input_message.names, rgba_colors=[255,0,0,255])
    client.send_message(output_message)
