"""
============================
Tracked image data server
============================

Simple application that starts a server that sends images, transforms, and strings

"""

from pyigtl.comm import OpenIGTLinkServer
from pyigtl.messages import ImageMessage, TransformMessage, StringMessage
from math import cos, sin, pi
from time import sleep
import numpy as np

server = OpenIGTLinkServer(port=18944, localServer=True)

image_size = [400, 200]

timestep = 0

while True:

    if not server.is_connected():
        # Wait for client to connect
        sleep(0.1)
        continue

    timestep += 1

    # Generate image
    data = np.random.randn(image_size[0], image_size[1])*50+100
    image_message = ImageMessage(data, device_name="Image")

    # Generate transform
    matrix = np.eye(4)
    matrix[0,3] = sin(timestep*0.01) * 20.0
    rotation_angle_rad = timestep*0.5 * pi/180.0
    matrix[1,1] = cos(rotation_angle_rad)
    matrix[2,1] = -sin(rotation_angle_rad)
    matrix[1,2] = sin(rotation_angle_rad)
    matrix[2,2] = cos(rotation_angle_rad)
    transform_message = TransformMessage(matrix, device_name="ImageToReference", timestamp=image_message.timestamp)

    # Generate string
    string_message = StringMessage("TestingString_"+str(timestep), device_name="Text", timestamp=image_message.timestamp)

    # Send messages
    server.send_message(image_message)
    server.send_message(transform_message)
    server.send_message(string_message)

    # Print received messages
    messages = server.get_latest_messages()
    for message in messages:
        print(message.device_name)

    # Allow time for network transfer
    sleep(0.01)
