"""
============================
Simple client
============================

Simplest example of getting a transform from a device.

"""

import pyigtl  # pylint: disable=import-error

client = pyigtl.OpenIGTLinkClient(host="127.0.0.1", port=18946)
message = client.wait_for_message("ImageToReference", timeout=3)
print(message)
