# *pyigtl*

Python implementation of [OpenIGTLink](http://openigtlink.org/), a lightweight real-time data transfer protocol developed for image-guided therapy applications (surgical navigation, image-guided surgery, ultrasound-guided interventions, etc.).

Tested with [3D Slicer](https://www.slicer.org) and [PLUS Toolkit](http://plustoolkit.org/).

Implemented message types: IMAGE, TRANSFORM, STRING.

## Installation

Using [pip](https://pip.pypa.io/en/stable/):

```
pip install pyigtl
```

## Example

```
from pyigtl.comm import OpenIGTLinkClient
client = OpenIGTLinkClient(host="127.0.0.1", port=18944)
message = client.wait_for_message("ToolToReference")
print(message)
```
