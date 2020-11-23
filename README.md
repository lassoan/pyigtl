[![Python package](https://github.com/lassoan/pyigtl/workflows/Python%20package/badge.svg)](https://github.com/lassoan/pyigtl/actions?query=workflow%3A%22Python+package%22)
![Upload Python Package](https://github.com/lassoan/pyigtl/workflows/Upload%20Python%20Package/badge.svg)
[![PyPI version](https://badge.fury.io/py/pyigtl.svg)](https://badge.fury.io/py/pyigtl)

# *pyigtl*

Python implementation of [OpenIGTLink](http://openigtlink.org/), a lightweight real-time data transfer protocol developed for image-guided therapy applications (surgical navigation, image-guided surgery, ultrasound-guided interventions, etc.).

Tested with [3D Slicer](https://www.slicer.org), [SlicerIGT](https://www.slicerigt.org) and [PLUS Toolkit](http://plustoolkit.org/).

Supports latest OpenIGTLink protocol (version 3) and message types: IMAGE, TRANSFORM, STRING, POINT.

## Installation

Using [pip](https://pip.pypa.io/en/stable/):

```
pip install pyigtl
```

## Example

Wait until a message is received from a device named `ToolToReference` and print the message content:

```
import pyigtl
client = pyigtl.OpenIGTLinkClient("127.0.0.1", 18944)
message = client.wait_for_message("ToolToReference", timeout=5)
print(message)
```
