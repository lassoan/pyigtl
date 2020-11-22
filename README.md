# *pyigtl*

Python implementation of [OpenIGTLink](http://openigtlink.org/), a lightweight real-time data transfer protocol developed for image-guided therapy applications (surgical navigation, image-guided surgery, ultrasound-guided interventions, etc.).

Tested with [3D Slicer](https://www.slicer.org) and [PLUS Toolkit](http://plustoolkit.org/).

Implemented message types: IMAGE, TRANSFORM, STRING, POINT. Supports latest OpenIGTLink protocol (version 3), including custom metadata fields.

## Installation

The package is not yet available on PyPI or conda, therefore you need to download the files to your computer and add it to `PYTHONPATH` environent variable (for example, `PYTHONPATH=c:/dev/pyigtl`) before starting Python. Alternatively, you can add the module path in your Python code before importing pyigtl, for example:

```
import sys
sys.path.append("c:/dev/pyigtl")
```

## Example

Wait until a message is received from a device named `ToolToReference` and print the message content:

```
from pyigtl.comm import OpenIGTLinkClient
client = OpenIGTLinkClient(host="127.0.0.1", port=18944)
message = client.wait_for_message("ToolToReference")
print(message)
```
