"""pyigtl package -- native Python implementation of OpenIGTLink communication protocol.

-----------
Quick Start
-----------

1. A simple program to receive a transform from an OpenIGTLink server
   file::

    from pyigtl.comm import OpenIGTLinkClient
    client = OpenIGTLinkClient(host="127.0.0.1", port=18944)
    message = client.wait_for_message("ToolToReference")
    print(message)

2. See the files in the examples directory that came with this package for more examples.

3. Any questions or need help? Post on 3D Slicer forum:
   https://discourse.slicer.org

5. Bugs and other issues can be reported in the issue tracker:
   https://www.github.com/slicerigt/pyigtl

"""

from .comm import OpenIGTLinkServer, OpenIGTLinkClient
from .messages import MessageBase, ImageMessage, TransformMessage, StringMessage, PointMessage

from ._version import __version__, __version_info__

__all__ = ['OpenIGTLinkServer',
           'OpenIGTLinkClient',
           'MessageBase',
           'ImageMessage',
           'TransformMessage',
           'StringMessage',
           'PointMessage',
           '__version__',
           '__version_info__'
           ]
