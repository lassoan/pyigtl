"""
============================
Send polygonal mesh
============================

Sends a surface mesh that is read from a .ply file.

"""

import pyigtl  # pylint: disable=import-error
import vtk  # pylint: disable=import-error

# Read mesh
reader = vtk.vtkPLYReader()
reader.SetFileName("path/to/cylinder.ply")
reader.Update()
polydata = reader.GetOutput()

# Send mesh to OpenIGTLink server
client = pyigtl.OpenIGTLinkClient(host="127.0.0.1", port=18944)
output_message = pyigtl.PolyDataMessage(polydata, device_name='Mesh')
client.send_message(output_message)
