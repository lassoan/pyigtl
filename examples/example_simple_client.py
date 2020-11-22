from pyigtl.comm import OpenIGTLinkClient
from pyigtl.messages import PointMessage
client = OpenIGTLinkClient(host="127.0.0.1", port=18945)
message = client.wait_for_message("ImageToReference", timeout=3)
