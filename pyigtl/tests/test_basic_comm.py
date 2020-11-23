# -*- coding: utf-8 -*-

import numpy as np
import pyigtl
import time
import unittest


class TestServerClientRoundtrip(unittest.TestCase):
    """
    Test complete message roundtrip: server send -> client receive -> client send -> server receive
    """

    def setUp(self):
        self.server = pyigtl.OpenIGTLinkServer(port=19944)
        self.client = pyigtl.OpenIGTLinkClient(host="127.0.0.1", port=19944)
        time.sleep(1.0)

    def tearDown(self):
        self.server.stop()
        self.client.stop()

    def test_send_messages(self):
        """Test if both server and client can send and receive messages"""

        # Server send
        original_message = pyigtl.StringMessage("TEST Message", device_name="TestString")
        self.assertTrue(self.server.send_message(original_message))
        time.sleep(1.0)

        # Client receive
        received_messages = self.client.get_latest_messages()
        self.assertTrue(len(received_messages) > 0)

        # Client send
        for message in received_messages:
            self.client.send_message(message)
        time.sleep(1.0)

        # Server receive
        received_messages = self.server.get_latest_messages()
        self.assertTrue(len(received_messages) > 0)

        # Verify message content
        message_matched = False
        for message in received_messages:
            if message.device_name == original_message.device_name:
                # Original message was received
                self.assertEqual(message.string, original_message.string)
                message_matched = True
                break
        self.assertTrue(message_matched)


class TestMessageTypes(unittest.TestCase):
    """
    Tests that all message types can be packed and unpacked and message content is preserved.
    """

    def setUp(self):
        self.server = pyigtl.OpenIGTLinkServer(port=19945)
        self.client = pyigtl.OpenIGTLinkClient(host="127.0.0.1", port=19945)
        time.sleep(1.0)

    def tearDown(self):
        self.server.stop()
        self.client.stop()

    def test_send_receive(self):
        device_name = "Test"

        test_messages = [
            pyigtl.StringMessage("some message", device_name=device_name),
            pyigtl.ImageMessage(np.random.randn(30, 10, 5) * 50 + 100, device_name=device_name),
            pyigtl.PointMessage([[20, 30, 10], [2, -5, -10], [12.4, 11.3, 0.3]], device_name=device_name),
            pyigtl.TransformMessage(np.eye(4), device_name=device_name),
        ]

        for message in test_messages:
            self.assertTrue(self.server.send_message(message))
            self.assertIsNotNone(self.client.wait_for_message(device_name=device_name, timeout=5))

    def test_pack_unpack(self):
        device_name = "Test"

        test_messages = [
            pyigtl.StringMessage("some message", device_name=device_name),
            pyigtl.ImageMessage(np.random.randn(30, 10, 5)*50+100, device_name=device_name),
            pyigtl.PointMessage([[20, 30, 10], [2, -5, -10], [12.4, 11.3, 0.3]], device_name=device_name),
            pyigtl.TransformMessage(np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [0, 0, 0, 1]]), device_name=device_name),
        ]

        pack_unpack_inconsistencies_found = 0
        for message in test_messages:
            print("Original message:\n"+str(message))
            packed = message.pack()
            header_fields = pyigtl.MessageBase.parse_header(packed[:pyigtl.MessageBase.IGTL_HEADER_SIZE])
            new_message = pyigtl.MessageBase.create_message(header_fields['message_type'])
            new_message.unpack(header_fields, packed[pyigtl.MessageBase.IGTL_HEADER_SIZE:])
            print("Packed/unpacked message:\n"+str(new_message))
            new_packed = new_message.pack()
            if packed == new_packed:
                print(" -- Correct")
            else:
                print(" -- Mismatch")
                pack_unpack_inconsistencies_found += 1

        self.assertEqual(pack_unpack_inconsistencies_found, 0)


if __name__ == '__main__':
    unittest.main()
