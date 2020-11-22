# -*- coding: utf-8 -*-

import unittest
import numpy as np

import pyigtl

import time

# class TestServer(unittest.TestCase):
    # """
    # Tests the server
    # """

    # def setUp(self):     # pylint: disable=invalid-name
        # """ things to be run when tests are started. """
        # self.server = pyIGTLink.PyIGTLinkServer(localServer=True)
        # self.server.start()

    # def tearDown(self):  # pylint: disable=invalid-name
        # """ Stop down stuff we started. """
        # self.server.stop()

    # def test_server(self):
        # self.assertFalse(self.server.is_connected())
        # self.assertEqual(self.server.get_ip_adress(), "127.0.0.1")
        # self.assertEqual(self.server.get_port_no(), 18944)

    # def test_add_msgs(self):
        # samples = 500
        # beams = 100
        # k = 0

        # for i in range(10):
            # k = k+1
            # data = np.random.randn(samples, beams)*50+100
            # imageMessage = pyIGTLink.ImageMessage(data)
            # self.assertTrue(self.server.send_message(imageMessage))
        # self.assertFalse(self.server.send_message("invalidPackage"))
        # self.assertFalse(self.server.send_message(pyIGTLink.ImageMessage([1, 2, 3])))

class TestServerClient(unittest.TestCase):
    """
    Tests the server
    """

    def setUp(self):     # pylint: disable=invalid-name
        """ things to be run when tests are started. """
        self.server = pyIGTLink.PyIGTLinkServer(port=18944, localServer=True)
        self.client = pyIGTLink.PyIGTLinkClient(host="127.0.0.1", port=18944)

        self.server.start()
        self.client.start()

    def tearDown(self):  # pylint: disable=invalid-name
        """ Stop down stuff we started. """

        self.server.stop()
        self.client.stop()

    def test_send_messages(self):
        
        time.sleep(1.0)
        
        string = "TEST Message"
        string_message = pyIGTLink.StringMessage("TEST Message", device_name="TestString")
        self.assertTrue(self.server.send_message(string_message))
        
        time.sleep(1.0)

        received_messages = self.client.get_latest_messages()
        self.assertTrue(len(received_messages) > 0)
        for message in received_messages:
          self.server.send_message(message)
        
        time.sleep(1.0)
        
        received_messages = self.server.get_latest_messages()
        self.assertTrue(len(received_messages) > 0)
        messageReceived = False
        for message in received_messages:
          if message._device_name == string_message._device_name:
            self.assertEqual(string_message._string, message._string)
            messageReceived = True
        self.assertTrue(messageReceived)


class TestMsg(unittest.TestCase):
    """
    Tests the msg
    """

    def test_header_msg(self):
        msg = pyIGTLink.MessageBase()
        self.assertEqual(len(msg.get_binary_message()), IGTL_HEADER_SIZE)

    def test_image_msg(self):
        data = np.random.randn(500, 100)*50+100
        msg = pyIGTLink.ImageMessage(data)
        self.assertEqual(len(msg.get_binary_body()), msg.get_body_pack_size())
        self.assertEqual(len(msg.get_binary_message()), msg.get_body_pack_size() + IGTL_HEADER_SIZE)

    def test_transform_msg(self):
        data = np.random.randn(4, 4)
        msg = pyIGTLink.TransformMessage(data)
        self.assertEqual(len(msg.get_binary_body()), msg.get_body_pack_size())
        self.assertEqual(len(msg.get_binary_message()), msg.get_body_pack_size() + IGTL_HEADER_SIZE)


if __name__ == '__main__':
    unittest.main()
