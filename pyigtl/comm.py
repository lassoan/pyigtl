# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 19:17:05 2015

@author: Daniel Hoyer Iversen
"""

import collections
import logging
import signal
import socket
import socketserver as SocketServer
import struct
import sys
import threading
import time

from .messages import MessageBase

logger = logging.getLogger(__name__)


class OpenIGTLinkBase():
    """Abstract base class for client and server"""

    def __init__(self):
        self._started = False  # server/client started
        self._connected = False  # there is a socket connection that is successfully sending/receiving

        # Flags to request stopping of the thread
        # Accessed from main and communication threads.
        # They are simple Booleans, therefore they are not protected by locking.
        self.communication_thread_stop_requested = False
        self.communication_thread_stopped = True

        # Incoming messages.
        # Only one message is preserved for each device.
        # Accessed from main and communication threads, protected by lock.
        self.incoming_messages = {}
        self.lock_incoming_messages = threading.Lock()

        # Outgoing message queue.
        # Accessed from main and communication threads, protected by lock.
        self.outgoing_messages = collections.deque(maxlen=100)
        self.lock_outgoing_messages = threading.Lock()

    def send_message(self, message, wait=True):
        """Put the message in the outgoing message queue.
        wait: wait until the message is actually sent
        """
        return self._add_message_to_send_queue(message, wait)

    def wait_for_message(self, device_name, timeout=-1):
        """Get the most recent message from the specified device.
        If no message is available yet then wait up to the specified timeout (in seconds).
        If timeout value is reached then the method returns ``None``.
        If timeout is set to negative value then it waits indefinitely.
        """
        start_time = time.time()
        while True:
            with self.lock_incoming_messages:
                if device_name in self.incoming_messages:
                    message = self.incoming_messages.pop(device_name)
                    return message
            if timeout >= 0 and (time.time()-start_time > timeout):
                return None
            time.sleep(0.01)

    def get_latest_messages(self):
        messages = []
        with self.lock_incoming_messages:
            for device_name in self.incoming_messages:
                message = self.incoming_messages[device_name]
                messages.append(message)
            self.incoming_messages = {}
        return messages

    def _add_message_to_send_queue(self, message, wait=False):
        """Returns True if sucessful
        """
        if not isinstance(message, MessageBase) or not message.is_valid:
            logger.warning("Message must be derived from MessageBase class")
            return False
        with self.lock_outgoing_messages:
            self.outgoing_messages.append(message)  # copy.deepcopy(message))
        if wait:
            # wait until queue is empty
            while True:
                with self.lock_outgoing_messages:
                    if not self.outgoing_messages:
                        break
                time.sleep(0.001)
        return True

    def _send_queued_message_from_socket(self, socket):
        # called from the communication thread
        with self.lock_outgoing_messages:
            if not self.outgoing_messages:
                # nothing to send
                return False
            message = self.outgoing_messages.popleft()
            binary_message = message.pack()
        # send
        socket.sendall(binary_message)
        return True

    def _receive_message_from_socket(self, ssocket):
        # called from the communication thread
        header = b""
        received_header_size = 0
        while received_header_size < MessageBase.IGTL_HEADER_SIZE:
            try:
                header += ssocket.recv(MessageBase.IGTL_HEADER_SIZE - received_header_size)
            except socket.timeout:
                # no message received, it is not an error
                return False
            if len(header) == 0:
                return False
            received_header_size = len(header)

        header_fields = MessageBase.parse_header(header)
        body_size = header_fields['body_size']
        message_type = header_fields['message_type']

        body = b""
        received_body_size = 0
        while received_body_size < body_size:
            body += ssocket.recv(body_size - received_body_size)
            if len(body) == 0:
                return False
            received_body_size = len(body)

        message = MessageBase.create_message(message_type)
        if not message:
            # unknown message type
            return False

        message.unpack(header_fields, body)

        with self.lock_incoming_messages:
            self.incoming_messages[message.device_name] = message

        return True

    def is_connected(self):
        return self._connected

    def _communication_error_occurred(self):
        self._connected = False


class OpenIGTLinkServer(SocketServer.TCPServer, OpenIGTLinkBase):

    """ For streaming data over TCP with IGTLink"""
    def __init__(self, port=None, local_server=True, iface=None, start_now=True):
        OpenIGTLinkBase.__init__(self)

        self.port = port

        if iface is None:
            iface = 'eth0'

        if local_server:
            self.host = "127.0.0.1"
        else:
            if sys.platform.startswith('win32'):
                self.host = socket.gethostbyname(socket.gethostname())
            elif sys.platform.startswith('linux'):
                import fcntl  # not available on Windows => pylint: disable=import-error
                soc = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                try:
                    ifname = iface
                    self.host = socket.inet_ntoa(fcntl.ioctl(soc.fileno(), 0x8915, struct.pack('256s', ifname[:15]))[20:24])
                    # http://code.activestate.com/recipes/439094-get-the-ip-address-associated-with-a-network-inter/
                except: # noqa
                    ifname = 'lo'
                    self.host = socket.inet_ntoa(fcntl.ioctl(soc.fileno(), 0x8915, struct.pack('256s', ifname[:15]))[20:24])
            else:
                # the iface can be also an ip address in systems where the previous code won't work
                self.host = iface

        SocketServer.TCPServer.allow_reuse_address = True
        SocketServer.TCPServer.__init__(self, (self.host, self.port), TCPRequestHandler)

        # Register custom signal handler to properly close the socket when the process is killed
        self._previous_signal_handlers = {}
        self._previous_signal_handlers[signal.SIGTERM] = signal.signal(signal.SIGTERM, self._signal_handler)
        self._previous_signal_handlers[signal.SIGINT] = signal.signal(signal.SIGINT, self._signal_handler)

        if start_now:
            self.start()

    def start(self):
        server_thread = threading.Thread(target=self.serve_forever)
        server_thread.daemon = True
        server_thread.start()

        thread = threading.Thread(target=self._print_host_and_port_thread)
        thread.daemon = True
        thread.start()

    def stop(self):
        self._close_server()

    def _signal_handler(self, signum, stackframe):
        """Properly close the server if signal is received"""
        self._close_server()
        self._previous_signal_handlers[signum](signum, stackframe)

    def _close_server(self):
        """Will close connection and shutdown server"""
        self._connected = False

        self.communication_thread_stop_requested = True
        # It may take a while for _print_host_and_port_thread to stop, so don't wait for it

        self.shutdown()  # request stopping of the serving thread (waits until the current request is finished)
        self.server_close()  # clean up the server
        logger.debug("Server closed")

    def _print_host_and_port_thread(self):
        while True:
            # Wait for connection and print a message in every 5 seconds
            while not self._connected:
                if self.communication_thread_stop_requested:
                    logging.info("Client not connected (host: {0}, port: {1})".format(self.host, self.port))
                    break
                time.sleep(5)
            time.sleep(1)
            if self.communication_thread_stop_requested:
                self.communication_thread_stopped = True
                break


class TCPRequestHandler(SocketServer.BaseRequestHandler):
    """
    Help class for OpenIGTLinkServer
    """
    def handle(self):
        self.server._connected = True
        self.request.settimeout(0.01)

        while not self.server.communication_thread_stop_requested:

            try:
                while self.server._receive_message_from_socket(self.request):
                    pass
            except Exception as exp:
                import traceback
                traceback.print_exc()
                logging.error('Error while receiving data: '+str(exp))
                self.server._communication_error_occurred()
                break

            try:
                while self.server._send_queued_message_from_socket(self.request):
                    pass
            except Exception as exp:
                import traceback
                traceback.print_exc()
                logging.error('Error while sending data: '+str(exp))
                self.server._communication_error_occurred()
                break


class OpenIGTLinkClient(OpenIGTLinkBase):
    def __init__(self, host="127.0.0.1", port=18944, start_now=True):
        OpenIGTLinkBase.__init__(self)

        self.socket = None
        self.host = host
        self.port = port

        self._client_thread = threading.Thread(target=self._client_thread_function)
        self._client_thread.daemon = True

        self.lock_client_thread = threading.Lock()

        if start_now:
            self.start()

    def start(self):
        if self._started:
            return
        self._started = True

        self.communication_thread_stop_requested = False
        self.communication_thread_stopped = False
        self._client_thread.start()

    def stop(self):
        if not self._started:
            return
        self._started = False
        self._connected = False

        # Wait for the communication thread to stop
        self.communication_thread_stop_requested = True
        while True:
            if self.communication_thread_stopped:
                break
            time.sleep(0.1)

    def _client_thread_function(self):  # complex function, but clearly separates what runs in a thread => # noqa: C901
        while True:
            if self.communication_thread_stop_requested:
                break

            # Create socket
            if self.socket is None:
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.socket.settimeout(0.01)
                self._connected = False

            # Reconnect if needed
            if not self._connected:
                try:
                    self.socket.connect((self.host, self.port))
                    self._connected = True
                except Exception:
                    self.socket = None
                    time.sleep(0.01)
                    continue

            # Receive messages
            try:
                while self._receive_message_from_socket(self.socket):
                    pass
            except Exception as exp:
                import traceback
                traceback.print_exc()
                logging.error('Error while receiving data: '+str(exp))
                self._communication_error_occurred()

            # Send messages
            try:
                while self._send_queued_message_from_socket(self.socket):
                    pass
            except Exception as exp:
                import traceback
                traceback.print_exc()
                logging.error('Error while sending data: '+str(exp))
                self._communication_error_occurred()

        # Close socket
        if self.socket is not None:
            self.socket.close()
        self.communication_thread_stopped = True
