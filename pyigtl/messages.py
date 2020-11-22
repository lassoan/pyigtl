# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 19:17:05 2015

@author: Daniel Hoyer Iversen
"""

from __future__ import print_function
import crcmod
import numpy as np
import signal
import collections
import socket
import sys
import struct
import threading
import time
import string

if sys.version_info >= (3, 0):
    import socketserver as SocketServer
else:
    import SocketServer


# http://openigtlink.org/protocols/v2_header.html
class MessageBase(object):
    """OpenIGTLink message base class"""

    # message_type_to_class_constructor = {
    #     "TRANSFORM": TransformMessage,
    #     "IMAGE": ImageMessage,
    #     "STRING": StringMessage,
    # }

    def __init__(self, timestamp=None, device_name=None):

        # The device name field contains an ASCII character string specifying the name of the the message.
        self.device_name = device_name if (device_name is not None) else ''

        # The timestamp field contains a 64-bit timestamp indicating when the data is generated.
        # Please refer http://openigtlink.org/protocols/v2_timestamp.html for the format of the 64-bit timestamp.
        if timestamp is None:
            self.timestamp = time.time() * 1000
        else:
            self.timestamp = timestamp

        # Valid content is set in the image
        self._valid_message = False

        # Version number The version number field specifies the header format version.
        # Please note that this is different from the protocol version.
        # OpenIGTLink 1 and 2 uses headerVersion=1.
        # OpenIGTLink 3 uses headerVersion=2 by default, but can communicate with legacy client using headerVersion=1 messages.
        self.header_version = 1

        # The type field is an ASCII character string specifying the type of the data contained in
        # the message body e.g. TRANSFORM. The length of the type name must be within 12 characters.
        self._message_type = ""

        self.message_id = 0

        # Key/value string pairs in a map, defining custom metadata (only supported with headerVersion=2).
        self.metadata = {}

        self._endian = ">"  # big-endian

    @property
    def is_valid(self):
        """
        Message has valid content.
        """
        return self._valid_message

    @property
    def message_type(self):
        """
        Message type (IMAGE, TRANSFORM, ...).
        """
        return self._message_type

    def __str__(self):
        output = f'{self._message_type} message:'
        output += f'\n  Device name: {self.device_name}'
        output += f'\n  Timestamp: {self.timestamp}'
        output += f'\n  Header version: {self.header_version}'
        content = self.content_asstring()
        if content:
            output += '\n  '+content.replace('\n', '\n  ')
        metadata = self.metadata_asstring()
        if metadata:
            output += '\n  Metadata:\n    ' + metadata.replace('\n', '\n    ')
        return output

    def content_asstring(self):
        return ''

    def metadata_asstring(self):
        if not self.metadata:
            return ''
        return '\n'.join([f"{item[0]}: {item[1]}" for item in self.metadata.items()])

    def pack(self):
        """Return a buffer that contains the entire message as binary data"""

        # Pack metadata
        binary_metadata_header = b""
        binary_metadata_body = b""
        if self.header_version>1:
            for key, value in items(self.metadata):
                encoded_key = key.encode('ascii')
                encoded_value = value.encode('utf-8')  # TODO: use the encoding specified in self._encoding
                binary_metadata_header += struct.pack(self._endian+"H", len(encoded_key))
                binary_metadata_header += struct.pack(self._endian+"H", self._encoding)
                binary_metadata_header += struct.pack(self._endian+"I", len(encoded_value))
                binary_metadata_body += encoded_key
                binary_metadata_body += encoded_value

        # Pack extended header
        binary_extended_header = b""
        if self.header_version > 1:
            IGTL_EXTENDED_HEADER_SIZE = 12  # OpenIGTLink extended header has a fixed size
            binary_extended_header += struct.pack(self._endian+"H", IGTL_EXTENDED_HEADER_SIZE)
            binary_extended_header += struct.pack(self._endian+"H", len(binary_metadata_header))
            binary_extended_header += struct.pack(self._endian+"I", len(binary_metadata_body))
            binary_extended_header += struct.pack(self._endian+"I", self.message_id)

        # Pack content and assemble body
        binary_body = binary_extended_header + self._pack_content() + binary_metadata_header + binary_metadata_body

        # Pack header
        body_length = len(binary_body)
        crc = CRC64(binary_body)
        _timestamp1 = int(self.timestamp / 1000)
        _timestamp2 = _igtl_nanosec_to_frac(int((self.timestamp / 1000.0 - _timestamp1)*10**9))
        binary_header = struct.pack(self._endian+"H", self.header_version)
        binary_header += struct.pack(self._endian+"12s", self._message_type.encode('utf-8'))
        binary_header += struct.pack(self._endian+"20s", self.device_name.encode('utf-8'))
        binary_header += struct.pack(self._endian+"II", _timestamp1, _timestamp2)
        binary_header += struct.pack(self._endian+"Q", body_length)
        binary_header += struct.pack(self._endian+"Q", crc)

        # Assemble and return packed message
        return binary_header + binary_body

    @staticmethod
    def parse_header(header):
        s = struct.Struct('> H 12s 20s II Q Q')  # big-endian
        values = s.unpack(header)
        header_fields = {}
        header_fields['header_version'] = values[0]

        header_fields['message_type'] = values[1].decode().rstrip(' \t\r\n\0')
        header_fields['device_name'] = values[2].decode().rstrip(' \t\r\n\0')

        seconds = float(values[3])
        frac_of_second = int(values[4])
        nanoseconds = float(_igtl_frac_to_nanosec(frac_of_second))
        header_fields['timestamp'] = seconds + (nanoseconds * 1e-9)

        header_fields['body_size'] = values[5]
        return header_fields

    @staticmethod
    def create_message(message_type):
        if message_type not in message_type_to_class_constructor:
            return None
        return message_type_to_class_constructor[message_type]()

    def unpack(self, header_fields, body):
        """Set message content from parsed message header fields and binary message body.
        """

        # Header is already unpacked, just save the fields
        self.header_version = header_fields['header_version']
        self._message_type = header_fields['message_type']
        self.device_name = header_fields['device_name']
        self.timestamp = header_fields['timestamp']

        # Unpack extended header
        if self.header_version > 1:
            # Extended header is present
            s = struct.Struct('> H H I I')  # big-endian
            IGTL_EXTENDED_HEADER_SIZE = 12  # OpenIGTLink extended header has a fixed size
            values = s.unpack(body[:IGTL_EXTENDED_HEADER_SIZE])
            extended_header_size = values[0]
            metadata_header_size = values[1]
            metadata_body_size = values[2]
            self.message_id = values[3]
        else:
            # No extended header
            extended_header_size = 0
            metadata_header_size = 0
            metadata_body_size = 0
            self.message_id = 0

        # Unpack metadata
        self.metadata = {}
        metadata_size = metadata_header_size + metadata_body_size
        if metadata_size > 0:
            metadata = body[-metadata_size:]
            index_count = struct.Struct('> H').unpack(metadata[:2])[0]
            read_offset = 2+index_count*8
            for index in range(index_count):
                key_size, value_encoding, value_size = struct.Struct('> H H I').unpack(metadata[index*8+2:index*8+10])
                key = metadata[read_offset:read_offset+key_size].decode()
                read_offset += key_size
                # TODO: decode based on value_encoding, we assume UTF8 now
                value = metadata[read_offset:read_offset+value_size].decode()
                read_offset += value_size
                self.metadata[key] = value

        # Unpack content
        if metadata_size > 0:
            self._unpack_content(body[extended_header_size:-metadata_size])
        else:
            self._unpack_content(body[extended_header_size:])

    def _unpack_content(self, content):
        # no message content by default
        pass

    def _pack_content(self):
        # no message content by default
        return b""


# http://openigtlink.org/protocols/v2_image.html
class ImageMessage(MessageBase):
    def __init__(self, image=None, spacing=None, timestamp=None, device_name=None):
        """
        Image message
        image: image data
        spacing: spacing in mm
        timestamp: milliseconds since 1970
        device_name: name of the image
        """

        MessageBase.__init__(self, timestamp=timestamp, device_name=device_name)

        self._message_type = "IMAGE"

        if image is not None:
            try:
                self._image = np.asarray(image)
            except Exception as exp:
                raise ValueError('Invalid image, cannot get it as a numpy array: ' + str(exp))
            image_dimension = len(self._image.shape)
            if image_dimension < 1 or image_dimension > 4:
                raise ValueError("Invalid image, dimension must be between 1 and 4")
        else:
            self.image=None

        self.spacing = spacing

        self._valid_message = True

# Only int8 is supported now
#        if self._image.dtype == np.int8:
#            self._datatype_s = 2
#            self._format_data = "b"
#        elif self._image.dtype == np.uint8:
#            self._datatype_s = 3
#            self._format_data = "B"
#        elif self._image.dtype == np.int16:
#            self._datatype_s = 4
#            self._format_data = "h"
#        elif self._image.dtype == np.uint16:
#            self._datatype_s = 5
#            self._format_data = "H"
#        elif self._image.dtype == np.int32:
#            self._datatype_s = 6
#            self._format_data = "i"
#        elif self._image.dtype == np.uint32:
#            self._datatype_s = 7
#            self._format_data = "I"
#        elif self._image.dtype == np.float32:
#            self._datatype_s = 10
#            self._format_data = "f"
#        elif self._image.dtype == np.float64:
#            self._datatype_s = 11
#            self._format_data = "f"
#        else:
#            pass
        self._image = np.array(self._image, dtype=np.uint8)
        self._datatype_s = 3
        self._format_data = "B"

        self._spacing = spacing
        self._matrix = np.identity(4)  # A matrix representing the origin and the orientation of the image.

    def __str__(self):
        properties = []
        if self._valid_message:
            properties.append(f'Device name: {self.device_name}')
            properties.append(f'Timestamp: {self.timestamp}')
            properties.append('Matrix:\n  {0}'.format(str(self._matrix).replace('\n', '\n  ')))
        return f'{self._message_type} message\n  '+'\n  '.join(properties)

    def _pack_content(self):
        binary_message = struct.pack(self._endian+"H", self.header_version)
        # Number of Image Components (1:Scalar, >1:Vector). (NOTE: Vector data is stored fully interleaved.)
        number_of_components = self._image.shape[3] if len(self._image.shape) > 3 else 1
        binary_message += struct.pack(self._endian+"B", number_of_components)
        binary_message += struct.pack(self._endian+"B", self._datatype_s)

        if self._image.dtype.byteorder == "<":
            byteorder = "F"
            binary_message += struct.pack(self._endian+"B", 2)  # Endian for image data (1:BIG 2:LITTLE)
            # (NOTE: values in image header is fixed to BIG endian)
        else:
            self._image.dtype.byteorder == ">"
            byteorder = "C"
            binary_message += struct.pack(self._endian+"B", 1)  # Endian for image data (1:BIG 2:LITTLE)
            # (NOTE: values in image header is fixed to BIG endian)

        binary_message += struct.pack(self._endian+"B", 1)  # image coordinate (1:RAS 2:LPS)

        number_of_slices = self._image.shape[2] if len(self._image.shape) > 2 else 1
        binary_message += struct.pack(self._endian+"H", number_of_slices)
        number_of_rows = self._image.shape[1] if len(self._image.shape) > 1 else 1
        binary_message += struct.pack(self._endian+"H", number_of_rows)
        binary_message += struct.pack(self._endian+"H", self._image.shape[0])

        origin = np.zeros(3)
        norm_i = np.zeros(3)
        norm_j = np.zeros(3)
        norm_k = np.zeros(3)
        for i in range(3):
            norm_i[i] = self._matrix[i][0]
            norm_j[i] = self._matrix[i][1]
            norm_k[i] = self._matrix[i][2]
            origin[i] = self._matrix[i][3]

        spacing = [1.0, 1.0, 1.0]
        if self._spacing:
            for i in range(3):
                if len(self._spacing) > i:
                    spacing[i] = self._spacing[i]

        binary_message += struct.pack(self._endian+"f", norm_i[0] * spacing[0])
        binary_message += struct.pack(self._endian+"f", norm_i[1] * spacing[0])
        binary_message += struct.pack(self._endian+"f", norm_i[2] * spacing[0])
        binary_message += struct.pack(self._endian+"f", norm_j[0] * spacing[1])
        binary_message += struct.pack(self._endian+"f", norm_j[1] * spacing[1])
        binary_message += struct.pack(self._endian+"f", norm_j[2] * spacing[1])
        binary_message += struct.pack(self._endian+"f", norm_k[0] * spacing[2])
        binary_message += struct.pack(self._endian+"f", norm_k[1] * spacing[2])
        binary_message += struct.pack(self._endian+"f", norm_k[2] * spacing[2])
        binary_message += struct.pack(self._endian+"f", origin[0])
        binary_message += struct.pack(self._endian+"f", origin[1])
        binary_message += struct.pack(self._endian+"f", origin[2])

        binary_message += struct.pack(self._endian+"H", 0)      # Starting index of subvolume
        binary_message += struct.pack(self._endian+"H", 0)      # Starting index of subvolume
        binary_message += struct.pack(self._endian+"H", 0)      # Starting index of subvolume

        binary_message += struct.pack(self._endian+"H", self._image.shape[0])  # number of pixels of subvolume
        binary_message += struct.pack(self._endian+"H", self._image.shape[1])
        if len(self._image.shape) > 2:
            binary_message += struct.pack(self._endian+"H", self._image.shape[2])
        else:
            binary_message += struct.pack(self._endian+"H", 1)

        binary_message += self._image.tostring(byteorder)  # struct.pack(fmt,*data)
        
        return binary_message

    def _unpack_content(self, content):
        header_portion_len = 12 + (12 * 4) + 12
        s_head = struct.Struct('> H B B B B H H H f f f f f f f f f f f f H H H H H H')
        values_header = s_head.unpack(content[0:header_portion_len])

        numberOfComponents = values_header[1]
        endian = values_header[3]
        size_x = values_header[23]
        size_y = values_header[24]
        size_z = values_header[25]

        if endian == 2:
            endian = '<'
        else:
            endian = '>'

        values_img = body[header_portion_len:]
        dt = np.dtype(np.uint8)
        dt = dt.newbyteorder(endian)
        data = np.frombuffer(values_img, dtype=dt)

        self._image = np.reshape(data, [size_z, size_y, size_x, numberOfComponents])


class TransformMessage(MessageBase):
    def __init__(self, transform_matrix=None, timestamp=None, device_name=None):
        """
        Transform package
        transform_matrix: 4x4 homogeneous transformation matrix as numpy array
        timestamp: milliseconds since 1970
        device_name: name of the tool
        """

        MessageBase.__init__(self, timestamp=timestamp, device_name=device_name)
        self._message_type = "TRANSFORM"

        if transform_matrix is not None:
            try:
                self._matrix = np.asarray(transform_matrix, dtype=np.float32)
            except Exception as exp:
                raise ValueError("Invalid transorm_matrix")
            matrix_dimension = len(self._matrix.shape)
            if matrix_dimension != 2:
                raise ValueError("Invalid transorm_matrix dimension {0}".format(matrix_dimension))
        else:
            self._matrix = np.eye(4, dtype=np.float32)

# transforms are floats
#        if self._matrix.dtype == np.int8:
#            self._datatype_s = 2
#            self._format_data = "b"
#        elif self._matrix.dtype == np.uint8:
#            self._datatype_s = 3
#            self._format_data = "B"
#        elif self._matrix.dtype == np.int16:
#            self._datatype_s = 4
#            self._format_data = "h"
#        elif self._matrix.dtype == np.uint16:
#            self._datatype_s = 5
#            self._format_data = "H"
#        elif self._matrix.dtype == np.int32:
#            self._datatype_s = 6
#            self._format_data = "i"
#        elif self._matrix.dtype == np.uint32:
#            self._datatype_s = 7
#            self._format_data = "I"
#        elif self._matrix.dtype == np.float32:
#            self._datatype_s = 10
#            self._format_data = "f"
#        elif self._matrix.dtype == np.float64:
#            self._datatype_s = 11
#            self._format_data = "f"
#        else:
#            pass
        self._datatype_s = 10
        self._format_data = "f"

        self._valid_message = True

    def content_asstring(self):
        properties = []
        if self._valid_message:
            properties.append('Matrix:\n  {0}'.format(str(self._matrix).replace('\n', '\n  ')))
        return '\n'.join(properties)

    def _pack_content(self):
        binary_content = struct.pack(self._endian + "f", self._matrix[0, 0])  # R11
        binary_content += struct.pack(self._endian + "f", self._matrix[1, 0])  # R21
        binary_content += struct.pack(self._endian + "f", self._matrix[2, 0])  # R31

        binary_content += struct.pack(self._endian + "f", self._matrix[0, 1])  # R12
        binary_content += struct.pack(self._endian + "f", self._matrix[1, 1])  # R22
        binary_content += struct.pack(self._endian + "f", self._matrix[2, 1])  # R32

        binary_content += struct.pack(self._endian + "f", self._matrix[0, 2])  # R13
        binary_content += struct.pack(self._endian + "f", self._matrix[1, 2])  # R23
        binary_content += struct.pack(self._endian + "f", self._matrix[2, 2])  # R33

        binary_content += struct.pack(self._endian + "f", self._matrix[0, 3])  # TX
        binary_content += struct.pack(self._endian + "f", self._matrix[1, 3])  # TY
        binary_content += struct.pack(self._endian + "f", self._matrix[2, 3])  # TZ

        return binary_content

    def _unpack_content(self, content):
        s = struct.Struct('> f f f f f f f f f f f f')
        values = s.unpack(content)
        self._matrix = np.asarray([[values[0], values[3], values[6], values[9]],
                                   [values[1], values[4], values[7], values[10]],
                                   [values[2], values[5], values[8], values[11]],
                                   [0, 0, 0, 1]])

class StringMessage(MessageBase):
    def __init__(self, string=None, timestamp=None, device_name='', encoding=3):

        MessageBase.__init__(self)
        self._valid_message = True
        self._message_type = "STRING"
        if timestamp:
            self.timestamp = timestamp
        self.device_name = device_name
        self._encoding = encoding  # Character encoding type as MIBenum value. Default=3 (US-ASCII).
        if string is not None:
            self._string = string
        else:
            self.string = ""

    def _pack_content(self):
        binary_content = struct.pack(self._endian+"H", self._encoding)
        binary_content += struct.pack(self._endian+"H", len(self._string))
        binary_content += self._string.encode("utf-8")
        return binary_content

    def _unpack_content(self, content):
        header_portion_len = 2 + 2
        s_head = struct.Struct('> H H')

        values_header = s_head.unpack(content)
        self._encoding = values_header[0]
        stringLength = values_header[1]
        self._string = str(message[header_portion_len:header_portion_len + stringLength])


# http://slicer-devel.65872.n3.nabble.com/OpenIGTLinkIF-and-CRC-td4031360.html
CRC64 = crcmod.mkCrcFun(0x142F0E1EBA9EA3693, rev=False, initCrc=0x0000000000000000, xorOut=0x0000000000000000)

# https://github.com/openigtlink/OpenIGTLink/blob/cf9619e2fece63be0d30d039f57b1eb4d43b1a75/Source/igtlutil/igtl_util.c#L168
def _igtl_nanosec_to_frac(nanosec):
    base = 1000000000  # 10^9
    mask = 0x80000000
    r = 0x00000000
    while mask:
        base += 1
        base >>= 1
        if (nanosec >= base):
            r |= mask
            nanosec = nanosec - base
        mask >>= 1
    return r

# https://github.com/openigtlink/OpenIGTLink/blob/cf9619e2fece63be0d30d039f57b1eb4d43b1a75/Source/igtlutil/igtl_util.c#L193
def _igtl_frac_to_nanosec(frac):
    base = 1000000000 # 10^9
    mask = 0x80000000
    r = 0x00000000
    while mask:
        base += 1
        base >>= 1
        r += base if (frac & mask) else 0
        mask >>= 1
    return r

message_type_to_class_constructor = {
        "TRANSFORM": TransformMessage,
        "IMAGE": ImageMessage,
        "STRING": StringMessage,
    }
