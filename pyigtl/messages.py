# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 19:17:05 2015

@author: Daniel Hoyer Iversen
"""

import crcmod
import logging
import numpy as np
import struct
import time

logger = logging.getLogger(__name__)


# http://openigtlink.org/protocols/v2_header.html
# OpenIGTLink uses big-endian for all numeric value storage (therefore ">" is added to all struct descriptors).
class MessageBase(object):
    """OpenIGTLink message base class"""

    IANA_CHARACTER_SET_ASCII = 3
    IANA_CHARACTER_SET_UTF8 = 106

    IGTL_HEADER_SIZE = 58

    def __init__(self, timestamp=None, device_name=None):

        # The device name field contains an ASCII character string specifying the name of the the message.
        self.device_name = device_name if (device_name is not None) else ''

        # The timestamp field contains a 64-bit timestamp indicating when the data is generated.
        # Please refer http://openigtlink.org/protocols/v2_timestamp.html for the format of the 64-bit timestamp.
        if timestamp is None:
            self.timestamp = time.time()
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

    @staticmethod
    def encode_text(text):
        """Encode string as ASCII if possible, UTF8 otherwise"""
        try:
            encoded_text = text.encode('ascii')
            encoding = MessageBase.IANA_CHARACTER_SET_ASCII
        except UnicodeDecodeError:
            encoded_text = text.encode('utf8')
            encoding = MessageBase.IANA_CHARACTER_SET_UTF8
        return encoded_text, encoding

    @staticmethod
    def decode_text(encoded_text, encoding):
        """Get string by decoding from ASCII or UTF8"""
        if encoding == MessageBase.IANA_CHARACTER_SET_ASCII:
            text = encoded_text.decode('ascii')
        elif encoding == MessageBase.IANA_CHARACTER_SET_UTF8:
            text = encoded_text.decode('utf8')
        else:
            raise("Unsupported encoding: "+str(encoding))
        return text

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

        if self.metadata and self.header_version < 2:
            logger.warning("Metadata will not be packed, because message header version = 1 (metadata can only be sent in header version = 2).")

        # Pack metadata
        binary_metadata_header = b""
        binary_metadata_body = b""
        if self.header_version > 1:
            for key, value in self.metadata.items():
                encoded_key = key.encode('utf8')  # use UTF8 for all strings that specified without encoding
                encoded_value, encoding = MessageBase.encode_text(value)
                binary_metadata_header += struct.pack("> H", len(encoded_key))
                binary_metadata_header += struct.pack("> H", encoding)
                binary_metadata_header += struct.pack("> I", len(encoded_value))
                binary_metadata_body += encoded_key
                binary_metadata_body += encoded_value

        # Pack extended header
        binary_extended_header = b""
        if self.header_version > 1:
            IGTL_EXTENDED_HEADER_SIZE = 12  # OpenIGTLink extended header has a fixed size
            binary_extended_header += struct.pack("> H", IGTL_EXTENDED_HEADER_SIZE)
            binary_extended_header += struct.pack("> H", len(binary_metadata_header))
            binary_extended_header += struct.pack("> I", len(binary_metadata_body))
            binary_extended_header += struct.pack("> I", self.message_id)

        # Pack content and assemble body
        binary_body = binary_extended_header + self._pack_content() + binary_metadata_header + binary_metadata_body

        # Pack header
        body_length = len(binary_body)
        crc = CRC64(binary_body)
        _timestamp1 = int(self.timestamp)
        _timestamp2 = _igtl_nanosec_to_frac(int((self.timestamp - _timestamp1)*10**9))
        binary_header = struct.pack("> H", self.header_version)
        binary_header += struct.pack("> 12s", self._message_type.encode('utf8'))
        binary_header += struct.pack("> 20s", self.device_name.encode('utf8'))
        binary_header += struct.pack("> II", _timestamp1, _timestamp2)
        binary_header += struct.pack("> Q", body_length)
        binary_header += struct.pack("> Q", crc)

        # Assemble and return packed message
        return binary_header + binary_body

    @staticmethod
    def parse_header(header):
        s = struct.Struct('> H 12s 20s II Q Q')
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
            s = struct.Struct('> H H I I')
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
                encoded_value = metadata[read_offset:read_offset+value_size]
                read_offset += value_size
                self.metadata[key] = MessageBase.decode_text(encoded_value, value_encoding)

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

    scalar_types = {
        2: np.int8, 3: np.uint8,
        4: np.int16, 5: np.uint16,
        6: np.int32, 7: np.uint32,
        10: np.float32, 11: np.float64,
        }

    def __init__(self, image=None, ijk_to_world_matrix=None, world_coordinate_system=None, timestamp=None, device_name=None):
        """
        Image message

        image: image data. Axes are: (k, j, i, components).
            If a voxel array is created with (i, j, k) axes then it can be converted to
            (k, j, i) array by calling: ``voxels = np.transpose(voxels, axes=(2,1,0))``

        world_coordinate_system: lps (default) or ras

        ijk_to_world_matrix: specifies origin, spacing, and axis directions

        timestamp: seconds since 1970

        device_name: name of the image
        """

        MessageBase.__init__(self, timestamp=timestamp, device_name=device_name)

        self._message_type = "IMAGE"

        if image is not None:
            try:
                self.image = np.asarray(image)
            except Exception as exp:
                raise ValueError('Invalid image, cannot get it as a numpy array: ' + str(exp))
            image_dimension = len(self.image.shape)
            if image_dimension < 1 or image_dimension > 4:
                raise ValueError("Invalid image, dimension must be between 1 and 4")
        else:
            self.image = None

        self.world_coordinate_system = world_coordinate_system if world_coordinate_system is not None else "lps"

        if ijk_to_world_matrix is not None:
            self.ijk_to_world_matrix = np.asarray(ijk_to_world_matrix, dtype=np.float32)
        else:
            self.ijk_to_world_matrix = np.eye(4)

        self._valid_message = True

    def content_asstring(self):
        properties = []
        properties.append(f'Image size: {self.image.shape}')
        properties.append('Matrix:\n  {0}'.format(str(self.ijk_to_world_matrix).replace('\n', '\n  ')))
        properties.append(f'World coordinate system: {self.world_coordinate_system}')
        return '\n'.join(properties)

    def _pack_content(self):
        # Image header version is always 1
        binary_message = struct.pack("> H", 1)

        # Number of Image Components (1:Scalar, >1:Vector). (NOTE: Vector data is stored fully interleaved.)
        image_dim = len(self.image.shape)
        number_of_components = 1
        if image_dim == 4:
            image_size = np.array([self.image.shape[2], self.image.shape[1], self.image.shape[0]])
            number_of_components = self.image.shape[3]
        elif image_dim == 3:
            image_size = np.array([self.image.shape[2], self.image.shape[1], self.image.shape[0]])
        elif image_dim == 2:
            image_size = np.array([1, self.image.shape[1], self.image.shape[0]])
        elif image_dim == 1:
            image_size = np.array([1, 1, self.image.shape[0]])
        else:
            raise ValueError("Image dimension must be between 1 and 4")

        binary_message += struct.pack("> B", number_of_components)

        # Scalar type
        igtl_scalar_type = [igtl_type for igtl_type, numpy_type in ImageMessage.scalar_types.items() if self.image.dtype == numpy_type][0]
        binary_message += struct.pack("> B", igtl_scalar_type)

        # Endiannes of image data (1:BIG 2:LITTLE)
        # byteorder returns '=' for native, '<' for little-endian, '>' for big-endian, and '|' if not applicable.
        # Since Intel systems are little-endian, we use little-endian for everything else but '>'.
        # Note: all numerical values in OpenIGTLink messages are still stored as big endian.
        igtl_endianness = 1 if self.image.dtype.byteorder == ">" else 2
        binary_message += struct.pack("> B", igtl_endianness)

        # World coordinate system (1:RAS 2:LPS)
        binary_message += struct.pack("> B", 1 if self.world_coordinate_system == 'ras' else 2)

        # Image size
        binary_message += struct.pack("> H H H", image_size[0], image_size[1], image_size[2])

        spacing = np.ones(3)
        for axis_index in range(3):
            spacing[axis_index] = np.linalg.norm(self.ijk_to_world_matrix[:, axis_index])

        # Image to world transform

        # OpenIGTLink describes image position by specifying image center
        # instead of image origin.
        image_center_ijk = np.array([
            (image_size[0] - 1) / 2.0,
            (image_size[1] - 1) / 2.0,
            (image_size[2] - 1) / 2.0,
            1.0])
        image_center_world = self.ijk_to_world_matrix.dot(image_center_ijk)

        binary_message += struct.pack("> f f f f f f f f f f f f",
                                      *self.ijk_to_world_matrix[0:3, 0],  # tx, ty, tz
                                      *self.ijk_to_world_matrix[0:3, 1],  # sx, sy, sz
                                      *self.ijk_to_world_matrix[0:3, 2],  # nx, ny, nz
                                      *image_center_world[0:3])  # cx, cy, cz

        # Starting index and size of subvolume
        subvolume_size = image_size
        binary_message += struct.pack("> H H H H H H",
                                      0, 0, 0,
                                      subvolume_size[0], subvolume_size[1], subvolume_size[2])

        # Voxels
        binary_message += self.image.tobytes()

        return binary_message

    def _unpack_content(self, content):
        header_portion_len = 12 + (12 * 4) + 12
        s_head = struct.Struct('> H B B B B H H H f f f f f f f f f f f f H H H H H H')
        values_header = s_head.unpack(content[0:header_portion_len])

        numberOfComponents = values_header[1]
        igtl_scalar_type = values_header[2]
        igtl_endianness = values_header[3]

        world_coordinate_system = values_header[4]
        self.world_coordinate_system = 'ras' if world_coordinate_system == 1 else 'lps'

        image_size = np.array(values_header[5:8])

        self.ijk_to_world_matrix = np.eye(4)
        self.ijk_to_world_matrix[0:3, 0] = values_header[8:11]  # t
        self.ijk_to_world_matrix[0:3, 1] = values_header[11:14]  # s
        self.ijk_to_world_matrix[0:3, 2] = values_header[14:17]  # n
        self.ijk_to_world_matrix[0:3, 3] = values_header[17:20]  # c

        # OpenIGTLink describes image position by specifying image center
        # instead of image origin.
        image_origin_ijk = np.array([
            -(image_size[0] - 1) / 2.0,
            -(image_size[1] - 1) / 2.0,
            -(image_size[2] - 1) / 2.0,
            1.0])
        image_origin_world = self.ijk_to_world_matrix.dot(image_origin_ijk)
        self.ijk_to_world_matrix[0:3, 3] = image_origin_world[0:3]

        # Subvolume
        subvolume_start_index = values_header[20:23]
        subvolume_size = values_header[23:26]
        if subvolume_start_index != (0, 0, 0) or (subvolume_size != image_size).any():
            # subvolumes are rarely sent, just throw an error instead of implementing it
            raise NotImplementedError("Subvolume receving is not implemented")

        dt = np.dtype(ImageMessage.scalar_types[igtl_scalar_type])
        # Endiannes of image data (1:BIG 2:LITTLE)
        dt = dt.newbyteorder('<' if igtl_endianness == 2 else '>')
        data = np.frombuffer(content[header_portion_len:], dtype=dt)

        if numberOfComponents == 1:
            self.image = np.reshape(data, [image_size[2], image_size[1], image_size[0]])
        else:
            self.image = np.reshape(data, [image_size[2], image_size[1], image_size[0], numberOfComponents])


class TransformMessage(MessageBase):
    def __init__(self, matrix=None, timestamp=None, device_name=None):
        """
        Transform package
        matrix: 4x4 homogeneous transformation matrix as numpy array
        timestamp: milliseconds since 1970
        device_name: name of the tool
        """

        MessageBase.__init__(self, timestamp=timestamp, device_name=device_name)
        self._message_type = "TRANSFORM"

        if matrix is not None:
            try:
                self.matrix = np.asarray(matrix, dtype=np.float32)
            except Exception:
                raise ValueError("Invalid transform matrix (must be convertible to numpy array)")
            matrix_dimension = len(self.matrix.shape)
            if matrix_dimension != 2:
                raise ValueError("Invalid transorm matrix dimension {0} (2 is required)".format(matrix_dimension))
            if self.matrix.shape != (4, 4):
                raise ValueError("Invalid transorm matrix shape {0} (4x4 is required)".format(self.matrix.shape))
        else:
            self.matrix = np.eye(4, dtype=np.float32)

        self._valid_message = True

    def content_asstring(self):
        return 'Matrix:\n  {0}'.format(str(self.matrix).replace('\n', '\n  '))

    def _pack_content(self):
        s = struct.Struct('> f f f f f f f f f f f f')
        binary_content = s.pack(
            self.matrix[0, 0],  # R11
            self.matrix[1, 0],  # R21
            self.matrix[2, 0],  # R31
            self.matrix[0, 1],  # R12
            self.matrix[1, 1],  # R22
            self.matrix[2, 1],  # R32
            self.matrix[0, 2],  # R13
            self.matrix[1, 2],  # R23
            self.matrix[2, 2],  # R33
            self.matrix[0, 3],  # TX
            self.matrix[1, 3],  # TY
            self.matrix[2, 3]  # TZ
            )
        return binary_content

    def _unpack_content(self, content):
        s = struct.Struct('> f f f f f f f f f f f f')
        values = s.unpack(content)
        self.matrix = np.asarray([[values[0], values[3], values[6], values[9]],
                                  [values[1], values[4], values[7], values[10]],
                                  [values[2], values[5], values[8], values[11]],
                                  [0, 0, 0, 1]])


class StringMessage(MessageBase):
    def __init__(self, string=None, timestamp=None, device_name=None):
        MessageBase.__init__(self, timestamp=timestamp, device_name=device_name)
        self._message_type = "STRING"
        if string is not None:
            self.string = string
        else:
            self.string = ""
        self._valid_message = True

    def content_asstring(self):
        return 'String: ' + self.string

    def _pack_content(self):
        encoded_string, encoding = MessageBase.encode_text(self.string)
        binary_content = struct.pack("> H", encoding)
        binary_content += struct.pack("> H", len(encoded_string))
        binary_content += encoded_string
        return binary_content

    def _unpack_content(self, content):
        header_portion_len = 2 + 2
        values_header = struct.Struct('> H H').unpack(content[:header_portion_len])
        encoding = values_header[0]
        string_length = values_header[1]
        encoded_string = content[header_portion_len:header_portion_len + string_length]
        self.string = MessageBase.decode_text(encoded_string, encoding)


class PointMessage(MessageBase):
    def __init__(self, positions=None, names=None, rgba_colors=None, diameters=None, groups=None, owners=None,
                 timestamp=None, device_name=None):
        """
        positions: 3-element vector (for 1 point) or Nx3 matrix (for N points)
        """
        MessageBase.__init__(self, timestamp=timestamp, device_name=device_name)
        self._message_type = "POINT"
        self.positions = positions
        self.names = names
        self.rgba_colors = rgba_colors
        self.diameters = diameters
        self.groups = groups
        self.owners = owners
        self._valid_message = True

    def content_asstring(self):
        items = []
        name_array, group_array, rgba_array, xyz_array, diameter_array, owner_array = self._get_properties_as_arrays()
        point_count = len(name_array)
        for point_index in range(point_count):
            item = f"Point {point_index+1}: name: '{name_array[point_index]}'"
            if group_array[point_index]:
                item += f", group: '{group_array[point_index]}'"
            xyz = xyz_array[point_index]
            item += f", xyz: [{xyz[0]}, {xyz[1]}, {xyz[2]}]"
            rgba = rgba_array[point_index]
            item += f", rgba: [{rgba[0]}, {rgba[1]}, {rgba[2]}, {rgba[3]}]"
            item += f", diameter: {diameter_array[point_index]}"
            if owner_array[point_index]:
                item += f", owner: '{owner_array[point_index]}'"
            items.append(item)

        return "\n".join(items)

    @staticmethod
    def _get_string_property_as_array(string_value_or_array, point_count, label):
        if not string_value_or_array:
            return [''] * point_count
        elif isinstance(string_value_or_array, str):
            return [string_value_or_array] * point_count
        elif len(string_value_or_array) == point_count:
            return string_value_or_array
        else:
            raise ValueError(f"Point {label} must be either a string or list of strings with same number of items as positions")

    def _get_rgba_property_as_array(self, point_count):
        # rgba (4xN)
        if not self.rgba_colors:
            return [[255, 255, 0, 255]] * point_count
        else:
            try:
                rgba_array = np.array(self.rgba_colors, dtype=np.uint8)
                if len(rgba_array.shape) == 2 and rgba_array.shape[1] == 4 and rgba_array.shape[0] == point_count:
                    return rgba_array
                elif len(rgba_array.shape) == 1 and rgba_array.shape[0] == 4:
                    # single rgba vector, repeat it point_count times
                    return np.broadcast_to(rgba_array, (point_count, 4))
                raise ValueError()
            except Exception:
                raise ValueError("Point rgba must be either a vector of 4 integers or a matrix with 4 rows and same of columns as positions")

    def _get_diameter_property_as_array(self, point_count):
        if not self.diameters:
            return np.zeros(point_count)
        else:
            try:
                diameter_array = np.array(self.diameters, dtype=np.float32)
                if len(diameter_array.shape) == 1 and diameter_array.shape[0] == point_count:
                    return diameter_array
                elif len(diameter_array.shape) == 0:
                    # single diameter value, repeat it point_count times
                    return np.broadcast_to(diameter_array, point_count)
                raise ValueError()
            except Exception:
                raise("Point diameter must be either single float value or a vector with same number as positions")

    def _get_properties_as_arrays(self):
        # Get number of points from the number of specified point coordinate triplets
        xyz_array = np.asarray(self.positions, dtype=np.float32)
        point_count = 0
        if len(xyz_array.shape) == 1:
            if xyz_array.shape[0] == 3:
                point_count = 1
                xyz_array = [xyz_array]
        elif len(xyz_array.shape) == 2:
            if xyz_array.shape[1] == 3:
                point_count = xyz_array.shape[0]
        if point_count == 0:
            raise ValueError("Point positions must be 3-element vector or Nx3 matrix")

        name_array = PointMessage._get_string_property_as_array(self.names, point_count, 'names')
        group_array = PointMessage._get_string_property_as_array(self.groups, point_count, 'groups')
        owner_array = PointMessage._get_string_property_as_array(self.owners, point_count, 'owners')
        rgba_array = self._get_rgba_property_as_array(point_count)
        diameter_array = self._get_diameter_property_as_array(point_count)

        return name_array, group_array, rgba_array, xyz_array, diameter_array, owner_array

    def _pack_content(self):

        name_array, group_array, rgba_array, xyz_array, diameter_array, owner_array = self._get_properties_as_arrays()
        point_count = len(name_array)

        binary_content = b""
        for point_index in range(point_count):
            binary_content += struct.pack("> 64s", name_array[point_index].encode('utf8'))
            binary_content += struct.pack("> 32s", group_array[point_index].encode('utf8'))
            rgba = rgba_array[point_index]
            binary_content += struct.pack("> B B B B", rgba[0], rgba[1], rgba[2], rgba[3])
            xyz = xyz_array[point_index]
            binary_content += struct.pack("> f f f", xyz[0], xyz[1], xyz[2])
            binary_content += struct.pack("> f", diameter_array[point_index])
            binary_content += struct.pack("> 20s", owner_array[point_index].encode('utf8'))

        return binary_content

    def _unpack_content(self, content):
        self.positions = []
        self.names = []
        self.rgba_colors = []
        self.diameters = []
        self.groups = []
        self.owners = []
        s = struct.Struct('> 64s 32s B B B B f f f f 20s')
        item_length = 64+32+4+4*3+4+20
        point_count = int(len(content)/item_length)
        for point_index in range(point_count):
            values = s.unpack(content[point_index*item_length:(point_index+1)*item_length])
            self.names.append(values[0].decode().rstrip(' \t\r\n\0'))
            self.groups.append(values[1].decode().rstrip(' \t\r\n\0'))
            self.rgba_colors.append((values[2], values[3], values[4], values[5]))
            self.positions.append((values[6], values[7], values[8]))
            self.diameters.append(values[9])
            self.owners.append(values[10].decode().rstrip(' \t\r\n\0'))


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
    base = 1000000000  # 10^9
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
        "POINT": PointMessage,
    }
