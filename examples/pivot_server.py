from time import sleep

import pyigtl  # pylint: disable=import-error
import click
import numpy as np
from scipy.spatial.transform import Rotation as R


def normalize_vector(v: np.ndarray):
    """
    Normalize a vector by its Euclidean norm.
    If the vector is 0, return the zero vector of same dimension.
    """
    if np.linalg.norm(v) == 0:
        return np.zeros_like(v)
    return v / np.linalg.norm(v)


def sample_pivot_tool_position(
    needle_length: float, needle_offset_x: float, needle_offset_y: float
):
    """
    ...
    """
    tool_position = np.zeros(3)
    while np.linalg.norm(tool_position) == 0:
        tool_position = np.random.random(size=3)
    tool_position = normalize_vector(tool_position)
    tool_position *= needle_length
    tool_position += np.array([needle_offset_x, needle_offset_y, 0])
    return tool_position


def sample_pivot_tool_orientation(
    reference_position: np.ndarray, tool_position: np.ndarray
):
    """
    ...
    """
    look_dir = normalize_vector(reference_position - tool_position)
    look_xz_length = np.sqrt(look_dir[2] ** 2 + look_dir[0] ** 2)
    rot_x = -np.arctan2(look_dir[1], look_xz_length)
    rot_y = np.arctan2(look_dir[0], look_dir[2])
    rot_z = np.random.random() * np.pi * 2
    R_x = R.from_euler("x", rot_x).as_matrix()
    R_y = R.from_euler("y", rot_y).as_matrix()
    R_z = R.from_euler("z", rot_z).as_matrix()
    R_full = R_z @ R_y @ R_x
    return R_full


def sample_spin_tool_position(
    needle_length: float, needle_offset_x: float, needle_offset_y: float
):
    """
    ...
    """
    tool_position = np.array([needle_offset_x, needle_offset_y, needle_length])
    return tool_position


def sample_spin_tool_orientation():
    rot_z = np.random.random() * 2 * np.pi
    R_z = R.from_euler("z", rot_z).as_matrix()
    return R_z


def run_server(
    needle_length: float = 10.0,
    needle_offset_x: float = 0.0,
    needle_offset_y: float = 0.0,
    spin: bool = False,
    port: int = 18944,
):
    """
    ...
    """
    server = pyigtl.OpenIGTLinkServer(port=port)
    connected = False
    click.secho(f"Running IGT server on port {port}.", fg="blue")

    while True:
        # wait for client to connect
        if not server.is_connected():
            if connected:
                click.secho("Client disconnected.", fg="blue")
            connected = False
            sleep(0.1)
            continue

        # new client connected
        if not connected:
            click.secho("Client connected.", fg="blue")
            connected = True

        # set reference position
        reference_position = np.array([5, 1, 2])
        matrix_reference = np.eye(4)
        matrix_reference[0:3, 3] = reference_position + np.random.normal(0, 0.1, size=3)
        transform_message_reference = pyigtl.TransformMessage(
            matrix_reference, device_name="Reference"
        )
        server.send_message(transform_message_reference, wait=True)

        # create tool matrix based on needle geometry and ground truth reference position (without noise)
        matrix_tool = np.eye(4)
        tool_position = sample_pivot_tool_position(
            needle_length, needle_offset_x, needle_offset_y
        )

        # set tool orientation, pivoting around reference
        R_orientation = sample_pivot_tool_orientation(reference_position, tool_position)

        # send full tool matrix to client
        matrix_tool[0:3, 3] = tool_position
        matrix_tool[0:3, 0:3] = R_orientation
        transform_message_tool = pyigtl.TransformMessage(
            matrix_tool, device_name="Tool"
        )
        server.send_message(transform_message_tool, wait=True)

        # compute tool2reference transform for convenience
        matrix_tool2reference = matrix_tool @ matrix_reference
        transform_message_tool2reference = pyigtl.TransformMessage(
            matrix_tool2reference, device_name="ToolToReference"
        )
        server.send_message(transform_message_tool2reference, wait=True)


@click.command()
@click.option(
    "--needle-length", "-n", type=float, default=10, help="Needle length in mm"
)
@click.option("--needle-offset-x", "-x", type=float, default=0, help="")
@click.option("--needle-offset-y", "-y", type=float, default=0, help="")
@click.option("--spin", "-s", is_flag=True)
@click.option("--port", "-p", type=int, default=18944)
def main(
    needle_length: float,
    needle_offset_x: float,
    needle_offset_y: float,
    spin: bool,
    port: int,
):
    run_server(
        needle_length=needle_length,
        needle_offset_x=needle_offset_x,
        needle_offset_y=needle_offset_y,
        spin=spin,
        port=port,
    )


if __name__ == "__main__":
    main()
