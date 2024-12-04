import time
import serial
import numpy as np
from enum import Enum, auto, IntEnum
from dataclasses import dataclass
import struct


# class TestbedState(Enum):
#     HOMED = auto()
#     HOMING = auto()
#     LEFT = auto()
#     RIGHT = auto()
#     ERROR = auto()
#     NONE = auto()

class TestbedCommandType(IntEnum):
    HOME = 0
    SPEED = 1
    QUERY = 2

class TestbedState(IntEnum):
    HOMED = 0
    RIGHT = 1
    ERROR = 2
    HOMING = 3
    RUNNING = 4

@dataclass
class TestbedCommand:
    type: TestbedCommandType
    value: float | None = None

class Testbed:
    """
    Class to interface with the testbed. The testbed is a linear actuator that moves the camera and tool in the x-axis.
    """

    def __init__(self, port="/dev/ttyACM0"):
        """
        :param port: The serial port to connect to the testbed. Baud rate is 115200.
        """
        self.ser: serial.Serial = serial.Serial(port, 1000000, timeout=1)
        self.pos = 0
        self.speed = 0
        self.command_speed = 0
        self.state = TestbedState.RUNNING
        self.no_response = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.ser.close()

    def __read_response(self) -> None or (TestbedState, float, float):
        """Reads a line from the serial buffer,
        decodes it and returns its contents."""
        line = self.ser.readline().decode('utf-8').strip()
        if line == "":
            return None
        state, speed, pos = line.split(',')
        self.state = TestbedState(int(state))
        self.speed = float(speed)
        self.pos = float(pos)
        return self.state, self.speed, self.pos

    def __send_serial_command(self, command: TestbedCommand) -> bool:
        """
        Send a command to the testbed.
        :param command: TestbedCommand - The command to send.
        :return: bool - True if the command was sent successfully.
        """
        self.ser.write(f"<{command.type}:".encode('utf-8'))
        if command.value is not None:
            self.ser.write(f"{command.value}>".encode('utf-8'))
        else:
            self.ser.write(f">".encode('utf-8'))
        return True

    def home(self) -> bool:
        """
        Home the testbed to the far left.
        :return: True if command sent successfully.
        """
        self.__send_serial_command(TestbedCommand(TestbedCommandType.HOME, 0))
        while self.state == TestbedState.HOMING:
            self.query()
            time.sleep(0.1)
        # self.ser.reset_output_buffer()
        # self.ser.reset_input_buffer()
        return self.state == TestbedState.HOMED

    def query(self) -> (TestbedState, float, float):
        """
        Get the current position of the testbed.
        :return: float - The position in mm.
        """
        self.__send_serial_command(TestbedCommand(TestbedCommandType.QUERY))
        self.__read_response()
        return self.state, self.speed, self.pos


    def set_speed(self, speed: float) -> bool:
        """
        Set the speed of the testbed. (positive is right, negative is left)
        :param speed: speed in mm/s
        :return: True if the speed was set successfully.
        """
        self.__send_serial_command(TestbedCommand(TestbedCommandType.SPEED, speed))
        self.query()
        return np.isclose(self.speed, speed, atol=0.1)


    def stop(self) -> bool:
        """
        Stop the testbed.
        :return: True if the testbed was stopped successfully.
        """
        self.set_speed(0)
        self.query()
        return self.speed == 0
