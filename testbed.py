import time
import serial
import numpy as np
from enum import Enum, auto
from dataclasses import dataclass


class TestbedCommandType(Enum):
    HOME = "?H"
    POSITION = "?P"
    SPEED = auto()

    def __len__(self):
        return len(self.value)


class TestbedResponse(Enum):
    HOMED = "homed"
    LEFT = "left"
    RIGHT = "right"
    ERROR = "-1"
    HOMING = "homing"
    NONE = ""


@dataclass
class TestbedCommand:
    type: TestbedCommandType
    speed: float | None = None


class Testbed:
    """
    Class to interface with the testbed. The testbed is a linear actuator that moves the camera and tool in the x-axis.
    """

    def __init__(self, port="/dev/ttyACM0"):
        """
        :param port: The serial port to connect to the testbed. Baud rate is 115200.
        """
        self.ser: serial.Serial = serial.Serial(port, 115200, timeout=1)

    def __send_serial_command(self, command: TestbedCommand) -> bool:
        """
        Send a command to the testbed.
        :param command: TestbedCommand - The command to send.
        :return: bool - True if the command was sent successfully.
        """
        if command.type == TestbedCommandType.SPEED:
            msg = f'{command.speed}\n'.encode()
        else:
            msg = f'{command.type.value}\n'.encode()
        return self.ser.write(msg) == len(msg)

    def home(self) -> bool:
        """
        Home the testbed to the far left.
        :return: True if command sent successfully.
        """
        self.__send_serial_command(TestbedCommand(TestbedCommandType.HOME))
        while (response := self.__read_serial()) != TestbedResponse.HOMING:
            if response == TestbedResponse.ERROR:
                return False
            time.sleep(0.1)
            self.ser.reset_output_buffer()
            self.__send_serial_command(TestbedCommand(TestbedCommandType.HOME))
        while (response := self.__read_serial()) != TestbedResponse.HOMED:
            if response == TestbedResponse.ERROR:
                return False
            time.sleep(1)
        return True

    def get_position(self) -> float:
        """
        Get the current position of the testbed.
        :return: float - The position of the testbed in mm.
        """
        ret = self.__send_serial_command(TestbedCommand(TestbedCommandType.POSITION))
        data = self.__read_serial()
        try:
            response = TestbedResponse(data)
            if response == TestbedResponse.ERROR:
                return -1
        except ValueError:
            return float(data)

    def set_speed(self, speed: float) -> bool:
        """
        Set the speed of the testbed. (positive is right, negative is left)
        :param speed: speed in mm/s
        :return: True if the speed was set successfully.
        """
        ret = self.__send_serial_command(TestbedCommand(TestbedCommandType.SPEED, speed))
        if not ret:
            return False
        data = self.__read_serial()
        if data is float:
            if not np.isclose(data, speed, atol=0.01):
                return False
            return True
        if data == TestbedResponse.ERROR:
            self.stop()
            return False
        elif data == TestbedResponse.HOMING or data == TestbedResponse.HOMED:
            return False
        elif data == TestbedResponse.LEFT or data == TestbedResponse.RIGHT:
            self.stop()
            return False

    def stop(self) -> bool:
        """
        Stop the testbed.
        :return: True if the testbed was stopped successfully.
        """
        return self.set_speed(0)

    def __read_serial(self) -> float | TestbedResponse:
        """
        Read the serial port.
        :return: str - The data read from the serial port.
        """
        if not self.ser.in_waiting:
            return TestbedResponse.NONE
        data = self.ser.readline().decode().strip().lower()
        try:
            return TestbedResponse(data)
        except ValueError:
            try:
                return float(data)
            except ValueError:
                return -1
