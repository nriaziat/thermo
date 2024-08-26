import time
import serial
import numpy as np
from enum import Enum, auto
from dataclasses import dataclass


class TestbedCommandType(Enum):
    HOME = "?H"
    POSITION = "?P"
    SPEED = "?S"
    ABS_POS = "!P"
    INC_POS = "!IP"

    def __len__(self):
        return len(self.value)


class TestbedResponse(Enum):
    HOMED = "homed"
    ALREADY_HOMED = "already homed"
    LEFT = "left"
    RIGHT = "right"
    ERROR = "-1"
    HOMING = "homing"
    NONE = ""


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
        self.ser: serial.Serial = serial.Serial(port, 115200, timeout=1)

    def __send_serial_command(self, command: TestbedCommand) -> bool:
        """
        Send a command to the testbed.
        :param command: TestbedCommand - The command to send.
        :return: bool - True if the command was sent successfully.
        """
        if command.type == TestbedCommandType.SPEED:
            msg = f'{command.value}\n'.encode()
        elif command.type == TestbedCommandType.ABS_POS or command.type == TestbedCommandType.INC_POS:
            msg = f'{command.type.value}{command.value}\n'.encode()
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
            time.sleep(0.1)
        self.ser.reset_output_buffer()
        self.ser.reset_input_buffer()
        return True

    def get_position(self) -> float:
        """
        Get the current position of the testbed.
        :return: float - The position of the testbed in mm.
        """
        ret = self.__send_serial_command(TestbedCommand(TestbedCommandType.POSITION))
        if not ret:
            print("Error sending position command")
            return -1
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
            print("Error sending speed command")
            return False
        data = self.__read_serial()
        if data is float:
            if not np.isclose(data, speed, atol=0.01):
                print(f"Speed not set correctly. Expected: {speed}, Got: {data}")
                return False
            return True
        if data == TestbedResponse.ERROR:
            print("Error setting speed")
            self.stop()
            return False
        elif data == TestbedResponse.LEFT or data == TestbedResponse.RIGHT:
            print("Endstop reached")
            self.stop()
            return False
        elif data == TestbedResponse.HOMING:
            print("Testbed state: ", data)
            return False
        return True

    def move_relative(self, distance: float) -> bool:
        """
        Move the testbed a relative distance.
        :param distance: float - The distance to move in mm.
        :return: bool - True if the testbed moved successfully.
        """
        ret = self.__send_serial_command(TestbedCommand(TestbedCommandType.INC_POS, distance))
        if not ret:
            print("Error sending move command")
            return False
        data = self.__read_serial()
        if data is float:
            if not np.isclose(data, distance, atol=0.01):
                print(f"Distance not set correctly. Expected: {distance}, Got: {data}")
                return False
            return True
        if data == TestbedResponse.ERROR:
            print("Error moving")
            self.stop()
            return False
        elif data == TestbedResponse.LEFT or data == TestbedResponse.RIGHT:
            print("Endstop reached")
            self.stop()
            return False
        elif data == TestbedResponse.HOMING:
            print("Testbed state: ", data)
            return False
        return True

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
