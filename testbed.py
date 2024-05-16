import serial
import numpy as np


class Testbed:
    """
    Class to interface with the testbed. The testbed is a linear actuator that moves the camera and tool in the x-axis.
    """
    def __init__(self, port="/dev/ttyACM0"):
        """
        :param port: The serial port to connect to the testbed. Baud rate is 115200.
        """
        self.ser = serial.Serial(port, 115200)

    def __send_serial_command(self, command: str) -> bool:
        """
        Send a command to the testbed.
        :param command: str
        :return: bool - True if the command was sent successfully.
        """
        self.ser.write(f'{command}\n'.encode())
        return True

    def home(self) -> bool:
        """
        Home the testbed to the far left.
        :return: True if command sent succesfully.
        """
        self.__send_serial_command('-1')
        return True

    def get_position(self) -> float:
        """
        Get the current position of the testbed.
        :return: float - The position of the testbed in mm.
        """
        self.__send_serial_command('-2')
        data = self.__read_serial()
        if data == -1:
            return -1
        return float(data)

    def set_speed(self, speed: float) -> bool:
        """
        Set the speed of the testbed. (positive is right, negative is left)
        :param speed: speed in mm/s
        :return: True if the speed was set successfully.
        """
        self.__send_serial_command(f'{speed:.4f}')
        data = self.__read_serial()
        if data == -1:
            self.stop()
            return False
        conf = float(data)
        if not np.isclose(conf, speed, atol=0.01):
            return False
        return True

    def stop(self) -> bool:
        """
        Stop the testbed.
        :return: True if the testbed was stopped successfully.
        """
        return self.set_speed(0)

    def __read_serial(self) -> str or int:
        """
        Read the serial port.
        :return: str or int - The data read from the serial port.
        """
        data = self.ser.readline().decode().strip().lower()
        if data == "left" or data == "right":
            print("Endstop reached")
            return -1
        return data
