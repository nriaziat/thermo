import serial
import numpy as np


class Testbed:
    def __init__(self, port="/dev/ttyACM0"):
        self.ser = serial.Serial(port, 115200)

    def send_serial_command(self, command: str) -> bool:
        self.ser.write(f'{command}\n'.encode())
        return True

    def home(self) -> bool:
        self.send_serial_command('-1')
        return True

    def get_position(self) -> float:
        self.send_serial_command('-2')
        data = self.read_serial()
        if data == -1:
            return -1
        return float(data)

    def set_speed(self, speed: float) -> bool:
        self.send_serial_command(f'{speed:.4f}')
        data = self.read_serial()
        if data == -1:
            self.stop()
            return False
        conf = float(data)
        if not np.isclose(conf, speed, atol=0.01):
            return False
        return True

    def stop(self) -> bool:
        return self.set_speed(0)

    def read_serial(self) -> str or int:
        data = self.ser.readline().decode().strip().lower()
        if data == "left" or data == "right":
            print("Endstop reached")
            return -1
        return data
