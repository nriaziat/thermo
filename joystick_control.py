import time
import pygame
from testbed import Testbed, TestbedState
import numpy as np

pygame.init()

joystick = pygame.joystick.Joystick(0)
tb = Testbed()
print("Joystick connected")
last_pos = 0
last_time = time.time()
pos = []
times = []
speed = 0
while True:
    pygame.event.get()
    speed += 0.25 * ((joystick.get_axis(5) + 1)/2 + - 0.25 * (joystick.get_axis(2) + 1)/2) - 0.01 * speed
    speed = np.clip(speed, -15, 15)
    ret = tb.set_speed(speed)
    if not ret and tb.state == TestbedState.RUNNING:
        print("Error setting speed")
    if joystick.get_button(1):
        tb.stop()
        break
    elif joystick.get_button(0):
        tb.home()
    pos.append(tb.pos)
    times.append(time.time())
    last_time = time.time()
    state_text = TestbedState(tb.state).name
    print(f"State: {state_text}")

