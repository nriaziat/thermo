#include <AccelStepper.h>

AccelStepper stepper(AccelStepper::FULL2WIRE, 9, 8);

const int steps_per_rev = 6400;
const int mm_per_rev = 10;
const float steps_per_mm = (float)steps_per_rev / (float)mm_per_rev;

const byte left_stop_pin = 2;
const byte right_stop_pin = 3;
const byte LED_pin = 13;
const float reset_dist = 1;
const float max_speed = 55;
const float home_speed = max_speed;
volatile byte left_stop = LOW;
volatile byte right_stop = LOW;

const byte numChars = 32;
char receivedChars[numChars];  // an array to store the received data

boolean newData = false;

float speed = 0;
bool cur_dir = 1;
float pos;

void setup() {
  Serial.begin(115200);
  pinMode(left_stop_pin, INPUT_PULLUP);
  pinMode(right_stop_pin, INPUT_PULLUP);
  pinMode(13, OUTPUT);
  attachInterrupt(digitalPinToInterrupt(left_stop_pin), left_reset, FALLING);
  attachInterrupt(digitalPinToInterrupt(right_stop_pin), right_reset, FALLING);
  stepper.setMaxSpeed(max_speed * steps_per_mm);
  stepper.setAcceleration(125000.0 * steps_per_mm);
  Serial.println("Press enter to home: ");
  while (Serial.available()==0){

  }
  home();
  delay(1000);
  stepper.setSpeed( speed * steps_per_mm);
}

void loop() {
  recvWithEndMarker();
  if (newData) {
    speed = atof(receivedChars);
    if (speed < 0){
      cur_dir = !cur_dir;
      speed = -speed;
    }
    speed = max(0, min(speed, max_speed));
    Serial.print("Speed set to ");
    Serial.print(speed);
    Serial.println(" mm/s.");
    stepper.setSpeed((2*cur_dir-1) * speed * steps_per_mm);
    newData = false;
  }
  if (left_stop) {
    cur_dir = 1;
    stepper.setCurrentPosition(0);
    stepper.runToNewPosition(reset_dist * steps_per_mm);
    Serial.println("Left Home.");
    left_stop = LOW;
  } else if (right_stop) {
    cur_dir = 0;
    stepper.runToNewPosition(stepper.currentPosition() - (reset_dist * steps_per_mm));
    Serial.println("Right home.");
    right_stop = LOW;
  }
  pos = stepper.currentPosition() / steps_per_mm;
  stepper.runSpeed();
}

void left_reset() {
  stepper.stop();
  left_stop = HIGH;
}

void right_reset() {
  stepper.stop();
  right_stop = HIGH;
}

void home() {
  stepper.setSpeed(-home_speed * steps_per_mm);
  while (!left_stop) {
    stepper.runSpeed();
  }
  stepper.stop();
  stepper.setCurrentPosition(0);
  delay(250);
  stepper.setMaxSpeed(max_speed * steps_per_mm);
  stepper.runToNewPosition(reset_dist * steps_per_mm);
  left_stop = LOW;
}

void recvWithEndMarker() {
  static byte ndx = 0;
  char endMarker = '\n';
  char rc;

  if (Serial.available() > 0) {
    rc = Serial.read();

    if (rc != endMarker) {
      receivedChars[ndx] = rc;
      ndx++;
      if (ndx >= numChars) {
        ndx = numChars - 1;
      }
    } else {
      receivedChars[ndx] = '\0';  // terminate the string
      ndx = 0;
      newData = true;
    }
  }
}