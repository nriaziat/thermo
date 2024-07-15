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
const float home_speed = 0.5 * max_speed;
volatile byte left_stop = LOW;
volatile byte right_stop = LOW;
volatile unsigned long last_interrupt_time = 0;

String receivedChars;  

bool newData = false;
bool homed=false;
bool start_home=false;

float speed = 0;
int msg = 0;
float pos;

void setup() {
  Serial.begin(115200);
  pinMode(left_stop_pin, INPUT_PULLUP);
  pinMode(right_stop_pin, INPUT_PULLUP);
  pinMode(LED_pin, OUTPUT);
  // attachInterrupt(digitalPinToInterrupt(left_stop_pin), left_reset, FALLING);
  // attachInterrupt(digitalPinToInterrupt(right_stop_pin), right_reset, FALLING);
  stepper.setMaxSpeed(max_speed * steps_per_mm);
  stepper.setAcceleration(125000.0 * steps_per_mm);
}

void loop() {
  recvWithEndMarker();
  left_stop = !digitalRead(left_stop_pin);
  right_stop = !digitalRead(right_stop_pin);
  // Serial.print("End stops:");
  // Serial.println(end_stop_triggered);
  if (newData) {
    if (receivedChars.equals("?H")){
      if (!homed){
        Serial.println("homing");
        start_home=true;
      } else {
        Serial.println("already homed");
      }
    }
    else if (receivedChars.equals("?P")){
      Serial.println(pos);
      // digitalWrite(LED_pin, HIGH);
    }
    else if (receivedChars.indexOf("!P") != -1){
      float new_pos = receivedChars.substring(2).toFloat() * steps_per_mm;
      stepper.runToNewPosition(new_pos);
      homed=false;
    }
    else if (receivedChars.indexOf("!IP") != -1){
      float new_pos = (pos + receivedChars.substring(3).toFloat()) * steps_per_mm;
      stepper.runToNewPosition(new_pos);
      homed=false;
    }
    else {
      speed = receivedChars.toFloat();
      speed = max(-max_speed, min(speed, max_speed));
      Serial.println(speed);
      stepper.setSpeed(speed * steps_per_mm);
      homed=false;
      // digitalWrite(LED_pin, LOW);
    }
    newData = false;
  }
  if (left_stop) {
    stepper.stop();
    stepper.setCurrentPosition(0);
    stepper.runToNewPosition(reset_dist * steps_per_mm);
    left_stop = false;
    if (start_home){
      Serial.println("homed");
      start_home=false;
    } else {
      Serial.println("left");
    }
  } else if (right_stop) {
    stepper.stop();
    Serial.println("right");
    right_stop = false;
  }
  pos = stepper.currentPosition() / steps_per_mm;
  if (start_home){
    stepper.setSpeed(-home_speed * steps_per_mm);
  }
  stepper.runSpeed();
}

void left_reset() {
  detachInterrupt(digitalPinToInterrupt(left_stop_pin));
  unsigned long interrupt_time = millis();
  // If interrupts come faster than 200ms, assume it's a bounce and ignore
  if (interrupt_time - last_interrupt_time > 200) 
  {
    stepper.stop();
    left_stop = true;
    last_interrupt_time = interrupt_time;
  }
}

void right_reset() {
  detachInterrupt(digitalPinToInterrupt(right_stop_pin));
  unsigned long interrupt_time = millis();
  // If interrupts come faster than 200ms, assume it's a bounce and ignore
  if (interrupt_time - last_interrupt_time > 200) 
  {
    stepper.stop();
    right_stop = true;
    last_interrupt_time = interrupt_time;
  }
}

void recvWithEndMarker() {
  char endMarker = '\n';

  if (Serial.available()) {

    receivedChars = Serial.readStringUntil(endMarker);
    newData = true;

  }
}