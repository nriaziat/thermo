#include <AccelStepper.h>

const int directionPin = 8;
const int pulsePin = 9;
AccelStepper stepper(AccelStepper::FULL2WIRE, pulsePin, directionPin);
void recvOrder(void);
void recvData(void);

const float mm_per_rev = 10;
const int steps_per_rev = 800;
const float steps_per_mm = (float)steps_per_rev / mm_per_rev;
const byte numChars = 32;
char receivedChars[numChars];
char tempChars[numChars];  // temporary array for use when parsing

const byte left_stop_pin = 2;
const byte right_stop_pin = 3;
const byte LED_pin = 13;
const float reset_dist = 1;
const float max_speed = 20;
const float home_speed = max_speed;
volatile bool left_stop = LOW;
volatile bool right_stop = LOW;
long int curr_micros = micros();
long int last_micros = micros();

enum Order {
  HOME = 0,
  SPEED = 1,
  QUERY = 2
};

enum State {
  HOMED = 0,
  RIGHT = 1,
  ERROR = 2,
  HOMING = 3,
  RUNNING = 4
};

Order receivedOrder;
float receivedSpeed = 0;
State state = RUNNING;

bool newOrder = false;

float speed = 0;
float pos;

void leftStopISR() {
  left_stop = true;
}

void rightStopISR() {
  right_stop = true;
}

void setup() {
  Serial.begin(1000000);
  pinMode(left_stop_pin, INPUT_PULLUP);
  pinMode(right_stop_pin, INPUT_PULLUP);
  pinMode(LED_pin, OUTPUT);
  attachInterrupt(digitalPinToInterrupt(left_stop_pin), leftStopISR, FALLING);
  attachInterrupt(digitalPinToInterrupt(right_stop_pin), rightStopISR, FALLING);
  stepper.setMaxSpeed(max_speed * steps_per_mm);
  stepper.setAcceleration(500.0 * steps_per_mm);
  left_stop = !digitalRead(left_stop_pin);
  right_stop = !digitalRead(right_stop_pin);
}

void loop() {
  recvWithStartEndMarkers();
  if (newOrder) {
    strcpy(tempChars, receivedChars);
    parseData();
  }
  //  left_stop = !digitalRead(left_stop_pin);
  //  right_stop = !digitalRead(right_stop_pin);
  //  digitalWrite(LED_pin, right_stop);
  switch (state) {
    case HOMED:
      if (newOrder) {
        if (receivedOrder == SPEED) {
          updateSpeed(HOMED);
        } else if (receivedOrder == QUERY) {
          respondToQuery();
        } else {
          newOrder = false;
        }
      }
      break;
    case RIGHT:
      if (newOrder) {
        if (receivedOrder == SPEED) {
          updateSpeed(RIGHT);
        } else if (receivedOrder == QUERY) {
          respondToQuery();
        } else if (receivedOrder == HOME) {
          state = HOMING;
          newOrder = false;
        }
      }
      break;
    case ERROR:
      stop();
      break;
    case HOMING:
      speed = -home_speed;
      stepper.setSpeed(speed * steps_per_mm);
      if (left_stop) {
        left_stop = LOW;
        stepper.setCurrentPosition(0);
        stop();
        state = HOMED;
      }
      if (newOrder) {
        if (receivedOrder == QUERY) {
          respondToQuery();
        } else {
          newOrder = false;
        }
      }
      if (newOrder) newOrder=false;
      break;
    case RUNNING:
      if (newOrder) {
        if (receivedOrder == SPEED) {
          updateSpeed(RUNNING);
        } else if (receivedOrder == QUERY) {
          respondToQuery();
        } else if (receivedOrder == HOME) {
          state = HOMING;
          newOrder = false;
        }
      }
      if (left_stop && receivedSpeed <= 0) {
        stepper.setCurrentPosition(0);
        state = HOMED;
        speed = 0;
        left_stop = LOW;
      } else if (right_stop && receivedSpeed >= 0) {
        state = RIGHT;
        speed = 0;
        right_stop = LOW;
      }
      break;
    default:
      break;
  }
  stepper.runSpeed();
  pos = stepper.currentPosition() / steps_per_mm;
  last_micros = curr_micros;
  curr_micros = micros();
}

void respondToQuery(){
  Serial.print(state);
  Serial.print(",");
  Serial.print(speed);
  Serial.print(",");
  Serial.println(pos);
  newOrder = false;
}

void stop(){
  speed = 0;
  stepper.setSpeed(speed);
}

void updateSpeed(State currentState){
  switch (currentState) {
    case HOMED:
      if (receivedSpeed <= 0) {
        speed = 0;
      } else {
        speed = receivedSpeed;
        state = RUNNING;
      }
      break;
    case RIGHT:
      if (receivedSpeed >= 0) {
        speed = 0;
      } else {
        speed = receivedSpeed;
        state = RUNNING;
      }
      break;
    case RUNNING:
      speed = receivedSpeed;
  }
  stepper.setSpeed(speed * steps_per_mm);
  newOrder = false;
}

void recvWithStartEndMarkers() {
  static boolean recvInProgress = false;
  static byte ndx = 0;
  char startMarker = '<';
  char endMarker = '>';
  char rc;

  if (Serial.available() > 0 && newOrder == false) {
    rc = Serial.read();

    if (recvInProgress == true) {
      if (rc != endMarker) {
        receivedChars[ndx] = rc;
        ndx++;
        if (ndx >= numChars) {
          ndx = numChars - 1;
        }
      } else {
        receivedChars[ndx] = '\0';  // terminate the string
        recvInProgress = false;
        ndx = 0;
        newOrder = true;
      }
    }

    else if (rc == startMarker) {
      recvInProgress = true;
    }
  }
}

//============

void parseData() {      // split the data into its parts
  char* strtokIndx;     // this is used by strtok() as an index

  strtokIndx = strtok(tempChars, ":");  // get the first part - the string
  receivedOrder = (Order)atoi(strtokIndx);
  strtokIndx = strtok(NULL, ":");    // this continues where the previous call left off
  receivedSpeed = atof(strtokIndx);  // convert this part to a float

}