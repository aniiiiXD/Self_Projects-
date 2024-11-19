#include <EnableInterrupt.h>
#include <Servo.h>
#include <ros.h>
#include <std_msgs/Int8.h>
#include <std_msgs/Float64.h>

ros::NodeHandle nh;
float steer = 0;
float speedo_input = 0;
int flag = 0;

void vel_callback(const std_msgs::Float64 &desired_velocity)
{
  speedo_input = desired_velocity.data;
}

void gear_callback(const std_msgs::Float64 &gear_)
{
  steer = gear_.data;
}

ros::Subscriber<std_msgs::Float64> vel_sub("/velocity", vel_callback);
ros::Subscriber<std_msgs::Float64> gear_sub("/steer", gear_callback);

std_msgs::Float64 throttle;
ros::Publisher curr_thr_pub("/odom", &throttle);

Servo servo; // create servo object to control a servo

int DIR = 12;
int PWM = 11;
const int encoderPin1 = 3; // old |
const int encoderPin2 = 4; // old |
float sped = 0;
// PID constants
float kp = 1, ki = 0.5, kd = 0.0008; //.36

// PID variables
double elapsedTime, previousTime, prev_Time, currentTime = 0;
double error = 0, lastError = 0, cumError = 0, rateError = 0;

volatile int lastEncoded = 0; // old odometer with 600 ppr pin 2 and 3
volatile long encoderValue = 0;
// float prev_rotation = 0;
long prev_rotation = 0;
int i = 0;
long lastencoderValue = 0;
float elapsed_time = 0;
float elapsed_time_after = 0;
float velocity = 0, p = 0;
double s = 0;
int lastMSB = 0;
int lastLSB = 0;
float RPM = 0;
float arr[10] = {0};
double curr_Time = 0;
bool dir;

void setup()
{
  nh.subscribe(vel_sub);
  nh.subscribe(gear_sub);
  nh.advertise(curr_thr_pub);
  prev_Time = float(millis());
  servo.attach(6); // attaches the servo on pin 9 to the servo objectư
  Serial.begin(115200);
  pinMode(DIR, OUTPUT);
  pinMode(PWM, OUTPUT);
  pinMode(encoderPin1, INPUT_PULLUP);
  pinMode(encoderPin2, INPUT_PULLUP);

  enableInterrupt(encoderPin1, updateEncoder_old, CHANGE);
  enableInterrupt(encoderPin2, updateEncoder_old, CHANGE);
}

void loop()
{
  nh.spinOnce();
  Serial.println(velocity);
  Serial.println(speedo_input);
  // for steer following part
  float ster = 0;
  ster = (steer + 14);
  String str = String(ster, 5);
  const char *cstr = str.c_str();
  nh.loginfo(cstr);
  int pwm = 0;
  pwm = (speedo_input * 155) / (0.71) + 100;
  Serial.println(pwm);
  analogWrite(PWM, pwm);
  digitalWrite(DIR, LOW);
  float pwm_float = pwm;
  String pwm_str = String(pwm_float, 5);
  const char *motor_str = pwm_str.c_str();
  //  nh.loginfo(motor_str);
  //   while(throttle.data==0){
  throttle.data = velocity; // percent of 5V
  //    continue;
  //   }
  curr_thr_pub.publish(&throttle);
  //
  servo.write(ster);
  //   delay(200);         // waits 10ms for the servo to reach the position
}

void updateEncoder_old()
{
  int MSB = digitalRead(encoderPin1);     // MSB = most significant bit
  int LSB = digitalRead(encoderPin2);     // LSB = least significant bit
  int encoded = (MSB << 1) | LSB;         // converting the 2 pin value to single number
  int sum = (lastEncoded << 2) | encoded; // adding it to the previous encoded value
  elapsed_time_after = float(micros());
  prev_rotation = encoderValue; // 0.0026 for 2pie/2400
  if (sum == 0b1101 || sum == 0b0100 || sum == 0b0010 || sum == 0b1011)
    encoderValue++;
  if (sum == 0b1110 || sum == 0b0111 || sum == 0b0001 || sum == 0b1000)
    encoderValue--;

  lastEncoded = encoded; // store this value for next time

  velocity = ((prev_rotation - encoderValue) * 62.76) / (elapsed_time_after - elapsed_time);
  s = velocity + s;
  elapsed_time = elapsed_time_after;
  if (i == 10)
  {
    i = 0;
    sped = s / 10;
    s = 0;
  }
  i = i + 1;
}

int computePID()
{
  currentTime = millis();                                    // get current time
  elapsedTime = (double)(currentTime - previousTime) / 1000; // compute time elapsed from previous computation
  error = velocity - speedo_input;
  // windup on integral term, can implement Clegg integrator also
  if (abs(cumError * ki) >= 1 && cumError * error > 0)
    ;
  else
    cumError += error * elapsedTime;
  rateError = (error - lastError) / elapsedTime;

  double out = kp * error + ki * cumError + kd * rateError;

  lastError = error;                 // remember current error
  previousTime = currentTime;        // remember current time
  int pwm = out * 205 / (0.71) + 50; // PWM MAPPING ASSUMING GADI SPEED BETWEEN 50 to 250
  if (out < 0)
    dir = HIGH;
  else
    dir = LOW;
  return pwm;
}
