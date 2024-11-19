#include<EnableInterrupt.h>
#include<time.h>
#include <ros.h>
#include <std_msgs/Int8.h>
#include<std_msgs/Float64.h>
ros::NodeHandle nh;
std_msgs::Float64 throttle;
ros::Publisher curr_thr_pub("/odo", &throttle);


double ch1=7;
int a=4; 
int b=5;
int absolute;
int mapped;
int Max=0;
int Min=2000;

int encoderPin1=2;
int encoderPin2=3;
double elapsedTime, previousTime, prev_Time, currentTime = 0;
double error = 0, lastError = 0, cumError = 0, rateError = 0;



volatile int lastEncoded = 0; // old odometer with 600 ppr pin 2 and 3
volatile long encoderValue = 0;
//float prev_rotation = 0;
long prev_rotation = 0;
int i = 0;
long lastencoderValue = 0;
float elapsed_time = 0;
float elapsed_time_after = 0;
float velocity = 0, p = 0;
double s = 0;
int lastMSB = 0;
float RPM = 0;
double curr_Time = 0;
void setup()
{
  Serial.begin(9600);
  
  pinMode(7,INPUT);
  pinMode(4,OUTPUT); 
  pinMode(5,OUTPUT); 

  pinMode(encoderPin1, INPUT_PULLUP);
  pinMode(encoderPin2, INPUT_PULLUP);

  enableInterrupt(2, updateEncoder_old, CHANGE);
  enableInterrupt(3, updateEncoder_old, CHANGE);
   
}

void loop()
{
nh.spinOnce();
throttle.data = velocity; // percent of 5V
curr_thr_pub.publish(&throttle);

//ch1 =  1600;
ch1 = pulseIn(7,HIGH);
absolute=abs(ch1-1450);
float pwm_float = ch1;
  String pwm_str = String(ch1, 5);
  const char* motor_str = pwm_str.c_str();
    nh.loginfo(motor_str);
//if (ch1>Max)
//{Max =ch1;}
//if (ch1<Min)
//{Min =ch1;}
////Serial.println(ch1);
//
////Serial.println(velocity);
//if((ch1>1500))
//{  mapped=map(absolute,0,370,230,255);   
// digitalWrite(a,LOW);
// analogWrite(5,mapped);
//}
//
//else if((ch1<1100))
//{     
//  mapped=map(absolute,0,336,230,255);
//  digitalWrite(a,HIGH);
//  analogWrite(5,mapped); 
//}
//
//else if((ch1>1100 && ch1<1500))
//{     
//  digitalWrite(a,LOW);
//  analogWrite(5,0); 
//}
}
void updateEncoder_old ()
{
  int MSB = digitalRead(encoderPin1); //MSB = most significant bit
  int LSB = digitalRead(encoderPin2); //LSB = least significant bit
  int encoded = (MSB << 1) |LSB; //converting the 2 pin value to single number
  int sum  = (lastEncoded << 2) | encoded; //adding it to the previous encoded value
  elapsed_time_after = float(micros());
  prev_rotation = encoderValue; // 0.0026 for 2pie/2400 
  if(sum == 0b1101 || sum == 0b0100 || sum == 0b0010 || sum == 0b1011) encoderValue ++;
  if(sum == 0b1110 || sum == 0b0111 || sum == 0b0001 || sum == 0b1000) encoderValue --;
  
  lastEncoded = encoded; //store this value for next time

velocity = ((prev_rotation-encoderValue)*3436.1116)/(18.75*2.92*((elapsed_time_after)-(elapsed_time)));
elapsed_time = elapsed_time_after;
}
