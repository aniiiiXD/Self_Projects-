#include <ros/ros.h>
#include <std_msgs/Float64.h>
#include "cv_only/Speed.h"

class Converter
{
private:
    ros::NodeHandle nh;
    ros::Subscriber sub;
    ros::Publisher pub;
    cv_only::Speed speed;

public:
    Converter()
    {
        pub = nh.advertise<cv_only::Speed>("/odom_with_time", 2);
        sub = nh.subscribe<std_msgs::Float64>("/odom", 2, &Converter::callback, this);
    }

private:
    void callback(const std_msgs::Float64ConstPtr &msg)
    {
        speed.data = msg->data;
        speed.header.stamp = ros::Time::now();
        pub.publish(speed);
    }
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "odom_converter");
    Converter obj;
    ros::spin();
}