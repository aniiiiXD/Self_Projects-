#include "ros/ros.h"
#include "sensor_msgs/Imu.h"
#include "std_msgs/Float64.h"
#include <cmath>
#include <chrono>
#include <eigen3/Eigen/Dense>
#include "cv_only/Speed.h"

using namespace std;

class Fusion
{
private:
    ros::NodeHandle nh;
    ros::Publisher imu_pub;
    ros::Publisher encoder_pub;
    ros::Publisher yaw_pub;
    ros::Subscriber imu_sub;
    ros::Subscriber encoder_sub;
    cv_only::Speed global_speed;

public:
    Fusion() : nh("~")
    {
        imu_sub = nh.subscribe<sensor_msgs::Imu>("/imu/data", 2, &Fusion::imu_callback, this);
        encoder_sub = nh.subscribe<cv_only::Speed>("/odom_with_time", 2, &Fusion::encoder_callback, this);
        imu_pub = nh.advertise<std_msgs::Float64>("filtered_yaw", 2);
        encoder_pub = nh.advertise<std_msgs::Float64>("/final_speed", 2);
        yaw_pub = nh.advertise<std_msgs::Float64>("raw_yaw", 2);
    }

private:
    void imu_callback(const sensor_msgs::ImuConstPtr &msg)
    {
        cv_only::Speed speed = global_speed;

        static double current_t;
        static double t;
        static bool flag2 = 0;
        // double measure[] = {0, 0}; // yaw, z_comp of angular velocity

        static Eigen::Matrix<double, 3, 1> X_yaw;
        static Eigen::Matrix<double, 3, 3> P_yaw;
        Eigen::Matrix<double, 3, 3> F;
        Eigen::Matrix<double, 3, 3> Q; // process noise
        Eigen::Matrix<double, 2, 2> R; // measurement noise
        Eigen::Matrix<double, 3, 2> K;
        Eigen::Matrix<double, 2, 3> H;
        Eigen::Matrix<double, 2, 1> Z;
        Eigen::Matrix<double, 3, 3> I;
        Eigen::Matrix3d I3 = Eigen::Matrix3d::Identity();
        Eigen::Matrix2d I2 = Eigen::Matrix2d::Identity();
        double b = 0.08; // Bias
        double scaling_factor = 1;
        double m_yaw = 0, w = 0; // measured yaw and z comp of angular velocity

        if (!flag2) // initial iteration
        {
            current_t = seconds();
            X_yaw << m_yaw,
                w,
                b;
            P_yaw << scaling_factor * I;
            flag2 = 1;
        }

        t = seconds() - current_t;
        current_t = seconds();
        m_yaw = atan2(2 * (msg->orientation.w * msg->orientation.z + msg->orientation.x * msg->orientation.y), 1 - 2 * (msg->orientation.y * msg->orientation.y + msg->orientation.z * msg->orientation.z));
        w = msg->angular_velocity.z;
        Q = 0.5 * I3;
        // Q = 0.1 * I3; // quantifies the trust on the measurement
        R = 2 * I2;    // quantifies the trust on our prediction
        F << 1, t, -t,
            0, 1, 0,
            0, 0, 1;
        H << 1, 0, 0,
            0, 1, 0;
        Z << m_yaw,
            w;

        // prediction step
        X_yaw = F * X_yaw;
        P_yaw = F * P_yaw * F.transpose() + Q;

        // updation step
        K = P_yaw * H.transpose() * (H * P_yaw * H.transpose() + R).inverse();
        X_yaw = X_yaw + K * (Z - H * X_yaw);
        P_yaw = (I - K * H) * P_yaw;

        cout << "Yaw: " << m_yaw << " " << X_yaw(0, 0) << endl
             << endl;

        std_msgs::Float64 yaw;
        std_msgs::Float64 measured_yaw;
        measured_yaw.data = m_yaw;
        yaw.data = X_yaw(0, 0);
        imu_pub.publish(yaw);
        yaw_pub.publish(measured_yaw);

        speed_calculate(*msg, speed, current_t, t, X_yaw(0, 0));
    }

    void encoder_callback(const cv_only::SpeedConstPtr &speed)
    {
        global_speed = *speed;
    }

    void speed_calculate(const sensor_msgs::Imu imu_data, cv_only::Speed speed, double current_t, double t, double yaw)
    {
        static bool flag1 = 0;
        double scaling_factor = 1;

        static Eigen::Matrix<double, 3, 1> X_speed;
        static Eigen::Matrix<double, 3, 3> P_speed;

        Eigen::Matrix<double, 3, 3> F;
        Eigen::Matrix<double, 3, 3> Q; // process noise
        Eigen::Matrix<double, 3, 3> R; // measurement noise
        Eigen::Matrix<double, 3, 3> K;
        Eigen::Matrix<double, 3, 3> H;
        Eigen::Matrix<double, 3, 1> Z;
        Eigen::Matrix3d I = Eigen::Matrix3d::Identity();

        if (!flag1) // initialization
        {
            X_speed << speed.data,
                imu_data.linear_acceleration.x,
                imu_data.linear_acceleration.y;
            P_speed << scaling_factor * I;
            flag1 = 1;
        }

        F << 1, t * cos(yaw), t * sin(yaw),
            0, 1, 0,
            0, 0, 1;
        Q = 0.6 * I; // quantifies the trust on the measurement
        // Q = 0.3 * I; // quantifies the trust on the measurement
        R = 2 * I;   // quantifies the trust on the prediction
        H = I;

        if (isnan(speed.data))
        {
            speed.data = 0.25;
        }

        else if (speed.data > 0.75)
        {
            speed.data = 0.25;
        }

        Z << speed.data,
            imu_data.linear_acceleration.x,
            imu_data.linear_acceleration.y;

        // prediction step
        X_speed = F * X_speed;
        P_speed = F * P_speed * F.transpose() + Q;

        // updation step
        K = P_speed * H.transpose() * (H * P_speed * H.transpose() + R).inverse();
        X_speed = X_speed + K * (Z - H * X_speed);
        P_speed = (I - K * H) * P_speed;

        std_msgs::Float64 speed_msg;
        speed_msg.data = X_speed(0, 0);
        // cout << speed.data << " " << speed_msg.data << endl;
        encoder_pub.publish(speed_msg);
    }

    double seconds()
    {
        using namespace std::chrono;

        auto now = system_clock::now();
        auto since_epoch = now.time_since_epoch();
        auto millisec = duration_cast<milliseconds>(since_epoch);
        return (double)(millisec.count() / 1000.0);
    }
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "fusion_node");
    Fusion object;
    ros::spin();
    return 0;
}