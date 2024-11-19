#include "ros/ros.h"
#include "sensor_msgs/Imu.h"
#include "std_msgs/Float64.h"
#include "std_msgs/Int16.h"
#include <cmath>
#include <chrono>
#include <eigen3/Eigen/Dense>

using namespace std;

class Fusion
{
private:
    ros::NodeHandle nh;
    ros::Publisher imu_pub;
    ros::Publisher encoder_pub;
    ros::Publisher yaw_pub;
    ros::Publisher global_yaw_pub;
    ros::Subscriber imu_sub;
    ros::Subscriber encoder_sub;

    ros::Rate rate;

    std_msgs::Float64 global_speed;
    std_msgs::Float64 prev_speed;

    double prev_yaw;

    double yaw_m_trust, yaw_p_trust;
    double speed_m_trust, speed_p_trust;

public:
    Fusion() : nh("~"), prev_yaw(0.0), rate(nh.param("/dm_node/loop_rate", 10))
    {
        set_pub_sub();
        set_kf_params();
        prev_speed.data = 0;
    }

private:
    void set_pub_sub()
    {
        encoder_sub = nh.subscribe<std_msgs::Int16>("/odom", 2, &Fusion::encoder_callback, this);
        imu_sub = nh.subscribe<sensor_msgs::Imu>("/imu/data", 2, &Fusion::imu_callback, this);
        imu_pub = nh.advertise<std_msgs::Float64>("/filtered_yaw", 2);
        encoder_pub = nh.advertise<std_msgs::Float64>("/final_speed", 2);
        yaw_pub = nh.advertise<std_msgs::Float64>("/raw_yaw", 2);
    	global_yaw_pub = nh.advertise<std_msgs::Float64>("/global_yaw", 2);
    }

    void set_kf_params()
    {
        nh.getParam("/YAW_M_TRUST", yaw_m_trust);
        nh.getParam("/YAW_P_TRUST", yaw_p_trust);

        nh.getParam("/SPEED_M_TRUST", speed_m_trust);
        nh.getParam("/SPEED_P_TRUST", speed_p_trust);
    }

    void imu_callback(const sensor_msgs::ImuConstPtr &msg)
    {
        std_msgs::Float64 speed = global_speed;

        static double current_t;
        static double t;
        static bool flag2 = 0;

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
        double b = 0.01; // Bias
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

	rate.sleep();

        t = seconds() - current_t;
        current_t = seconds();
        m_yaw = atan2(2 * (msg->orientation.w * msg->orientation.z + msg->orientation.x * msg->orientation.y), 1 - 2 * (msg->orientation.y * msg->orientation.y + msg->orientation.z * msg->orientation.z));
        w = msg->angular_velocity.z;

        Q = yaw_m_trust * I3; // quantifies the trust on the measurement
        R = yaw_p_trust * I2; // quantifies the trust on our prediction
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

        std_msgs::Float64 yaw;
        std_msgs::Float64 measured_yaw;
        measured_yaw.data = m_yaw;
        // yaw.data = X_yaw(0, 0);
        yaw.data = X_yaw(0, 0) - prev_yaw;
        prev_yaw = X_yaw(0, 0);
        imu_pub.publish(yaw);
        yaw_pub.publish(measured_yaw);

	std_msgs::Float64 global_yaw_msg;
	global_yaw_msg.data = X_yaw(0, 0);
	global_yaw_pub.publish(global_yaw_msg);


        ROS_INFO("KF YAW: %f, M YAW: %f", yaw.data, prev_yaw);

        // std::cout << speed.data << std::endl;

        speed_calculate(*msg, speed, current_t, t, yaw.data);
    }

    void encoder_callback(const std_msgs::Int16ConstPtr &speed)
    {
        double float_speed = speed->data / (1000.0);


        if (float_speed < 1)
        {
            global_speed.data = float_speed;
        }
        else
        {
            global_speed.data = prev_speed.data;
        }
    }

    void speed_calculate(const sensor_msgs::Imu imu_data, std_msgs::Float64 speed, double current_t, double t, double yaw)
    {
        prev_speed.data = speed.data;

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
        Q = speed_m_trust * I; // quantifies the trust on the measurement
        R = speed_p_trust * I; // quantifies the trust on the prediction
        H = I;
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
        speed_msg.data = nh.param("/DUMMY_SPEED", 0.25);
        // speed_msg.data = X_speed(0, 0);
        ROS_INFO("KF SPEED: %f, M SPEED: %f", speed_msg.data, speed.data);
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
