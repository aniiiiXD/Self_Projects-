#include <ros/ros.h>
#include <geometry_msgs/PoseArray.h>
#include <std_msgs/Float64.h>
#include <std_msgs/Bool.h>
#include <geometry_msgs/Twist.h>
#include <vector>
#include <cmath>
#include <chrono>
#include <iostream>

class Controller {
public:
    Controller(ros::NodeHandle& nh) : nh_(nh), publish_enabled_(true), turn_flag_(false),stopline_seen(false){
        nh_.getParam("/controls_node/q", q_);
        nh_.getParam("/controls_node/m", m_);
        nh_.getParam("/controls_node/t_clip_min", t_clip_min_);
        nh_.getParam("/controls_node/t_clip_max", t_clip_max_);
        nh_.getParam("/controls_node/wheelbase", wheelbase_);
        nh_.getParam("/controls_node/loop_rate", loop_rate_);
        nh_.getParam("/controls_node/agression_1", aggression_1_);
        nh_.getParam("/controls_node/agression_2", aggression_2_);
        nh_.getParam("/controls_node/agression_3", aggression_3_);
        nh_.getParam("/controls_node/steer_bound", steer_bound_);
        nh_.getParam("/controls_node/velocity_boost", velocity_boost_);
        nh_.getParam("/controls_node/steer_threshold_1", steer_threshold_1_);
        nh_.getParam("/controls_node/steer_threshold_2", steer_threshold_2_);



        velocity_pub_ = nh_.advertise<std_msgs::Float64>("/velocity", 1);
            // if (!publish_enabled_) {
            //     velocity_msg.data = 0;
            // }
        steer_pub_ = nh_.advertise<std_msgs::Float64>("/steer", 1);

        wp_sub_ = nh_.subscribe("/best_trajectory", 1, &Controller::wp_cb, this);
        stopline_sub_ = nh_.subscribe("/stopline_flag", 1, &Controller::stopline_cb, this);
        on_off_sub_ = nh_.subscribe("/on_off", 1, &Controller::on_off_cb, this);
        turn_rad_sub_ = nh_.subscribe("/turn_rad", 1, &Controller::turn_rad_cb, this);
    }

    void control_loop() {
        ros::Rate rate(loop_rate_);

        while (ros::ok()) {
            auto start_time = std::chrono::steady_clock::now();  // Start time

            if (wps_.empty()) {
                ros::spinOnce();
                rate.sleep();
                continue;
            }
            if(!turn_flag_ ){
                double v_target = wps_[0][2];
                double lookahead_distance = m_ * v_target + q_;
                lookahead_distance = std::clamp(lookahead_distance, t_clip_min_, t_clip_max_);

                ROS_INFO("ld: %f", lookahead_distance);

                auto lookahead_point = lookaheadpoint(lookahead_distance);

                ROS_INFO("x: %f y: %f", lookahead_point[0], lookahead_point[1]);

                double steering_angle = get_actuation(lookahead_point);
                double aggression = interpolate_aggression(steering_angle);
                steering_angle = std::clamp(std::atan(steering_angle) * 180.0 / M_PI, -steer_bound_, steer_bound_) * aggression;

                if (std::abs(v_target) <= 0.1) {
                    steering_angle = 0;
                }

                ROS_INFO("velocity: %f", v_target);
                ROS_INFO("steer: %f", steering_angle);
                //float v_correction_coeff=1-sqrt(abs(steering_angle/0.7)*steer_bound_);
                float v_correction_coeff=0.5;
                std_msgs::Float64 velocity_msg;
                if (velocity_boost_ == -1) {
                    velocity_msg.data = v_target * (v_correction_coeff);
                } else {
                    velocity_msg.data = velocity_boost_;
                }
                if (!publish_enabled_ ) {
                    velocity_msg.data = -1;
                }
                // velocity_msg.data = -1;
                velocity_pub_.publish(velocity_msg);
                std_msgs::Float64 steer_msg;
                steer_msg.data = steering_angle;
                steer_pub_.publish(steer_msg);
            }

            else{
                double required_time=ttime;
                if(init_time==-1){
                    init_time =ros::Time::now().toSec();
                }
                double v_target =turn_speed_;
                double steering_angle = turn_steering_angle_;
                double aggression = interpolate_aggression(steering_angle);
                // steering_angle = std::clamp(std::atan(steering_angle) * 180.0 / M_PI, -steer_bound_, steer_bound_) ;
                if (std::abs(v_target) <= 0.01) {
                steering_angle = 0;
                }
                ROS_INFO("turning");
                ROS_INFO("Req time: %f", required_time);
                double e=ros::Time::now().toSec()-init_time;
                ROS_INFO("Elapsed time: %f",e);
                std_msgs::Float64 velocity_msg;
                if (velocity_boost_ == -1) {
                    velocity_msg.data = v_target ;
                } else {
                    velocity_msg.data = velocity_boost_;
                }
                if (!publish_enabled_) {
                    velocity_msg.data = -1;
                }
               

                velocity_pub_.publish(velocity_msg);
                std_msgs::Float64 steer_msg;
                steer_msg.data = steering_angle;
                steer_pub_.publish(steer_msg);
                if(ros::Time::now().toSec()-init_time>=required_time){
                    turn_flag_=false;
                    init_time=-1.0;
                    last_time=ros::Time::now().toSec();
                }
            }

            auto end_time = std::chrono::steady_clock::now();  // End time
            std::chrono::duration<double> elapsed_seconds = end_time - start_time;
            ROS_INFO("Loop runtime: %f seconds\n", elapsed_seconds.count());

            ros::spinOnce();
            rate.sleep();
        }
    }

private:
    ros::NodeHandle& nh_;
    ros::Publisher cmd_vel_pub_;  // Publisher for cmd_vel
    ros::Publisher velocity_pub_;
    ros::Publisher steer_pub_;
    ros::Subscriber wp_sub_;
    ros::Subscriber stopline_sub_;
    ros::Subscriber on_off_sub_;
    ros::Subscriber turn_rad_sub_;

    double q_, m_, t_clip_min_, t_clip_max_, wheelbase_, loop_rate_, aggression_1_, aggression_2_, aggression_3_, steer_bound_, velocity_boost_;
    double steer_threshold_1_, steer_threshold_2_;
    std::vector<std::vector<double>> wps_;
    bool publish_enabled_;
    bool turn_flag_;
    bool stopline_seen;
    double turn_radius_;
    double turn_speed_;
    double turn_steering_angle_;
    double init_time=-1.0;
    double last_time=-1.0;
    double ttime=-1.0;


    void wp_cb(const geometry_msgs::PoseArray::ConstPtr& data) {
        wps_.clear();
        for (const auto& pose : data->poses) {
            double x = pose.position.x;
            double y = pose.position.y;
            double v = pose.position.z;
            double w = pose.orientation.w;
            wps_.push_back({x, y, v, w});
        }
    }

    void stopline_cb(const std_msgs::Bool::ConstPtr& msg) {
        if(msg->data==true){
        ROS_INFO("                         \
                                            \
                                             STOOOOP");
        }
        stopline_seen=msg->data;
    }

    void on_off_cb(const geometry_msgs::Twist::ConstPtr& msg) {
        if (msg->linear.x == 0 && msg->linear.y == 0 && msg->linear.z == 0 &&
            msg->angular.x == 0 && msg->angular.y == 0 && msg->angular.z == 0) {
            publish_enabled_ = !publish_enabled_;
            ROS_INFO("Publishing %s", publish_enabled_ ? "enabled" : "disabled");
        }
    }
    void turn_rad_cb(const std_msgs::Float64::ConstPtr& msg) {
        if(turn_flag_==true){
            return;
        }
        ttime=msg->data;
        turn_radius_ = 1.55;
        turn_speed_ = 0.5;  // Set your desired turning speed here
        // turn_steering_angle_ = std::atan(wheelbase_ / turn_radius_);
        turn_steering_angle_=25;
        turn_flag_ = true;
        ROS_INFO("Turning with radius: %f and speed: %f", turn_radius_, turn_speed_);
    }

    double e_distance(const std::vector<double>& arrA, const std::vector<double>& arrB) {
        return std::sqrt(std::pow(arrA[0] - arrB[0], 2) + std::pow(arrA[1] - arrB[1], 2));
    }

    std::vector<double> lookaheadpoint(double distance) {
        double dist = 0;
        int i = 0;
        if(e_distance({wps_[wps_.size()-1][0], wps_[wps_.size()-2][1]}, {wps_[0][0], wps_[0][1]}) <0.001){
            ROS_INFO("MP fucked up");
        }
        while (dist < distance) {
            i = (i + 1) % wps_.size();
            if (i == 0) {
                ROS_INFO("REDUCE LOOKAHEAD DISTANCE");
                break;
            }
            dist = e_distance({wps_[i][0], wps_[i][1]}, {wps_[0][0], wps_[0][1]});
        }
        return {wps_[i][0], wps_[i][1]};
    }

    double get_actuation(const std::vector<double>& lookahead_point) {
        double waypoint_y = lookahead_point[1];
        if (std::abs(waypoint_y) < 1e-6) {
            return 0;
        }
        double radius = std::pow(0.15 + std::hypot(lookahead_point[0], lookahead_point[1]), 2) / (2.0 * waypoint_y);
        double steering_angle = std::atan(wheelbase_ / radius);
        return steering_angle;
    }

    double interpolate_aggression(double steering_angle) {
        double abs_steering_angle = std::abs(steering_angle);
        if (abs_steering_angle <= steer_threshold_1_) {
            return aggression_1_;
        } else if (abs_steering_angle <= steer_threshold_2_) {
            double t = (abs_steering_angle - steer_threshold_1_) / (steer_threshold_2_ - steer_threshold_1_);
            return aggression_1_ + t * (aggression_2_ - aggression_1_);
        } else {
            return aggression_3_;
        }
    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "controller_node");
    ros::NodeHandle nh;

    Controller controller(nh);
    controller.control_loop();

    return 0;
}
