#include <ros/ros.h>
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/Image.h>
#include <std_msgs/Bool.h>
#include <cmath>
#include <eigen3/Eigen/Dense>
#include <chrono>

using namespace std;

class Stopline
{
private:
    ros::NodeHandle nh;
    ros::Publisher stopline_pub;
    ros::Subscriber img_sub;

    ros::Rate _rate;

    cv_bridge::CvImagePtr cv_ptr;
    cv::Point tl;
    cv::Point bl;
    cv::Point tr;
    cv::Point br;
    cv::Mat M;

    int h, w, phy_length, binary_threshold;
    double pxls_per_cm;
    double res_x;

public:
    Stopline() : nh("~"), _rate(nh.param("/dm_node/loop_rate", 10))
    {
        set_img_params();
        stopline_pub = nh.advertise<std_msgs::Bool>("/stopline_flag", 5);
        img_sub = nh.subscribe<sensor_msgs::Image>("/camera/color/image_raw", 5, &Stopline::stopline_cbk, this);
    }

private:
    void set_img_params()
    {
        binary_threshold = nh.param("/BINARY_THRESHOLD", 170);

        h = 480;
        w = 640;
        phy_length = 60;
        res_x = 1 / (100 * 0.00140625);
        pxls_per_cm = double(h) / phy_length;

        tl = cv::Point(nh.param("/TL_X", 120), nh.param("/T_Y", 350));
        bl = cv::Point(nh.param("/BL_X", -110), nh.param("/B_Y", 480));
        tr = cv::Point(nh.param("/TR_X", 350), nh.param("/T_Y", 350));
        br = cv::Point(nh.param("/BR_X", 390), nh.param("/B_Y", 480));

        cv::Point2f src_pts[] = {bl, br, tr, tl};
        cv::Point2f dest_pts[] = {
            cv::Point2f(w / 2 - 40 * res_x, h),
            cv::Point2f(w / 2 + 20 * res_x, h),
            cv::Point2f(w / 2 + 20 * res_x, 0),
            cv::Point2f(w / 2 - 40 * res_x, 0)};

        M = cv::getPerspectiveTransform(src_pts, dest_pts);
    }

    void stopline_cbk(const sensor_msgs::ImageConstPtr &msg)
    {
        double start = get_time();
        cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        cv::Mat img = cv_ptr->image;
        cv::Mat erosion = process_img(img);

        remove_stopline(erosion);

        double end = get_time();

        std::cout << "RUNTIME: " << end - start << endl << endl;
    }

    void remove_stopline(const cv::Mat &img)
    {
        int stopline_threshold = nh.param("/STOPLINE_THRES", 100);
        std_msgs::Bool msg;

        Eigen::VectorXd histogram = Eigen::VectorXd::Zero(img.rows);
        for (int row = 0; row < img.rows; ++row)
        {
            for (int col = img.cols / 4; col < img.cols * 3.0 / 4; ++col)
            {
                histogram(row) += img.at<uchar>(row, col) / 255;
            }

            //std::cout << static_cast<int>(histogram(row)) << endl << endl;

            if (histogram(row) > stopline_threshold)
            {
                std::cout << "STOPLINE!!! NO. OF PTS: " << static_cast<int>(histogram(row)) << "\n\n\n";
                msg.data = true;
                stopline_pub.publish(msg);
                return;
            }
        }
    }

    cv::Mat process_img(const cv::Mat &img)
    {
        cv::Mat img_resized;
        cv::resize(img, img_resized, cv::Size(w, h), 0, 0, cv::INTER_AREA);

        // Initialization of Image params
        // cv::circle(img_resized, tl, 5, cv::Scalar(0, 0, 255), 5);
        // cv::circle(img_resized, bl, 5, cv::Scalar(0, 0, 255), 5);
        // cv::circle(img_resized, tr, 5, cv::Scalar(0, 0, 255), 5);
        // cv::circle(img_resized, br, 5, cv::Scalar(0, 0, 255), 5);

        cv::Mat img_warped;
        cv::warpPerspective(img_resized, img_warped, M, cv::Size(w, h), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0));

        cv::Mat img_gray;
        cv::cvtColor(img_warped, img_gray, cv::COLOR_BGR2GRAY);

        cv::Mat edges; // = applyGammaCorrectionAndOtsuThreshold(img_warped);
        cv::threshold(img_gray, edges, binary_threshold, 255, cv::THRESH_BINARY);

        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
        cv::Mat erosion;
        cv::erode(edges, erosion, kernel, cv::Point(-1, -1), 8); // -1, -1 Selects the center or something, last argument is number of iterations

        int blacken_pixels = 5;
        erosion.colRange(0, blacken_pixels).setTo(cv::Scalar(0));
        erosion.colRange(erosion.cols - blacken_pixels, erosion.cols).setTo(cv::Scalar(0));
        return erosion;
    }

    double get_time()
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
    ros::init(argc, argv, "Stopline_node");
    ROS_INFO("STOPLINE NODE HAS BEEN LAUNCHED");
    Stopline obj;
    ros::spin();
    return 0;
}
