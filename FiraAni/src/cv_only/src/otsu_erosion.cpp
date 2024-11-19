#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/Image.h>
#include <opencv2/opencv.hpp>

class OtsuErosion
{
private:
    ros::NodeHandle nh;
    ros::Publisher erosion_pub;
    ros::Subscriber camera_sub;

    cv_bridge::CvImagePtr cv_ptr;
    cv::Point tl;
    cv::Point bl;
    cv::Point tr;
    cv::Point br;
    cv::Mat M;

    int binary_threshold, h, w;

    double res_x;
    double gamma;

public:
    OtsuErosion() : nh("~")
    {
        set_img_params();

        camera_sub = nh.subscribe<sensor_msgs::Image>("/camera/color/image_raw", 5, &OtsuErosion::camera_cbk, this);
        erosion_pub = nh.advertise<sensor_msgs::Image>("/erosion", 2);
    }

    void set_img_params()
    {
        res_x = 1 / (100 * 0.00140625);
        h = 480;
        w = 640;

        nh.param("/BINARY_THRESHOLD", binary_threshold, 150);
        gamma = nh.param("/GAMMA", 0.10);
        ROS_INFO("BINARY THRESHOLD: %d", binary_threshold);
        ROS_INFO("GAMMA : %f", gamma);

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

    void camera_cbk(const sensor_msgs::ImageConstPtr &msg)
    {
        ros::Time start = ros::Time::now();

        try
        {
            cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        }
        catch (cv_bridge::Exception &e)
        {
            ROS_ERROR("cv_bridge exception: %s", e.what());
            return;
        }

        cv::Mat img = cv_ptr->image;

        cv::Mat img_resized;
        cv::resize(img, img_resized, cv::Size(w, h), 0, 0, cv::INTER_AREA);

        cv::Mat img_warped;
        cv::warpPerspective(img_resized, img_warped, M, cv::Size(w, h), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0));

        cv::Mat img_gray;
        cv::cvtColor(img_warped, img_gray, cv::COLOR_BGR2GRAY);

        int blur_size = 5;
        cv::Mat blurred_img;
        cv::GaussianBlur(img_gray, blurred_img, cv::Size(blur_size, blur_size), 0);

        cv::Mat equalized_img;
        cv::equalizeHist(blurred_img, equalized_img);

        cv::Mat lookupTable(1, 256, CV_8U);
        uchar *p = lookupTable.ptr();
        for (int i = 0; i < 256; ++i)
        {
            p[i] = cv::saturate_cast<uchar>(std::pow(i / 255.0, 1.0 / gamma) * 255.0);
        }
        cv::Mat gamma_img;
        cv::LUT(equalized_img, lookupTable, gamma_img);

        cv::Mat otsu_img;
        cv::threshold(gamma_img, otsu_img, binary_threshold, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);

        cv::imwrite("otsu_erosion.jpg", otsu_img);

        cv_bridge::CvImage out_msg;
        sensor_msgs::ImagePtr erosion_msg = cv_bridge::CvImage(std_msgs::Header(), "mono8", otsu_img).toImageMsg();
        erosion_pub.publish(erosion_msg);

        ros::Time end = ros::Time::now();
        ROS_INFO("Processing time: %f seconds", (end - start).toSec());
    }
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "erosion_node_o");
    OtsuErosion obj;
    ros::spin();
    return 0;
}
