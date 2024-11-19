#include <ros/ros.h>
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/Image.h>
#include <std_msgs/Bool.h>
#include <std_msgs/Float64.h>
#include <std_msgs/Int16.h>
#include <std_msgs/Float64MultiArray.h>
#include <std_msgs/Int64MultiArray.h>
#include <cv_only/LaneCoordinates.h>
#include <cmath>
#include <chrono>
#include <eigen3/Eigen/Dense>

using namespace std;

class SlidingWindow
{
private:
    bool racing = false;

    ros::NodeHandle nh;

    ros::Publisher lane_img_pub;
    ros::Publisher lane_coords_pub;
    ros::Publisher erosion_pub;
    ros::Publisher occ_points_pub;
    // ros::Publisher stopline_flag_pub;
    // extras
    // ros::Publisher hist_pub;
    // ros::Publisher max_pub;
    // ros::Publisher ipm_pub;

    ros::Subscriber camera_sub;
    ros::Subscriber yaw_sub;
    ros::Subscriber steer_sub;
    ros::Subscriber encoder_sub;
    ros::Subscriber reset_sub;

    ros::Subscriber erosion_sub;

    ros::Rate rate;

    std_msgs::Float64 global_yaw, global_speed, global_steer;

    cv_bridge::CvImagePtr cv_ptr;
    cv::Point tl;
    cv::Point bl;
    cv::Point tr;
    cv::Point br;
    cv::Mat M;

    int h,
        w, phy_length, binary_threshold;
    double pxls_per_cm;
    double res_x, res_y;

    int nwin, minpix, margin;
    double win_ht;

    bool initial, reset;

    int l_x, l_y, r_x, r_y;

    int cam_ctr;
    int init_maximas[2];

    double prev_time;
    double iterations;

    int lane_dist;
    int min_dist, max_dist;

    int false_ctr, true_ctr;
    int false_threshold;

    double gamma;

    bool left_empty, right_empty;

public:
    SlidingWindow() : nh("~"), initial(false), rate(nh.param("/dm_node/loop_rate", 10)), cam_ctr(0), init_maximas{0, 0}, reset(true), iterations(5), false_ctr(0), true_ctr(0), false_threshold(nh.param("/FALSE_THRES", 2)), left_empty(false), right_empty(false)
    {
        set_pub_sub();
        set_img_params();
        set_win_params();

        prev_time = get_time();
    }

private:
    void set_pub_sub()
    {
        lane_img_pub = nh.advertise<sensor_msgs::Image>("/lane_img", 2);
        lane_coords_pub = nh.advertise<cv_only::LaneCoordinates>("/lane_coordinates", 5); // Change to cv_only
        occ_points_pub = nh.advertise<std_msgs::Float64MultiArray>("/occupancy_points", 5);
        // stopline_flag_pub = nh.advertise<std_msgs::Bool>("/stopline_flag", 10);
        // extras
        // hist_pub = nh.advertise<std_msgs::Float64MultiArray>("/histogram", 5);
        // max_pub = nh.advertise<std_msgs::Float64MultiArray>("/maximas", 5);
        // ipm_pub = nh.advertise<sensor_msgs::Image>("ipm_img", 5);

        erosion_sub = nh.subscribe<sensor_msgs::Image>("/erosion", 5, &SlidingWindow::erosion_cbk, this);
        yaw_sub = nh.subscribe<std_msgs::Float64>("/filtered_yaw", 5, &SlidingWindow::yaw_cbk, this);
        steer_sub = nh.subscribe<std_msgs::Int16>("/steer", 5, &SlidingWindow::steer_cbk, this);
        encoder_sub = nh.subscribe<std_msgs::Float64>("/final_speed", 5, &SlidingWindow::encoder_cbk, this);
        reset_sub = nh.subscribe<std_msgs::Bool>("/reset_cv", 5, &SlidingWindow::reset_cbk, this);
    }

    void set_img_params()
    {
        binary_threshold = nh.param("/BINARY_THRESHOLD", 100);
        gamma = nh.param("/GAMMA", 0.15);

        h = 480;
        w = 640;
        phy_length = 60;
        res_x = 1 / (100 * 0.00140625);
        pxls_per_cm = double(h) / phy_length;

        lane_dist = nh.param("/LANE_DIST", 58);
        min_dist = lane_dist * res_x;
        max_dist = min_dist + 5;

        tl = cv::Point(nh.param("/TL_X", 120), nh.param("/T_Y", 350));
        bl = cv::Point(nh.param("/BL_X", -110), nh.param("/B_Y", 480));
        tr = cv::Point(nh.param("/TR_X", 350), nh.param("/T_Y", 350));
        br = cv::Point(nh.param("/BR_X", 390), nh.param("/B_Y", 480));

        cv::Point2f src_pts[] = {bl, br, tr, tl};

        if (!racing)
        {
            cv::Point2f dest_pts[] = {
                cv::Point2f(w / 2 - 40 * res_x, h),
                cv::Point2f(w / 2 + 20 * res_x, h),
                cv::Point2f(w / 2 + 20 * res_x, 0),
                cv::Point2f(w / 2 - 40 * res_x, 0)};
            M = cv::getPerspectiveTransform(src_pts, dest_pts);
        }
        else
        {
            cv::Point2f dest_pts[] = {
                cv::Point2f(w / 2 - 38 * res_x, h),
                cv::Point2f(w / 2 + 14 * res_x, h),
                cv::Point2f(w / 2 + 14 * res_x, 0),
                cv::Point2f(w / 2 - 38 * res_x, 0)};

            M = cv::getPerspectiveTransform(src_pts, dest_pts);
        }
    }

    void set_win_params()
    {
        nwin = nh.param("/NWINDOWS", 30);
        minpix = nh.param("/MINPIX", 50);
        margin = nh.param("/MARGIN", 70);
        win_ht = (double)h / nwin;
        std::cout << win_ht << endl;
    }

    void reset_cbk(const std_msgs::BoolConstPtr &msg)
    {
        reset = msg->data;
    }

    void yaw_cbk(const std_msgs::Float64ConstPtr &msg)
    {
        global_yaw = *msg;
    }

    void encoder_cbk(const std_msgs::Float64ConstPtr &msg)
    {
        global_speed = *msg;
    }

    void steer_cbk(const std_msgs::Int16ConstPtr &msg)
    {
        global_steer.data = (*msg).data / 10.0;
    }

    void erosion_cbk(const sensor_msgs::ImageConstPtr &msg)
    {
        double start = get_time();

        cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::MONO8);
        cout << "RECEIVED\n";
        cv::Mat erosion = cv_ptr->image;
        cout << "COPIED IMAGE\n";

        std_msgs::Float64 yaw, speed, steer;
        yaw = global_yaw;
        speed = global_speed;
        steer = global_steer;

        cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        cv::Mat img = cv_ptr->image;

        // checkIPM(img);

        std::vector<int> maximas;

        if (reset)
        {
            cam_ctr++;
            if (cam_ctr < iterations)
            {
                std::cout << "CTR: " << cam_ctr << endl;
                std::cout << get_init_maximas(erosion)[0] << " " << get_init_maximas(erosion)[1] << endl;
                init_maximas[0] += get_init_maximas(erosion)[0];
                init_maximas[1] += get_init_maximas(erosion)[1];
                return;
            }

            else if (cam_ctr == iterations)
            {
                std::cout << "CTR: " << cam_ctr << endl;
                ROS_INFO("INIT");
                std::cout << "ITERATIONS: " << iterations << "\n";
                maximas.push_back(init_maximas[0] / (iterations - 1));
                maximas.push_back(init_maximas[1] / (iterations - 1));

                std::cout << "LEFT: " << maximas[0] << " "
                          << "RIGHT: " << maximas[1] << std::endl;

                l_x = maximas[0];
                l_y = 0;
                r_x = maximas[1];
                r_y = 0;

                reset = false;
                cam_ctr = 0;
                init_maximas[0] = 0;
                init_maximas[1] = 0;
            }
        }

        else
        {
            maximas = estimate_maximas(erosion, yaw, speed, steer, start - prev_time);
            prev_time = start;
            std::cout << "LEFT: " << maximas[0] << " " << l_y << endl;
            std::cout << "RIGHT: " << maximas[1] << " " << r_y << endl
                      << endl;
        }

        // cv::Mat lane_img = check_erosion(erosion, yaw, speed, steer, start - prev_time);
        // sensor_msgs::ImagePtr lane_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", lane_img).toImageMsg();
        // lane_img_pub.publish(lane_msg);

        cv::Mat lane_img = cv::Mat::zeros(erosion.size(), img.type());
        std_msgs::Float64MultiArray occ_pts;
        cv_only::LaneCoordinates lane_coords;

        std::vector<int> right_ind = make_red(lane_img, erosion, maximas[1], occ_pts, lane_coords);
        std::vector<int> left_ind = make_blue(lane_img, erosion, maximas[0], occ_pts, lane_coords);
        cv::imwrite("lane_image.jpg", lane_img);

        sensor_msgs::ImagePtr lane_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", lane_img).toImageMsg();
        lane_img_pub.publish(lane_msg);
        occ_points_pub.publish(occ_pts);
        lane_coords_pub.publish(lane_coords);

        rate.sleep();

        double end = get_time();
        double time = end - start;

        ROS_INFO("TIME: %f", time);
    }

    cv::Mat process_img(const cv::Mat &img)
    {
        cv::Mat img_resized;
        cv::resize(img, img_resized, cv::Size(w, h), 0, 0, cv::INTER_AREA);

        cv::Mat img_warped;
        cv::warpPerspective(img_resized, img_warped, M, cv::Size(w, h), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0));

        cv::Mat img_gray;
        cv::cvtColor(img_warped, img_gray, cv::COLOR_BGR2GRAY);

        cv::Mat edges;
        cv::threshold(img_gray, edges, binary_threshold, 255, cv::THRESH_BINARY);

        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
        cv::Mat erosion;
        cv::erode(edges, erosion, kernel, cv::Point(-1, -1), 8); // -1, -1 Selects the center or something, last argument is number of iterations

        int blacken_pixels = 5;
        erosion.colRange(0, blacken_pixels).setTo(cv::Scalar(0));
        erosion.colRange(erosion.cols - blacken_pixels, erosion.cols).setTo(cv::Scalar(0));
        return erosion;
    }

    cv::Mat applyNewProcessing(const cv::Mat &img)
    {
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

        // CAN REMOVE TOMORROW!!!!!!!!!!!!!!!!!!!1111
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
        cv::Mat erosion;
        cv::erode(otsu_img, erosion, kernel, cv::Point(-1, -1), 15); // -1, -1 Selects the center or something, last argument is number of iterations

        return erosion;
    }

    std::vector<int> get_init_maximas(const cv::Mat &img)
    {
        // Histogram Calculation
        static double img_ratio = nh.param("/IMG_RATIO", 1.0 / 3);
        cv::Mat ROI = img(cv::Range(img.rows * (1 - img_ratio), img.rows), cv::Range(0, img.cols));
        cv::Mat ROI_double;
        ROI.convertTo(ROI_double, CV_64F);
        ROI_double /= 255.0;
        Eigen::VectorXd histogram = Eigen::VectorXd::Zero(ROI_double.cols);
        for (int col = 0; col < ROI_double.cols; ++col)
        {
            for (int row = 0; row < ROI_double.rows; ++row)
            {
                histogram(col) += ROI_double.at<double>(row, col);
            }
        }

        // Histogram Smoothening
        int sigma = 5;
        int kernel_radius = static_cast<int>(sigma * 3);
        int kernel_size = 2 * kernel_radius + 1;
        Eigen::VectorXd kernel(kernel_size);
        double sum = 0.0;

        for (int i = 0; i < kernel_size; ++i)
        {
            int x = i - kernel_radius;
            kernel(i) = std::exp(-(x * x) / (2 * sigma * sigma));
            sum += kernel(i);
        }
        kernel /= sum;
        Eigen::VectorXd smooth_histogram = Eigen::VectorXd::Zero(histogram.size());
        for (int i = 0; i < histogram.size(); ++i)
        {
            for (int j = 0; j < kernel_size; ++j)
            {
                int idx = i + j - kernel_radius;
                if (idx >= 0 && idx < histogram.size())
                {
                    smooth_histogram(i) += histogram(idx) * kernel(j);
                }
            }
        }

        int tolerance = 4;
        std::vector<int> maximas;

        for (int i = tolerance; i < smooth_histogram.size() - tolerance; i++)
        {
            int curr = smooth_histogram[i];
            int prev = smooth_histogram[i - tolerance];
            int next = smooth_histogram[i + tolerance];

            if (curr > prev && curr > next)
            {
                maximas.push_back(i);
            }
        }

        std::vector<int> filtered_max;
        int prev = maximas[0];

        filtered_max.push_back(prev);

        for (int i = 1; i < maximas.size(); i++)
        {
            int element = maximas[i];
            if (element - prev >= 40)
                filtered_max.push_back(element);
            prev = element;
        }

        // publishHist(smooth_histogram); // Use with histogram script, to see first frame
        int midpoint = smooth_histogram.size() / 2;

        maximas = {filtered_max[0], filtered_max[filtered_max.size() - 1]};
        return maximas;
    }

    std::vector<int> estimate_maximas(const cv::Mat &img, const std_msgs::Float64 yaw, const std_msgs::Float64 speed, const std_msgs::Float64 steer, double t)
    {
        Eigen::Matrix<double, 2, 2> Rotational;
        Rotational << cos(yaw.data), -sin(yaw.data),
            sin(yaw.data), cos(yaw.data);

        double d = speed.data * t * 800.0;

        Eigen::Matrix<double, 2, 1> Translational;
        Translational << (-d / 2.0) * sin(2 * steer.data),
            (d)*cos(steer.data) * cos(steer.data);

        Eigen::Matrix<double, 2, 1> P_L, P_R;
        Eigen::Matrix<double, 2, 1> new_P_L, new_P_R;

        P_L << l_x, l_y;
        P_R << r_x, r_y;
        new_P_L = Rotational * P_L + Translational;
        new_P_R = Rotational * P_R + Translational;
        l_x = int(new_P_L(0, 0));
        l_y = int(new_P_L(1, 0));
        r_x = int(new_P_R(0, 0));
        r_y = int(new_P_R(1, 0));

        int base_dist = abs(l_x - r_x);
        if (base_dist <= min_dist)
        {
            int push_dist = abs(min_dist - base_dist) / 2;

            if (left_empty)
            {
                l_x -= 2 * push_dist;
                cout << "BLUE PUSHED TO LEFT BY: " << 4 * push_dist << endl;
            }
            else if (right_empty)
            {
                r_x += 2 * push_dist;
                cout << "RED PUSHED TO RIGHT BY: " << 4 * push_dist << endl;
            }
            else
            {
                l_x -= push_dist;
                r_x += push_dist;
                cout << "BOTH PUSHED AWAY BY: " << push_dist << endl;
            }
        }

        else if (base_dist >= max_dist)
        {
            int push_dist = abs(max_dist - base_dist) / 2;

            if (left_empty)
            {
                l_x += 2 * push_dist;
                cout << "BLUE PUSHED TO RIGHT BY: " << 2 * push_dist << endl;
            }
            else if (right_empty)
            {
                r_x -= 2 * push_dist;
                cout << "RED PUSHED TO LEFT BY: " << 2 * push_dist << endl;
            }
            else
            {
                l_x += push_dist;
                r_x -= push_dist;
                cout << "BOTH PUSHED CLOSER BY: " << push_dist << endl;
            }
        }

        std::vector<int> maximas = {l_x, r_x};

        return maximas;
    }

    std::vector<int> make_blue(cv::Mat &lane_img, const cv::Mat &erosion_img, int base, std_msgs::Float64MultiArray &occ_pts, cv_only::LaneCoordinates &lane_coords)
    {
        int y_low, y_high;
        int x_low, x_high;

        std::vector<int> lane_ind;
        std::vector<int> win_pxl;

        int curr_x = base;
        int win_ctr = 0;
        int empty_ctr = 0;

        cv::circle(lane_img, cv::Point(base, 475), 10, cv::Scalar(0, 255, 0), -1);

        left_empty = false;

        for (int win = 0; win < nwin; win++)
        {
            int ctr = 0;
            double avg = 0.0;
            win_pxl.clear();

            y_low = h - (win + 1) * win_ht; // Numerically lower
            y_high = h - win * win_ht;
            x_high = curr_x + margin;
            x_low = curr_x - margin;

            // cv::circle(lane_img, cv::Point(curr_x, (y_low + y_high) / 2), 4, cv::Scalar(255, 0, 0), -1);
            cv::rectangle(lane_img, cv::Rect(x_low, y_low, 2 * margin, win_ht), cv::Scalar(255, 0, 0), 2);

            for (int y = y_high; y >= y_low; y--)
            {
                std::vector<int> x_array;
                for (int x = x_low; x <= x_high; x++)
                {

                    if (x >= 0 && x < w && y >= 0 && y < h)
                        if (erosion_img.at<uchar>(y, x) == 255)
                        {
                            x_array.push_back(x);
                            avg = (avg * ctr + x) / ++ctr;
                        }
                }
                if (x_array.size())
                {
                    int x_avg = (x_array[0] + x_array[x_array.size() - 1]) / 2;
                    lane_img.at<cv::Vec3b>(y, x_avg) = cv::Vec3b(255, 0, 0);
                    win_pxl.push_back(x_avg);
                    win_pxl.push_back(y);
                }
            }

            if (ctr >= minpix)
            {
                win_ctr += 1;
                lane_ind.insert(lane_ind.end(), win_pxl.begin(), win_pxl.end());
                curr_x = (int)avg;

                for (int i = 0; i < win_pxl.size(); i += 2)
                {
                    occ_pts.data.push_back(win_pxl[i]);
                    occ_pts.data.push_back(win_pxl[i + 1]);
                    lane_coords.lx.push_back(win_pxl[i]);
                    lane_coords.ly.push_back(win_pxl[i + 1]);
                }

                if (win_ctr == 1)
                {
                    l_x = win_pxl[0];
                    l_y = h - win_pxl[1];
                }
            }
            else
            {
                empty_ctr++;
            }

            if (empty_ctr >= nwin * 0.2)
            {
                left_empty = true;
                break;
            }
        }

        return lane_ind;
    }

    std::vector<int> make_red(cv::Mat &lane_img, const cv::Mat &erosion_img, int base, std_msgs::Float64MultiArray &occ_pts, cv_only::LaneCoordinates &lane_coords)
    {
        int y_low, y_high;
        int x_low, x_high;

        std::vector<int> lane_ind;
        std::vector<int> win_pxl;

        int curr_x = base;
        int win_ctr = 0;
        int empty_ctr = 0;

        cv::circle(lane_img, cv::Point(base, 475), 10, cv::Scalar(0, 255, 0), -1);

        right_empty = false;

        for (int win = 0; win < nwin; win++)
        {
            int ctr = 0;
            double avg = 0.0;
            win_pxl.clear();

            y_low = h - (win + 1) * win_ht; // Numerically lower
            y_high = h - win * win_ht;
            x_high = curr_x + margin;
            x_low = curr_x - margin;

            // cv::circle(lane_img, cv::Point(curr_x, (y_low + y_high) / 2), 4, cv::Scalar(0, 0, 255), -1);
            cv::rectangle(lane_img, cv::Rect(x_low, y_low, 2 * margin, win_ht), cv::Scalar(0, 0, 255), 2);

            for (int y = y_high; y >= y_low; y--)
            {
                std::vector<int> x_array;
                for (int x = x_low; x <= x_high; x++)
                {
                    if (x >= 0 && x < w && y >= 0 && y < h)
                        if (erosion_img.at<uchar>(y, x) == 255)
                        {
                            x_array.push_back(x);
                            avg = (avg * ctr + x) / ++ctr;
                        }
                }
                if (x_array.size())
                {
                    int x_avg = (x_array[0] + x_array[x_array.size() - 1]) / 2;
                    lane_img.at<cv::Vec3b>(y, x_avg) = cv::Vec3b(0, 0, 255);
                    win_pxl.push_back(x_avg);
                    win_pxl.push_back(y);
                }
            }

            if (ctr >= minpix)
            {
                win_ctr += 1;
                lane_ind.insert(lane_ind.end(), win_pxl.begin(), win_pxl.end());
                curr_x = (int)avg;

                for (int i = 0; i < win_pxl.size(); i += 2)
                {
                    occ_pts.data.push_back(win_pxl[i]);
                    occ_pts.data.push_back(win_pxl[i + 1]);
                    lane_coords.rx.push_back(win_pxl[i]);
                    lane_coords.ry.push_back(win_pxl[i + 1]);
                }

                if (win_ctr == 1)
                {
                    r_x = win_pxl[0];
                    r_y = h - win_pxl[1];
                }
            }
            else
            {
                empty_ctr++;
            }

            if (empty_ctr >= nwin * 0.2)
            {
                right_empty = true;
                break;
            }
        }
        return lane_ind;
    }

    // void publishHist(Eigen::VectorXd smooth_histogram)
    // {
    //     int tolerance = 4;
    //     std_msgs::Float64MultiArray maximas;

    //     for (int i = tolerance; i < smooth_histogram.size() - tolerance; i++)
    //     {
    //         int curr = smooth_histogram[i];
    //         int prev = smooth_histogram[i - tolerance];
    //         int next = smooth_histogram[i + tolerance];

    //         if (curr > prev && curr > next)
    //         {
    //             maximas.data.push_back(i);
    //         }
    //     }

    //     std_msgs::Float64MultiArray filtered_max;
    //     int prev = maximas.data[0];
    //     filtered_max.data.push_back(prev);

    //     for (int i = 1; i < maximas.data.size(); i++)
    //     {
    //         int element = maximas.data[i];
    //         if (element - prev >= 40)
    //             filtered_max.data.push_back(element);
    //         prev = element;
    //     }

    //     std_msgs::Float64MultiArray msg;
    //     for (int i = 0; i < smooth_histogram.size(); i++)
    //     {
    //         msg.data.push_back(smooth_histogram[i]);
    //     }

    //     hist_pub.publish(msg);
    //     max_pub.publish(filtered_max);
    // }

    // void checkIPM(const cv::Mat img)
    // {
    //     cv::Mat img_resized;
    //     cv::resize(img, img_resized, cv::Size(w, h), 0, 0, cv::INTER_AREA);

    //     // // cv::Mat black_img = cv::Mat::zeros(img_resized.size(), img_resized.type());
    //     cv::circle(img_resized, tl, 5, cv::Scalar(0, 0, 255), -1);
    //     cv::circle(img_resized, bl, 5, cv::Scalar(0, 0, 255), -1);
    //     cv::circle(img_resized, tr, 5, cv::Scalar(0, 0, 255), -1);
    //     cv::circle(img_resized, br, 5, cv::Scalar(0, 0, 255), -1);
    //     // cv::circle(img_resized, cv::Point(320, 480), 3, cv::Scalar(0, 255, 0), -1);
    //     cv::imwrite("/home/sedrica/original.jpg", img_resized);

    //     cv::Mat ipm_black;

    //     cv::warpPerspective(img_resized, ipm_black, M, cv::Size(w, h), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0));
    //     cv::imwrite("/home/sedrica/ipm_img.jpg", ipm_black);

    //     // cv::circle(img_resized, tl, 5, cv::Scalar(0, 0, 255), -1);
    //     // cv::circle(img_resized, bl, 5, cv::Scalar(0, 0, 255), -1);
    //     // cv::circle(img_resized, tr, 5, cv::Scalar(0, 0, 255), -1);
    //     // cv::circle(img_resized, br, 5, cv::Scalar(0, 0, 255), -1);

    //     sensor_msgs::ImagePtr ipm_img = cv_bridge::CvImage(std_msgs::Header(), "bgr8", ipm_black).toImageMsg();
    //     ipm_pub.publish(ipm_img);
    // }

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
    ros::init(argc, argv, "Sliding_Window");
    ROS_INFO("SLIDING WINDOW HAS BEEN LAUNCHED");
    SlidingWindow obj;
    ros::MultiThreadedSpinner spinner(5); // 5 threads
    spinner.spin();
    return 0;
}
