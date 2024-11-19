#include <ros/ros.h>
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/Image.h>
#include <std_msgs/Bool.h>
#include <std_msgs/Float64.h>
#include <std_msgs/Float64MultiArray.h>
#include <std_msgs/Int64MultiArray.h>
#include <cv_only/LaneCoordinates.h>
// #include <cv_only/LaneCoordinates.h>
#include <cmath>
#include <chrono>
#include <eigen3/Eigen/Dense>
#include <string.h>

class SlidingWindow
{
private:
  ros::NodeHandle nh;
  ros::Publisher lane_img_pub;
  ros::Publisher lane_coords_pub;
  ros::Publisher erosion_img_pub;
  ros::Publisher window_pub;
  ros::Publisher occupancy_points_pub;
  ros::Publisher stopline_flag_pub;

  ros::Subscriber camera_sub;
  ros::Subscriber middle_lane_sub;

  std_msgs::Float64 speed;
  std_msgs::Bool middle_bool;

  cv_bridge::CvImagePtr cv_ptr;

  cv::Point tl;
  cv::Point bl;
  cv::Point tr;
  cv::Point br;
  cv::Point2f source_pts[4];

  cv::Mat box_img;

  int binary_threshold = 185;
  int blacken_pixels = 20;

  int false_ctr = 0, true_ctr = 0;
  int false_threshold = 8;

public:
  SlidingWindow() : nh("~"), tl(100, 350), bl(2, 433), tr(350, 350), br(417, 433)
  {
    middle_bool.data = false;

    camera_sub = nh.subscribe<sensor_msgs::Image>("/camera/color/image_raw", 100, &SlidingWindow::camera_cbk, this);
    middle_lane_sub = nh.subscribe<std_msgs::Bool>("/middle_lane", 2, &SlidingWindow::middle_cbk, this);

    lane_img_pub = nh.advertise<sensor_msgs::Image>("/lane_image", 10);
    lane_coords_pub = nh.advertise<cv_only::LaneCoordinates>("/lane_coordinates", 10);
    erosion_img_pub = nh.advertise<sensor_msgs::Image>("/erosion_lane_image", 10);
    window_pub = nh.advertise<sensor_msgs::Image>("/window_topic", 10);
    occupancy_points_pub = nh.advertise<std_msgs::Float64MultiArray>("/inverse_lanes_topic", 10);
    stopline_flag_pub = nh.advertise<std_msgs::Bool>("/stopline_flag", 10);

    source_pts[0] = bl;
    source_pts[1] = br;
    source_pts[2] = tr;
    source_pts[3] = tl;
  }

private:
  cv::Mat process_img(const cv::Mat image)
  {
    cv::Mat image_resized;
    cv::resize(image, image_resized, cv::Size(640, 480), 0, 0, cv::INTER_AREA);

    int h = image_resized.rows;
    int w = image_resized.cols;
    int ymax = 480;      // Check if ymax is needed, can only use h
    int phy_length = 60; // Urban
    int pixels_per_cm = ymax / phy_length;

    // 40 AND 20 IS FOR URBAN, 38 AND 14 FOR RACING
    cv::Point2f dest_pts[] = {
        cv::Point2f(w / 2 - 40 * pixels_per_cm, ymax),
        cv::Point2f(w / 2 + 20 * pixels_per_cm, ymax),
        cv::Point2f(w / 2 + 20 * pixels_per_cm, ymax - phy_length * pixels_per_cm),
        cv::Point2f(w / 2 - 40 * pixels_per_cm, ymax - phy_length * pixels_per_cm)};
    cv::Mat M = cv::getPerspectiveTransform(source_pts, dest_pts);
    cv::Mat image_warped;
    cv::warpPerspective(image_resized, image_warped, M, cv::Size(w, h), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0));

    cv::Mat image_gray;
    cv::cvtColor(image_warped, image_gray, cv::COLOR_BGR2GRAY);

    cv::Mat edges;
    cv::threshold(image_gray, edges, binary_threshold, 255, cv::THRESH_BINARY);

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::Mat erosion;
    cv::erode(edges, erosion, kernel, cv::Point(-1, -1), 6); // -1, -1 Selects the center or something

    cv::Mat erosion_half = erosion(cv::Rect(0, erosion.rows / 2, erosion.cols, erosion.rows / 2));
    cv::Mat erosion_half_resized;
    cv::resize(erosion_half, erosion_half_resized, cv::Size(640, 480), 0, 0, cv::INTER_LINEAR);

    erosion_half_resized.colRange(0, blacken_pixels).setTo(cv::Scalar(0));
    erosion_half_resized.colRange(erosion.cols - blacken_pixels, erosion.cols).setTo(cv::Scalar(0));

    return erosion_half_resized;
  }

  double get_time()
  {
    using namespace std::chrono;

    auto now = system_clock::now();
    auto since_epoch = now.time_since_epoch();
    auto millisec = duration_cast<milliseconds>(since_epoch);
    return (double)(millisec.count() / 1000.0);
  }

  void middle_cbk(const std_msgs::BoolConstPtr &msg)
  {
    middle_bool.data = msg->data;
  }

  Eigen::VectorXd gaussian_filter(const Eigen::VectorXd histogram, double sigma = 30)
  {
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

    // Normalize kernel
    kernel /= sum;

    // Apply Gaussian filter
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

    return smooth_histogram;
  }

  void find_lane_pxl(const cv::Mat img, std::vector<int> &left_lane_ind, std::vector<int> &right_lane_ind)
  {
    double start = get_time();
    double img_ratio = 1.0 / 3.0;
    // nh.getParam("IMG_RATIO", img_ratio);
    cv::Mat ROI = img(cv::Range(img.rows * (1 - img_ratio), img.rows), cv::Range(0, img.cols));
    cv::Mat ROI_double;
    ROI.convertTo(ROI_double, CV_64F);
    ROI_double /= 255.0;

    Eigen::VectorXd histogram = Eigen::VectorXd::Zero(ROI_double.cols);
    for (int col = 0; col < ROI_double.cols; ++col)
      for (int row = 0; row < ROI_double.rows; ++row)
        histogram(col) += ROI_double.at<double>(row, col);
    // std::cout << "HIST TIME: " << get_time() - start << std::endl;
    Eigen::VectorXd smooth_histogram = gaussian_filter(histogram);

    int rows = img.rows;
    int cols = img.cols;
    int midpoint = smooth_histogram.size() / 2;
    int leftx_base, rightx_base;
    smooth_histogram.segment(0, midpoint).maxCoeff(&leftx_base);
    smooth_histogram.segment(midpoint, smooth_histogram.size() - midpoint).maxCoeff(&rightx_base);
    rightx_base += midpoint;

    int nwin = 10, minpix = 300, margin = 60; // Add params
    int win_ht = rows / nwin;
    int leftx_curr = leftx_base, rightx_curr = rightx_base;
    int y_low, y_high;
    int x_low, x_high;
    bool hit_left = true, hit_right = true;

    std::vector<int> win_pxl;

    for (int win = 0; win < nwin; win++) // Left Window
    {
      int ctr = 0;
      double avg = 0.0;
      win_pxl.clear();

      y_low = rows - (win + 1) * win_ht; // Numerically low
      y_high = rows - win * win_ht;
      x_high = leftx_curr + margin;
      x_low = leftx_curr - margin;

      for (int y = y_high; y >= y_low; y--)
      {
        std::vector<int> x_array;
        for (int x = x_low; x <= x_high; x++)
          if (x >= 0 && x <= cols && y >= 0 && y <= rows)
            if (img.at<uchar>(y, x) == 255)
            {
              x_array.push_back(x);
              avg = (avg * ctr + x) / ++ctr;
            }

        if (x_array.size())
        {
          int x_avg = (x_array[0] + x_array[x_array.size() - 1]) / 2;
          win_pxl.push_back(x_avg);
          win_pxl.push_back(y);
        }
      }

      if (ctr >= minpix)
      {
        left_lane_ind.insert(left_lane_ind.end(), win_pxl.begin(), win_pxl.end());
        hit_left = true;
        leftx_curr = (int)avg;
      }
      else if (hit_left)
      {
        break;
      }
    }

    for (int win = 0; win < nwin; win++) // Right Window
    {
      int ctr = 0;
      double avg = 0.0;
      win_pxl.clear();

      y_low = rows - (win + 1) * win_ht;
      y_high = rows - win * win_ht;
      x_high = rightx_curr + margin;
      x_low = rightx_curr - margin;

      for (int y = y_high; y >= y_low; y--)
      {
        std::vector<int> x_array;
        for (int x = x_low; x <= x_high; x++)
          if (x >= 0 && x <= cols && y >= 0 && y <= rows)
            if (img.at<uchar>(y, x) == 255)
            {
              x_array.push_back(x);
              avg = (avg * ctr + x) / ++ctr;
            }

        if (x_array.size())
        {
          int x_avg = (x_array[0] + x_array[x_array.size() - 1]) / 2;
          win_pxl.push_back(x_avg);
          win_pxl.push_back(y);
        }
      }

      if (ctr >= minpix)
      {
        right_lane_ind.insert(right_lane_ind.end(), win_pxl.begin(), win_pxl.end());
        hit_right = true;
        rightx_curr = (int)avg;
      }
      else if (hit_right)
        break;
    }

    // ROS_INFO("NO ERRORS");
  }

  cv::Mat get_color_img(cv::Mat out_img, const std::vector<int> left_lane_ind, const std::vector<int> right_lane_ind)
  {
    for (int i = 0; i < left_lane_ind.size(); i += 2)
    {
      int x = left_lane_ind[i];
      int y = left_lane_ind[i + 1];
      out_img.at<cv::Vec3b>(y, x) = cv::Vec3b(255, 0, 0);
    }

    for (int i = 0; i < right_lane_ind.size(); i += 2)
    {
      int x = right_lane_ind[i];
      int y = right_lane_ind[i + 1];
      out_img.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 255);
    }

    return out_img;
  }

  void extrapolate_one_lane(const cv::Mat &img, const cv::Mat &channel_img, std::vector<int> lane_ind)
  {
    // ROS_INFO("ONE LANE");
    int size = lane_ind.size();

    if (size >= 4)
    {
      int x1 = lane_ind[0], x2 = lane_ind[lane_ind.size() - 2];
      cv::Mat empty = cv::Mat::zeros(channel_img.size(), channel_img.type());

      if (x2 - x1 >= 0) // Blue lane
      {
        std::vector<cv::Mat> channels = {channel_img, empty, empty};
        cv::merge(channels, img);
        publish_lane_coordinates(lane_ind, std::vector<int>(), std::vector<int>());
        // publish_lane_coordinates()
      }
      else
      {
        std::vector<cv::Mat> channels = {empty, empty, channel_img};
        cv::merge(channels, img);
        publish_lane_coordinates(std::vector<int>(), std::vector<int>(), lane_ind);
      }
    }
  }

  void extrapolate_two_lanes(cv::Mat &img, const std::vector<cv::Mat> channels, std::vector<int> &left_lane_ind, std::vector<int> &right_lane_ind)
  {
    int right_size = right_lane_ind.size(), left_size = left_lane_ind.size();
    int threshold_dist = 350;
    int min_dist = 30;
    int sum = 0, ctr = 0;
    std::vector<int> middle_lane;

    if (left_size - right_size >= 0)
    {
      for (int i = 0; i < right_size; i += 10)
      {
        sum += abs(left_lane_ind[i] - right_lane_ind[i]);
        ctr++;
        middle_lane.push_back((left_lane_ind[i] + right_lane_ind[i]) / 2);
        middle_lane.push_back((left_lane_ind[i + 1] + right_lane_ind[i + 1]) / 2);
      }
    }
    else
    {
      for (int i = 0; i < left_size; i += 10)
      {
        sum += abs(left_lane_ind[i] - right_lane_ind[i]);
        ctr++;
        middle_lane.push_back((left_lane_ind[i] + right_lane_ind[i]) / 2);
        middle_lane.push_back((left_lane_ind[i + 1] + right_lane_ind[i + 1]) / 2);
      }
    }

    int dist = sum / ctr;
    // ROS_INFO("DIST: %i", dist);

    if (dist > threshold_dist)
    {
      for (int i = 0; i < middle_lane.size(); i += 2)
      {
        int x = middle_lane[i];
        int y = middle_lane[i + 1];
        img.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 255, 0);
      }
      publish_lane_coordinates(left_lane_ind, middle_lane, right_lane_ind);
    }
    else
    {
      if (left_size <= right_size)
      {
        for (int i = 0; i < left_size; i += 2)
        {
          int x = left_lane_ind[i];
          int y = left_lane_ind[i + 1];
          img.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 255, 0);
        }
        publish_lane_coordinates(std::vector<int>(), middle_lane, right_lane_ind);
      }
      else
      {
        for (int i = 0; i < right_size; i += 2)
        {
          int x = right_lane_ind[i];
          int y = right_lane_ind[i + 1];
          img.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 255, 0);
        }
        publish_lane_coordinates(left_lane_ind, middle_lane, std::vector<int>());
      }
    }
  }

  void publish_lane_coordinates(const std::vector<int> left, const std::vector<int> middle, const std::vector<int> right)
  {
    cv_only::LaneCoordinates points;
    std_msgs::Float64MultiArray occupancy_pts;

    for (int i = 0; i < left.size(); i += 2)
    {
      points.lx.push_back(left[i]);
      points.ly.push_back(left[i + 1]);
      occupancy_pts.data.push_back(left[i]);
      occupancy_pts.data.push_back(left[i + 1]);
    }
    if (!middle_bool.data)
    {
      for (int i = 0; i < middle.size(); i += 2)
      {
        points.mx.push_back(middle[i]);
        points.my.push_back(middle[i + 1]);
        occupancy_pts.data.push_back(middle[i]);
        occupancy_pts.data.push_back(middle[i + 1]);
      }
    }

    for (int i = 0; i < right.size(); i += 2)
    {
      points.rx.push_back(right[i]);
      points.ry.push_back(right[i + 1]);
      occupancy_pts.data.push_back(right[i]);
      occupancy_pts.data.push_back(right[i + 1]);
    }
    if (middle_bool.data)
    {
      points.mx.clear();
      points.my.clear();
    }

    lane_coords_pub.publish(points);
    occupancy_points_pub.publish(occupancy_pts);
  }

  void remove_stopline(const cv::Mat &img)
  {
    // ROS_INFO("ENTERED");
    std_msgs::Int64MultiArray stopline;
    int stopline_threshold = 100;
    double start = get_time();
    Eigen::VectorXd histogram = Eigen::VectorXd::Zero(img.rows);
    for (int row = 0; row < img.rows; ++row)
    {
      for (int col = img.cols / 4; col < img.cols * 3.0 / 4; ++col)
      {
        histogram(row) += img.at<int>(row, col);
      }
      if (histogram(row) > stopline_threshold)
      {
        img.row(row).setTo(0);
      }
    }

    Eigen::Index maxIndex;
    double max_value = histogram.maxCoeff(&maxIndex);

    std_msgs::Bool msg;
    // ROS_INFO("%li", maxIndex);
    if (maxIndex > 470)
    {
      true_ctr++;
    }
    else
    {
      msg.data = false;
      stopline_flag_pub.publish(msg);
      if (true_ctr > 1)
      {
        false_ctr++;
      }
    }
    if (false_ctr > false_threshold)
    {
      msg.data = true;
      stopline_flag_pub.publish(msg);
      true_ctr = 0;
      false_ctr = 0;
    }
    // std::cout << " TIME " << get_time() - start << std::endl;
  }

  void camera_cbk(const sensor_msgs::ImageConstPtr &msg)
  {
    double start_time = get_time();

    cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    cv::Mat image = cv_ptr->image;
    cv::Mat erosion = process_img(image);
    sensor_msgs::ImagePtr erosion_msg = cv_bridge::CvImage(std_msgs::Header(), "mono8", erosion).toImageMsg();
    erosion_img_pub.publish(erosion_msg);

    remove_stopline(erosion);

    std::vector<int> left_lane_ind, right_lane_ind;
    cv::cvtColor(erosion, box_img, cv::COLOR_GRAY2BGR);

    find_lane_pxl(erosion, left_lane_ind, right_lane_ind);

    // When one lane lies entirely within the other, remove the inner lane.
    if (left_lane_ind.size() > 4 && right_lane_ind.size() > 4)
    {
      int left_start = left_lane_ind[0];
      int left_end = left_lane_ind[left_lane_ind.size() - 2];
      int right_start = right_lane_ind[0];
      int right_end = right_lane_ind[right_lane_ind.size() - 2];

      if (left_start >= right_start && left_end <= right_end)
        left_lane_ind.clear();
      else if (left_start <= right_start && left_end >= right_end)
        right_lane_ind.clear();
    }

    cv::Mat color_img = cv::Mat::zeros(image.size(), image.type());
    color_img = get_color_img(color_img, left_lane_ind, right_lane_ind);

    std::vector<cv::Mat> channels(3);
    cv::split(color_img, channels);

    if (cv::sum(channels[0]) == cv::Scalar(0))
    {
      extrapolate_one_lane(color_img, channels[2], right_lane_ind);
      // publish_lane_coordinates(std::vector<int>(), std::vector<int>(), right_lane_ind);
    }
    else if (cv::sum(channels[2]) == cv::Scalar(0))
    {
      extrapolate_one_lane(color_img, channels[0], left_lane_ind);
    }
    else
      extrapolate_two_lanes(color_img, channels, left_lane_ind, right_lane_ind);

    // cv::cvtColor(erosion, erosion, cv::COLOR_GRAY2BGR);
    // if (erosion.rows != color_img.rows)
    // {
    //   std::cerr << "Images do not have the same height" << std::endl;
    //   return;
    // }
    // else if (erosion.type() != color_img.type())
    // {
    //   std::cerr << "Images do not have the same type" << std::endl;
    //   return;
    // }
    // cv::Mat concatenatedImage;
    // cv::hconcat(erosion, color_img, concatenatedImage);
    // if (!cv::imwrite("concatenated_image" + std::to_string(get_time()) + ".jpg", concatenatedImage))
    // {
    //   std::cerr << "Could not save concatenated image" << std::endl;
    //   return;
    // }
    // std::cout << "Concatenated image saved successfully" << std::endl;

    sensor_msgs::ImagePtr lane_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", color_img).toImageMsg();
    lane_img_pub.publish(lane_msg);
    // sensor_msgs::ImagePtr box_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", box_img).toImageMsg();
    // window_pub.publish(box_msg);
    double end_time = get_time();
    ROS_INFO("%f", end_time - start_time);
  }
};

int main(int argc, char **argv)
{
  ros::init(argc, argv, "Sliding_Window");
  SlidingWindow obj;
  ros::spin();
  return 0;
}
