// #include <ros/ros.h>
// #include <cv_bridge/cv_bridge.h>
// #include <sensor_msgs/image_encodings.h>
// #include <std_msgs/Float64MultiArray.h>
// #include <nav_msgs/OccupancyGrid.h>
// #include <sensor_msgs/PointCloud2.h>
// #include <pcl_conversions/pcl_conversions.h>
// #include <pcl/point_cloud.h>
// #include <pcl/point_types.h>
// #include <pcl/filters/conditional_removal.h>
// #include <pcl_ros/transforms.h>
// #include <pcl/conversions.h>
// #include <pcl/PCLPointCloud2.h>
// #include "pcl_ros/point_cloud.h"
// #include <pcl/common/common.h>
// #include <Eigen/Dense>
// typedef pcl::PointCloud<pcl::PointXYZ>::Ptr pointCloudPtr;

// class pgmWriter
// {
// public:
//     Eigen::Matrix3d K; // Intrinsic matrix
//     double resolution = 0.01;
//     // double margin_m = 0.05;
//     int8_t freeSpace = 0;
//     int8_t collisionSpace = 100;
//     double floorY = 0.215; 
//     double collisionYmax = -1;
//     // double searchRadius = 0.06;
//     int mapWidth = 300;
//     int mapHeight = 300;
//     int camx=0.1;
//     int camy=0.1;
//     double fx = 303.7754; // Focal length along the x-axis
//     double fy = 302.5833; // Focal length along the y-axis
//     double cx = 305.9769; // Principal point x
//     double cy = 288.5363; // Principal point y

//     ros::Publisher pub;
//     std::vector<std::vector<int8_t>> gridMapValue;

//     pgmWriter()
//     {
//         K << fx, 0, cx,
//              0, fy, cy,
//              0, 0, 1;

//         // camx = (mapWidth / 2) * resolution;
//         // camy = (mapHeight / 2) * resolution;

//         for (int i = 0; i < mapHeight; i++)
//         {
//             std::vector<int8_t> temp(mapWidth, freeSpace);
//             gridMapValue.push_back(temp);
//         }
//     }

//     void setPcdMap(const sensor_msgs::PointCloud2ConstPtr& input);
//     void occ_callback(const std_msgs::Float64MultiArrayConstPtr& lanes_array_msg);
//     void image_callback(const sensor_msgs::ImageConstPtr& depth_image_msg);
//     void pcdToOccupancyGrid(pointCloudPtr pcdPoint);
//     Eigen::Vector3d convertPixelPointsToWorldCoordinates(double pixel_x, double pixel_y, const sensor_msgs::ImageConstPtr& depth_image_msg);
//     double getDepthValue(const cv::Mat& depth_image, double pixel_x, double pixel_y);
//     Eigen::Vector3d pixelToCamera(const Eigen::Vector2d& pixel, double depth);
//     void addPointToOccupancyGrid(const Eigen::Vector3d& cameraCoords);
//     void processLanesArray(const std_msgs::Float64MultiArrayConstPtr& lanes_array_msg, const sensor_msgs::ImageConstPtr& depth_image_msg);
//     void publishOccupancyGrid();

// private:
//     pointCloudPtr pcdMap;
//     std_msgs::Float64MultiArray occupancyGridData; // Variable to store occupancy grid data
//     sensor_msgs::ImageConstPtr depthImageData;
// };

// void pgmWriter::occ_callback(const std_msgs::Float64MultiArrayConstPtr& lanes_array_msg)
// {
//     occupancyGridData=*lanes_array_msg;
//     if (depthImageData) {
//         // ROS_INFO("Entered occ_c");
//         processLanesArray(lanes_array_msg, depthImageData);

//     };
// }

// void pgmWriter::image_callback(const sensor_msgs::ImageConstPtr& depth_array_msg)
// {
//     depthImageData=depth_array_msg;
//     // print(depthImageData.shape)
//     // ROS_INFO("Entered image_c");
// }


// void pgmWriter::setPcdMap(const sensor_msgs::PointCloud2ConstPtr& input)
// {
//     pcdMap.reset(new pcl::PointCloud<pcl::PointXYZ>());
//     pcl::fromROSMsg(*input, *pcdMap);
//     pcdToOccupancyGrid(pcdMap);
// }

// void pgmWriter::pcdToOccupancyGrid(pointCloudPtr pcdPoint)
// {
//     for (const auto &pt : pcdPoint->points)
//     {
//         if (std::isnan(pt.x) || std::isnan(pt.y) || std::isnan(pt.z))
//         {
//             continue;
//         }

//         if (pt.z >= mapHeight * resolution / 2 - 0.01 || abs(pt.x) >= mapWidth * resolution / 2 - 0.01)
//         {
//             continue;
//         }
//         float k=abs(pt.x)-0.1;
//         if(abs(k)>=0.1){
//             continue;
//         }
//         int col = (pt.x * (1 / resolution)) + mapWidth / 2;
//         int row = (pt.z * (1 / resolution)) + mapHeight / 2;

//         if (pt.y < floorY && pt.y > collisionYmax)
//         {
//             gridMapValue[row][col] = collisionSpace;
//         }
//     }
//     // publishOccupancyGrid();
// }

// Eigen::Vector3d pgmWriter::convertPixelPointsToWorldCoordinates(double pixel_x, double pixel_y, const sensor_msgs::ImageConstPtr& depth_image_msg)
// {
//     cv_bridge::CvImagePtr cv_ptr;
//     try
//     {
//         cv_ptr = cv_bridge::toCvCopy(depth_image_msg, sensor_msgs::image_encodings::TYPE_32FC1);
//     }
//     catch (cv_bridge::Exception& e)
//     {
//         ROS_ERROR("cv_bridge exception: %s", e.what());
//         return Eigen::Vector3d::Zero();
//     }

//     double depth = getDepthValue(cv_ptr->image, pixel_x, pixel_y);
//     depth *=0.001;

//     Eigen::Vector3d cameraCoords = pixelToCamera(Eigen::Vector2d(pixel_x, pixel_y), depth);

//     return cameraCoords;
// }

// double pgmWriter::getDepthValue(const cv::Mat& depth_image, double pixel_x, double pixel_y)
// {
//     if (pixel_x >= 0 && pixel_x < depth_image.cols && pixel_y >= 0 && pixel_y < depth_image.rows)
//     {
//         return depth_image.at<float>(pixel_y, pixel_x);
//     }
//     return 0.0;
// }

// Eigen::Vector3d pgmWriter::pixelToCamera(const Eigen::Vector2d& pixel, double depth)
// {
//     Eigen::Vector3d normalized(pixel.x(), pixel.y(), 1.0);
//     // normalized = normalized / normalized.y();
//     Eigen::Vector3d cameraCoords = K.inverse() * normalized;
//     cameraCoords *= depth;
//     return cameraCoords;
// }

// void pgmWriter::addPointToOccupancyGrid(const Eigen::Vector3d& cameraCoords)
// {
//     // int row = ((cameraCoords.x() - ((mapWidth / 2) * resolution - camx)) * (1 / resolution)) + mapWidth / 2;
//     // int col = ((cameraCoords.z() - ((mapHeight / 2) * resolution - camy)) * (1 / resolution)) + mapHeight / 2;
//     int col = (cameraCoords.x()* (1 / resolution)) + mapWidth / 2;
//     int row = (cameraCoords.z()* (1 / resolution)) + mapHeight / 2;
//     // ROS_INFO("x: %d", row);
//     // ROS_INFO("z: %d", col);
//     // double translatedX = cameraCoords.x() - ((mapWidth / 2) * resolution - camx);
//     // double translatedZ = cameraCoords.z() - ((mapHeight / 2) * resolution - camy);
//     // double tempX = translatedX - camx;
//     // double tempY = translatedZ - camy;
//     // double rotatedX = -tempY;
//     // double rotatedY = tempX;
//     // int col = (rotatedX + camx) * (1 / resolution) + mapWidth / 2;
//     // int row = (rotatedY + camy) * (1 / resolution) + mapHeight / 2;
//     if (row >= 0 && row < mapHeight && col >= 0 && col < mapWidth)
//     {
//         gridMapValue[row][col] = collisionSpace;

//     }
//     if(row>300 || col>300){
//         ROS_INFO("x: %d", row);
//         ROS_INFO("z: %d", col);
//     }
//     for(int i=50;i<150;i++){
//         gridMapValue[100][i] = collisionSpace;
//     }
// }

// void pgmWriter::processLanesArray(const std_msgs::Float64MultiArrayConstPtr& lanes_array_msg, const sensor_msgs::ImageConstPtr& depth_image_msg)
// {
//     std::vector<double> data = lanes_array_msg->data;
//     if (data.size() % 2 != 0) {
//         ROS_ERROR("Invalid number of elements in the Float64MultiArray");
//         return;
//     }
//     // ROS_INFO("Entered proc");
//     std::vector<Eigen::Vector3d> lanePoints;
//     for (size_t i = 0; i < data.size(); i += 2) {
//         double x = data[i];
//         double y = data[i + 1];

//         Eigen::Vector3d worldPoint = convertPixelPointsToWorldCoordinates(x, y, depth_image_msg);
//         // ROS_INFO("Pushed");
//         lanePoints.push_back(worldPoint);
//     }

//     for (const auto& lanePoint : lanePoints) {
//         // ROS_INFO("adding");
//         addPointToOccupancyGrid(lanePoint);
//     }

//     publishOccupancyGrid();
// }
// void pgmWriter::publishOccupancyGrid()
// {
//     nav_msgs::OccupancyGrid grid;
//     grid.header.stamp = ros::Time::now();
//     grid.header.frame_id = "camera_color_optical_frame";
//     grid.info.resolution = resolution;
//     grid.info.width = mapWidth;
//     grid.info.height = mapHeight;
//     grid.info.origin.position.x = 0;
//     grid.info.origin.position.y = 0;
//     grid.info.origin.position.z = 0;
//     grid.info.origin.orientation.x = 0;
//     grid.info.origin.orientation.y = 0;
//     grid.info.origin.orientation.z = 0;
//     grid.info.origin.orientation.w = 1;

//     for (int row = mapHeight - 1; row >= 0; row--)
//     {
//         for (int col = 0; col < mapWidth; col++)
//         {
//             grid.data.push_back(gridMapValue[row][col]);
//         }
//     }

//     pub.publish(grid);
// }

// int main(int argc, char **argv)
// {
//     ros::init(argc, argv, "pgm_writer");
//     ros::NodeHandle nh;
//     pgmWriter writer;
//     writer.pub = nh.advertise<nav_msgs::OccupancyGrid>("/occupancy_grid", 1);
//     ros::Subscriber sub2 = nh.subscribe<sensor_msgs::Image>("/camera/aligned_depth_to_color/image_raw", 1, &pgmWriter::image_callback, &writer);
//     ros::Subscriber sub = nh.subscribe<sensor_msgs::PointCloud2>("/camera/depth_registered/points", 1, &pgmWriter::setPcdMap, &writer);
//     ros::Subscriber sub1 = nh.subscribe<std_msgs::Float64MultiArray>("/inverse_lanes_topic", 1, &pgmWriter::occ_callback, &writer);
//     ros::spin();
//     return 0;
// }


#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <std_msgs/Float64MultiArray.h>
#include <nav_msgs/OccupancyGrid.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/conditional_removal.h>
#include <pcl_ros/transforms.h>
#include <pcl/conversions.h>
#include <pcl/PCLPointCloud2.h>
#include "pcl_ros/point_cloud.h"
#include <pcl/common/common.h>
#include <Eigen/Dense>
typedef pcl::PointCloud<pcl::PointXYZ>::Ptr pointCloudPtr;

class pgmWriter
{
public:
    Eigen::Matrix3d K; // Intrinsic matrix
    double resolution = 0.01;
    // double margin_m = 0.05;
    int8_t freeSpace = 0;
    int8_t collisionSpace = 100;
    double floorY = 0.185; 
    double collisionYmax = 0.115;
    // double searchRadius = 0.06;
    int mapWidth = 300;
    int mapHeight = 300;
    int camx=10;
    int camy=10;
    // double fx = 303.7754; // Focal length along the x-axis
    // double fy = 302.5833; // Focal length along the y-axis
    // double cx = 305.9769; // Principal point x
    // double cy = 288.5363; // Principal point y

    double fx=646.35693359375;
    double fy=645.5247192382812;
    double cx= 645.4840698242188;
    double cy=367.06915283203125;
    double D[5]={-0.05302947387099266, 0.06136186420917511, 0.00021476151596289128, 0.001250683912076056, -0.01992097683250904};

    ros::Publisher pub;
    std::vector<std::vector<int8_t>> gridMapValue;

    pgmWriter()
    {
        K << fx, 0, cx,
             0, fy, cy,
             0, 0, 1;

        // camx = (mapWidth / 2) * resolution;
        // camy = (mapHeight / 2) * resolution;

        for (int i = 0; i < mapHeight; i++)
        {
            std::vector<int8_t> temp(mapWidth, freeSpace);
            gridMapValue.push_back(temp);
        }
    }

    void setPcdMap(const sensor_msgs::PointCloud2ConstPtr& input);
    void occ_callback(const std_msgs::Float64MultiArrayConstPtr& lanes_array_msg);
    void image_callback(const sensor_msgs::ImageConstPtr& depth_image_msg);
    void pcdToOccupancyGrid(pointCloudPtr pcdPoint);
    Eigen::Vector3d convertPixelPointsToWorldCoordinates(double pixel_x, double pixel_y, const sensor_msgs::ImageConstPtr& depth_image_msg);
    double getDepthValue(const cv::Mat& depth_image, double pixel_x, double pixel_y);
    Eigen::Vector3d pixelToCamera(const Eigen::Vector2d& pixel, double depth);
    void addPointToOccupancyGrid(const Eigen::Vector3d& cameraCoords);
    void processLanesArray(const std_msgs::Float64MultiArrayConstPtr& lanes_array_msg, const sensor_msgs::ImageConstPtr& depth_image_msg);
    void publishOccupancyGrid();

private:
    pointCloudPtr pcdMap;
    std_msgs::Float64MultiArray occupancyGridData; // Variable to store occupancy grid data
    sensor_msgs::ImageConstPtr depthImageData;
};

void pgmWriter::occ_callback(const std_msgs::Float64MultiArrayConstPtr& lanes_array_msg)
{
    occupancyGridData=*lanes_array_msg;
    if (depthImageData) {
        // ROS_INFO("Entered occ_c");
        processLanesArray(lanes_array_msg, depthImageData);

    };
}

void pgmWriter::image_callback(const sensor_msgs::ImageConstPtr& depth_array_msg)
{
    depthImageData=depth_array_msg;
    // print(depthImageData.shape)
    // ROS_INFO("Entered image_c");
}


void pgmWriter::setPcdMap(const sensor_msgs::PointCloud2ConstPtr& input)
{
    pcdMap.reset(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::fromROSMsg(*input, *pcdMap);
    pcdToOccupancyGrid(pcdMap);
}

void pgmWriter::pcdToOccupancyGrid(pointCloudPtr pcdPoint)
{
    for (const auto &pt : pcdPoint->points)
    {
        if (std::isnan(pt.x) || std::isnan(pt.y) || std::isnan(pt.z))
        {
            continue;
        }

        if (pt.z >= mapHeight * resolution / 2 - 0.01 || abs(pt.x) >= mapWidth * resolution / 2 - 0.01)
        {
            continue;
        }
        float k=abs(pt.x)-0.1;
        if(abs(k)>=0.1){
            continue;
        }
        int col = (pt.x * (1 / resolution)) + mapWidth / 2;
        int row = (pt.z * (1 / resolution)) + mapHeight / 2;

        if (pt.y < floorY && pt.y > collisionYmax)
        {
            gridMapValue[row][col] = collisionSpace;
        }
    }
    // publishOccupancyGrid();
}

Eigen::Vector3d pgmWriter::convertPixelPointsToWorldCoordinates(double pixel_x, double pixel_y, const sensor_msgs::ImageConstPtr& depth_image_msg)
{
    cv_bridge::CvImagePtr cv_ptr;
    try
    {
        cv_ptr = cv_bridge::toCvCopy(depth_image_msg, sensor_msgs::image_encodings::TYPE_32FC1);
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return Eigen::Vector3d::Zero();
    }

    double depth = getDepthValue(cv_ptr->image, pixel_x, pixel_y);
    depth *=0.001;

    Eigen::Vector3d cameraCoords = pixelToCamera(Eigen::Vector2d(pixel_x, pixel_y), depth);

    return cameraCoords;
}

double pgmWriter::getDepthValue(const cv::Mat& depth_image, double pixel_x, double pixel_y)
{
    // if (pixel_x >= 0 && pixel_x < depth_image.cols && pixel_y >= 0 && pixel_y < depth_image.rows)
    // {
    double depth= depth_image.at<float>(pixel_y, pixel_x);
    // if(depth<300 && depth!=0) {
    //     ROS_INFO("depth: %f",depth);
    // }
    
    return depth;

    // }
    // return 0.0;
}

Eigen::Vector3d pgmWriter::pixelToCamera(const Eigen::Vector2d& pixel, double depth)
{
    // Eigen::Vector3d normalized(pixel.x(), pixel.y(), 1.0);
    // Eigen::Vector3d cameraCoords = K.inverse() * normalized;
    // // cameraCoords.x()=(pixel.x()-cx)/fx;
    // cameraCoords *= depth;
    // return cameraCoords;
    Eigen::Vector3d cameraCoords ;
    float x = (pixel.x() - cx) /fx;
    float y = (pixel.y() - cy) / fy;
    float r2  = x*x + y*y;
    float f = 1 + D[0]*r2 + D[1]*r2*r2 + D[4]*r2*r2*r2;
    float ux = x*f + 2*D[2]*x*y + D[3]*(r2 + 2*x*x);
    float uy = y*f + 2*D[3]*x*y + D[2]*(r2 + 2*y*y);
    x = ux;
    y = uy;
    cameraCoords.x() = depth * x;
    cameraCoords.y() = depth * y;
    cameraCoords.z() = depth;
    return cameraCoords;
}

void pgmWriter::addPointToOccupancyGrid(const Eigen::Vector3d& cameraCoords)
{
    // int row = ((-1*cameraCoords.x() - ((mapWidth / 2) * resolution - camx)) * (1 / resolution)) + mapWidth / 2;
    // int col = ((cameraCoords.z() - ((mapHeight / 2) * resolution - camy)) * (1 / resolution)) + mapHeight / 2;
    int col = ((-1*cameraCoords.x())* (1 / resolution)) + mapWidth / 2;
    int row = (cameraCoords.z()* (1 / resolution)) + mapHeight / 2;
    // int col1 = ((1*cameraCoords.x()/3)* (1 / resolution)) + mapWidth / 2;

    ROS_INFO("y: %f", cameraCoords.y());
    // // ROS_INFO("z: %d", col);
    // int colt=col1-camx;
    // int rowt=row-camy;

    // int colr=rowt +camx;
    // int rowr=colt+camy;

    // if (rowr >= 0 && rowr < mapHeight && colr >= 0 && colr < mapWidth)
    // {
    //     gridMapValue[rowr][colr] = collisionSpace;


    // }
    if (row >= 0 && row < mapHeight && col >= 0 && col < mapWidth)
    {
        gridMapValue[row][col] = collisionSpace;


    }
    if(row>300 || col>300){
        ROS_INFO("x: %d", row);
        ROS_INFO("z: %d", col);
    }
    for(int i=50;i<150;i++){
        gridMapValue[100][i] = collisionSpace;
    }
}

void pgmWriter::processLanesArray(const std_msgs::Float64MultiArrayConstPtr& lanes_array_msg, const sensor_msgs::ImageConstPtr& depth_image_msg)
{
    std::vector<double> data = lanes_array_msg->data;
    if (data.size() % 2 != 0) {
        ROS_ERROR("Invalid number of elements in the Float64MultiArray");
        return;
    }
    // ROS_INFO("Entered proc");
    std::vector<Eigen::Vector3d> lanePoints;
    for (size_t i = 0; i < data.size(); i += 2) {
        double x = data[i];
        double y = data[i + 1];

        Eigen::Vector3d worldPoint = convertPixelPointsToWorldCoordinates(x, y, depth_image_msg);
        // ROS_INFO("Pushed");
        lanePoints.push_back(worldPoint);
    }

    for (const auto& lanePoint : lanePoints) {
        // ROS_INFO("adding");
        addPointToOccupancyGrid(lanePoint);
    }

    publishOccupancyGrid();
}
void pgmWriter::publishOccupancyGrid()
{
    nav_msgs::OccupancyGrid grid;
    grid.header.stamp = ros::Time::now();
    grid.header.frame_id = "camera_color_optical_frame";  //originally camera_color_optical_frame
    grid.info.resolution = resolution;
    grid.info.width = mapWidth;
    grid.info.height = mapHeight;
    grid.info.origin.position.x = 0;
    grid.info.origin.position.y = 0;
    grid.info.origin.position.z = 0;
    grid.info.origin.orientation.x = 0;
    grid.info.origin.orientation.y = 0;
    grid.info.origin.orientation.z = 0;
    grid.info.origin.orientation.w = 1;

    for (int row = mapHeight - 1; row >= 0; row--)
    {
        for (int col = 0; col < mapWidth; col++)
        {
            grid.data.push_back(gridMapValue[row][col]);
        }
    }

    pub.publish(grid);
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "pgm_writer");
    ros::NodeHandle nh;
    pgmWriter writer;
    writer.pub = nh.advertise<nav_msgs::OccupancyGrid>("/occupancy_grid", 1);
    ros::Subscriber sub2 = nh.subscribe<sensor_msgs::Image>("/camera/aligned_depth_to_color/image_raw", 1, &pgmWriter::image_callback, &writer);
    ros::Subscriber sub = nh.subscribe<sensor_msgs::PointCloud2>("/camera/depth_registered/points", 1, &pgmWriter::setPcdMap, &writer);
    ros::Subscriber sub1 = nh.subscribe<std_msgs::Float64MultiArray>("/inverse_lanes_topic", 1, &pgmWriter::occ_callback, &writer);
    ros::spin();
    return 0;
}