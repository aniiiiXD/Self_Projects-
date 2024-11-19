#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <std_msgs/Float64MultiArray.h>
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
#include <nav_msgs/OccupancyGrid.h>
#include <geometry_msgs/Point.h>
#include <vector>
#include <chrono>

typedef pcl::PointCloud<pcl::PointXYZ>::Ptr pointCloudPtr;

class pgmWriter
{
private:
    ros::NodeHandle nh;

    ros::Publisher pub;
    ros::Publisher midpoint_pub;

    ros::Rate rate;

    ros::Subscriber sub;
    ros::Subscriber sub1;

public:
    // double resolution = 0.00125;
    
    // double margin_m = 0.05;
    int8_t freeSpace = 0;
    int8_t collisionSpace = 100;
    // double floorY = 0.185;
    double floorY = 0.155;
    double collisionYmax = 0.115;
    // double searchRadius = 0.06;
    // int mapWidth = 1600;
    // int mapHeight = 1600;
    int mapWidth = 125;
    int mapHeight = 125;
    double resolution = 2.0/mapHeight  ;
    // Parametyers to shift origin from lane image origin (top left corner of image) to bottom middle (real life position of camera in bird's eye view)
    int camx = 415;
    int camy = 480;
    double camera_blind_spot = 23 * 1.773;
    int cm_conversion_constant = 100;
    double lane_image_resolution_x = 0.00140625;
    double lane_image_resolution_y = 0.0010625;
    double camera_to_car_end_distance = 20;

    std::vector<std::vector<int8_t>>
        gridMapValue;

    pgmWriter() : nh("~"), rate(nh.param("/dm_node/loop_rate", 10))
    {
        for (int i = 0; i < mapHeight; i++)
        {
            std::vector<int8_t> temp(mapWidth, freeSpace);
            gridMapValue.push_back(temp);
        }

        pub = nh.advertise<nav_msgs::OccupancyGrid>("/occupancy_grid", 1);
        midpoint_pub = nh.advertise<geometry_msgs::Point>("/midpoint", 10);

        sub = nh.subscribe<sensor_msgs::PointCloud2>("/camera/depth_registered/points", 1, &pgmWriter::setPcdMap, this);
        sub1 = nh.subscribe<std_msgs::Float64MultiArray>("/occupancy_points", 1, &pgmWriter::processLanesArray, this);
    }

    void setPcdMap(const sensor_msgs::PointCloud2ConstPtr &input);
    void pcdToOccupancyGrid(pointCloudPtr pcdPoint);
    void addPointToOccupancyGrid(double x, double y);
    void processLanesArray(const std_msgs::Float64MultiArrayConstPtr &lanes_array_msg);
    void publishOccupancyGrid();
    void publishObjectMidpoint(int midpoint_x, int midpoint_y);

private:
    pointCloudPtr pcdMap;
    std_msgs::Float64MultiArray occupancyGridData; // Variable to store occupancy grid data
    sensor_msgs::ImageConstPtr depthImageData;
};

void pgmWriter::setPcdMap(const sensor_msgs::PointCloud2ConstPtr &input)
{
    pcdMap.reset(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::fromROSMsg(*input, *pcdMap);
    pcdToOccupancyGrid(pcdMap);
}

void pgmWriter::pcdToOccupancyGrid(pointCloudPtr pcdPoint)
{
    // ROS_INFO("pcd callback");
    int sum_x = 0, sum_y = 0, count = 0;

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
        float k = abs(pt.x) - 0.1;
        if (abs(k) >= 0.1)
        {
            continue;
        }
        int col = (pt.x * (1 / resolution)) + mapWidth / 2;
        int row = (pt.z * (1 / resolution)) + mapHeight / 2 - camera_to_car_end_distance / (cm_conversion_constant * resolution);

        // col += 500;
        // row += 100;
        // int col1=1600-row;
        // int row1=col;

        // if (row >= 1600)
        // {
        //     ROS_INFO("OUT OF BOUNDS");
        //     continue;
        // }

        if (pt.y < floorY && pt.y > collisionYmax)
        {
            gridMapValue[row][col] = collisionSpace;
            sum_x += col;
            sum_y += row;
            count++;
        }
    }

    int midpoint_x = 0, midpoint_y = 0;

    if (count != 0)
    {
        midpoint_x = sum_x / count;
        midpoint_y = sum_y / count;
    }
    std::cout << count << std::endl;
    if (count > 500)
    {
        publishObjectMidpoint(midpoint_x, midpoint_y);
    }
    // publishOccupancyGrid();
    else
    {
        publishObjectMidpoint(0, 0);
    }
}

void pgmWriter::addPointToOccupancyGrid(double x, double y)
{
    // int col = ((x+320)*mapWidth/2)/640+mapWidth/2;
    // int row = ((y+800)*mapHeight/2)/480+mapHeight/2; // Assuming blind spot is 40cm so 40*8 +480=800
    // int row1=1600-col;
    // int col1=row;
    // row=1600-row;
    // int col = x + 480;   // basically centre of camera which is 320 here plus mapWidth/2
    int col = (x - camx) / (resolution / lane_image_resolution_x) + mapWidth / 2;
    int row = (-y + camy) / (resolution / lane_image_resolution_y) + mapHeight / 2 + (camera_blind_spot - camera_to_car_end_distance) / (cm_conversion_constant * resolution);
    // int row = -y + 1600; // basically centre of camera which is 480 + height of camera x root3 x 8(resolution) + mapHeight/2
    if (row >= 0 && row < mapHeight && col >= 0 && col < mapWidth)
    {
        gridMapValue[row][col] = collisionSpace;
    }

    // for(int i=0;i<800;i++){
    //     gridMapValue[700][i]=collisionSpace;
    // };
}

void pgmWriter::processLanesArray(const std_msgs::Float64MultiArrayConstPtr &lanes_array_msg)
{
    // for (int i = 0; i < mapHeight; i++)
    // {
    //     std::vector<int8_t> temp(mapWidth, freeSpace);
    //     gridMapValue.push_back(temp);
    // }
    std::vector<double> data = lanes_array_msg->data;
    if (data.size() % 2 != 0)
    {
        ROS_ERROR("Invalid number of elements in the Float64MultiArray");
        return;
    }
    ROS_INFO("Entered proc");
    std::vector<Eigen::Vector3d> lanePoints;
    for (size_t i = 0; i < data.size(); i += 2)
    {
        double x = data[i];
        double y = data[i + 1];
        // double x1=-x-320;
        // double y1=(-1*y)-800;
        addPointToOccupancyGrid(x, y);
    }

    publishOccupancyGrid();
}

void pgmWriter::publishObjectMidpoint(int midpoint_x, int midpoint_y)
{
    // If there is no obstacle, x and y will be zero. Z will always be 0.
    // ROS_INFO("entered obj midpoint");
    // Note that these are grid coordinates, not real world coordinates
    geometry_msgs::Point objMidPoint;
    objMidPoint.x = midpoint_x;
    objMidPoint.y = midpoint_y;
    // objMidPoint.x = 0;
    // objMidPoint.y = 0;
    objMidPoint.z = 0;

    midpoint_pub.publish(objMidPoint);
}

void pgmWriter::publishOccupancyGrid()
{
    nav_msgs::OccupancyGrid grid;
    grid.header.stamp = ros::Time::now();
    grid.header.frame_id = "camera_color_optical_frame";
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

    int sum_x = 0, sum_y = 0;
    int count = 0;

    for (int row = 0; row < mapHeight; row++)
    {
        for (int col = 0; col < mapWidth; col++)
        {
            grid.data.push_back(gridMapValue[row][col]);
        }
    }

    pub.publish(grid);
    gridMapValue.clear();
    for (int i = 0; i < mapHeight; i++)
    {
        std::vector<int8_t> temp(mapWidth, freeSpace);
        gridMapValue.push_back(temp);
    }
    rate.sleep();
}

int main(int argc, char **argv)
{
    auto start = std::chrono::high_resolution_clock::now();
    ros::init(argc, argv, "occ_grid");

    pgmWriter writer;

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Time taken to execute the code: " << elapsed.count() << " seconds" << std::endl;

    ros::spin();
    return 0;
}
