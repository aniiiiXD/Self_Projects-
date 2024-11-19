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
#include <vector>
#include <chrono>

typedef pcl::PointCloud<pcl::PointXYZ>::Ptr pointCloudPtr;

class pgmWriter
{
public:
    double resolution = 0.00125;
    // double margin_m = 0.05;
    int8_t freeSpace = 0;
    int8_t collisionSpace = 100;
    double floorY = 0.185; 
    double collisionYmax = 0.115;
    // double searchRadius = 0.06;
    int mapWidth = 1600;
    int mapHeight = 1600;
    int camx=10;
    int camy=10;
    ros::Publisher pub;
    std::vector<std::vector<int8_t>> gridMapValue;

    pgmWriter()
    {
        for (int i = 0; i < mapHeight; i++)
        {
            std::vector<int8_t> temp(mapWidth, freeSpace);
            gridMapValue.push_back(temp);
        }
    }

    void setPcdMap(const sensor_msgs::PointCloud2ConstPtr& input);
    void pcdToOccupancyGrid(pointCloudPtr pcdPoint);
    void addPointToOccupancyGrid(double x, double y);
    void processLanesArray(const std_msgs::Float64MultiArrayConstPtr& lanes_array_msg);
    void publishOccupancyGrid();

private:
    pointCloudPtr pcdMap;
    std_msgs::Float64MultiArray occupancyGridData; // Variable to store occupancy grid data
    sensor_msgs::ImageConstPtr depthImageData;
};


void pgmWriter::setPcdMap(const sensor_msgs::PointCloud2ConstPtr& input)
{

    // for (int i = 0; i < mapHeight; i++)
    // {
    //     std::vector<int8_t> temp(mapWidth, freeSpace);
    //     gridMapValue.push_back(temp);
    // }

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
        // int col1=1600-row;
        // int row1=col;


        if (pt.y < floorY && pt.y > collisionYmax)
        {
            gridMapValue[row][col] = collisionSpace;
        }
    }
    // publishOccupancyGrid();
}

void pgmWriter::addPointToOccupancyGrid(double x, double y)
{
    // int col = ((x+320)*mapWidth/2)/640+mapWidth/2;
    // int row = ((y+800)*mapHeight/2)/480+mapHeight/2; // Assuming blind spot is 40cm so 40*8 +480=800
    // int row1=1600-col;
    // int col1=row;
    // row=1600-row;
    int col=x+480;  // basically centre of camera which is 320 here plus mapWidth/2
    int row=-y+1600; // basically centre of camera which is 480 + height of camera x root3 x 8(resolution) + mapHeight/2
    if (row >= 0 && row < mapHeight && col >= 0 && col < mapWidth)
    {
        gridMapValue[row][col] = collisionSpace;
    }

    // for(int i=0;i<800;i++){
    //     gridMapValue[700][i]=collisionSpace;
    // };

}

void pgmWriter::processLanesArray(const std_msgs::Float64MultiArrayConstPtr& lanes_array_msg)
{
    // for (int i = 0; i < mapHeight; i++)
    // {
    //     std::vector<int8_t> temp(mapWidth, freeSpace);
    //     gridMapValue.push_back(temp);
    // }
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
        // double x1=-x-320;
        // double y1=(-1*y)-800;
        addPointToOccupancyGrid(x,y);
    }

    publishOccupancyGrid();
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
}

int main(int argc, char **argv)
{   auto start = std::chrono::high_resolution_clock::now();
    ros::init(argc, argv, "occ_grid");
    ros::NodeHandle nh;
    pgmWriter writer;
    writer.pub = nh.advertise<nav_msgs::OccupancyGrid>("/occupancy_grid", 1);

    ros::Subscriber sub = nh.subscribe<sensor_msgs::PointCloud2>("/camera/depth_registered/points", 1, &pgmWriter::setPcdMap, &writer);
    ros::Subscriber sub1 = nh.subscribe<std_msgs::Float64MultiArray>("/occupancy_points", 1, &pgmWriter::processLanesArray, &writer);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    std::cout << "Time taken to execute the code: " << elapsed.count() << " seconds" << std::endl;

    ros::spin();
    return 0;
}
