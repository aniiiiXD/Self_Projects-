#include <pcl/point_types.h>
#include <pcl/filters/conditional_removal.h>
#include <pcl/io/pcd_io.h>
#include <pcl/octree/octree_search.h>
#include <pcl/common/common.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <std_msgs/Float64MultiArray.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/crop_box.h>
#include <vector>
#include <boost/shared_ptr.hpp>
#include <pcl/point_cloud.h>
#include <nav_msgs/OccupancyGrid.h>

typedef pcl::PointXYZ Point;
typedef pcl::PointCloud<Point> PointCloud;
typedef boost::shared_ptr<PointCloud> PointCloudPtr;

class PgmWriter
{
public:
    PointCloudPtr pcdMap;
    std::string pgmFileName;
    double resolution = 0.01;
    ros::Publisher pub;
    ros::Subscriber sub;
    int mapHeight = 1;
    int8_t freeSpace = 0;

    PgmWriter(ros::NodeHandle& nh)
    {
        pub = nh.advertise<nav_msgs::OccupancyGrid>("/occupancy_grid_signs", 1);
        sub = nh.subscribe<sensor_msgs::PointCloud2>("/camera/depth_registered/points", 1, &PgmWriter::setPcdMap, this);
    }
    void setPcdMap(const sensor_msgs::PointCloud2ConstPtr& input)
        {
            pcdMap.reset(new PointCloud());
            pcl::fromROSMsg(*input, *pcdMap);
            pgmFileName = "TestPGM";
            pcdToOccupancyGrid(pcdMap);
            // ROS_INFO("Entered set");
        }
        
    void pcdToOccupancyGrid(PointCloudPtr pcdPoint)
    {
        double k = 1;
        std::vector<double> points_pub;
        for (const auto& pt : pcdPoint->points)
        {
            if (std::isnan(pt.x) || std::isnan(pt.y) || std::isnan(pt.z))
            {
                continue;
            }
            double a =std::abs(pt.z - k);
            double b = 0.2;
            double ax=std::abs(std::abs(pt.x + b));
            if (a < 0.1 && pt.y <= 0.06 && pt.y >= -0.01 && ax<0.05)
            {
                points_pub.push_back(pt.x);
                points_pub.push_back(pt.y);
                points_pub.push_back(pt.z);
                // ROS_INFO("Pushed");
            }
        }
        if (points_pub.size() % 3 != 0)
        {
            ROS_WARN("Mismatch in point coordinates count. Points will be discarded.");
            return;
        }
        
        int mapWidth=points_pub.size();
        std::vector<std::vector<int8_t>> gridMapValue; 

        for (int i = 0; i < this->mapHeight; i++)
        {
            std::vector<int8_t> temp;
            temp.resize(mapWidth, freeSpace);
            gridMapValue.push_back(temp);
        }

        for(int i=0;i<points_pub.size();i++){
            gridMapValue[0][i] = points_pub[i]*100;
        }
        // ROS_INFO("Received Points:");
        //     for (size_t i = 0; i < points_pub.size(); i += 3)
        //     {
        //         double x = points_pub[i];
        //         double y = points_pub[i + 1];
        //         double z = points_pub[i + 2];
        //         // ROS_INFO("Point %zu: (%.2f, %.2f, %.2f)", i / 3, x, y, z);
        //     }
        nav_msgs::OccupancyGrid grid;
        grid.header.stamp = ros::Time::now();
        grid.header.frame_id = "map";
        grid.info.resolution = this->resolution;
        grid.info.width = mapWidth;
        grid.info.height = mapHeight;
        grid.info.origin.position.x = 0;
        grid.info.origin.position.y = 0;
        grid.info.origin.position.z = 0;
        grid.info.origin.orientation.x = 0;
        grid.info.origin.orientation.y = 0;
        grid.info.origin.orientation.z = 0;
        grid.info.origin.orientation.w = 1;

        for (int col = 0; col < mapWidth; col++)
        {
            grid.data.push_back(gridMapValue[0][col]);
        }

        pub.publish(grid);
        gridMapValue.clear();
        }
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "pgm_writer");
    ros::NodeHandle nh;
    PgmWriter writer(nh);
    ros::spin();
    return 0;
}