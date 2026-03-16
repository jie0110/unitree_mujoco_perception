#pragma once

#include <eigen3/Eigen/Dense>
#include <unitree/dds_wrapper/common/Publisher.h>

#include <unitree/idl/ros2/PointCloud2_.hpp>

namespace unitree
{
namespace robot
{
namespace g1
{
namespace publisher
{

class CameraData : public RealTimePublisher<sensor_msgs::msg::dds_::PointCloud2_>
{
public:
    CameraData(std::string topic = "rt/cameradata") : RealTimePublisher<MsgType>(topic)
    {
    }

  };
};
};
};
};
