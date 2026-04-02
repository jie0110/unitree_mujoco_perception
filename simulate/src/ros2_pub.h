#pragma once

#include <vector>
#include <cstdint>
#include <cmath>
#include <string>
#include <algorithm>

#include <eigen3/Eigen/Dense>
#include <unitree/dds_wrapper/common/Publisher.h>

#include <unitree/idl/ros2/PointCloud2_.hpp>
#include <unitree/idl/ros2/PointField_.hpp>

namespace unitree
{
namespace robot
{
namespace g1
{
namespace publisher
{

class CameraData
{
public:
    CameraData(std::string topic1 = "rt/camera/depth", std::string topic2 = "rt/camera/points")
        : camera_pub_(std::make_unique<RealTimePublisher<sensor_msgs::msg::dds_::PointCloud2_>>(topic1)), pointcloud_pub_(std::make_unique<RealTimePublisher<sensor_msgs::msg::dds_::PointCloud2_>>(topic2)) {

    }

    void publishDepth(const double* sensordata, int offset, int32_t time_sec, int32_t time_nanosec) {

        if (camera_pub_->trylock()) {
            camera_pub_->msg_.header().frame_id() = std::string("camera");
            camera_pub_->msg_.header().stamp().sec() = time_sec;
            camera_pub_->msg_.header().stamp().nanosec() = time_nanosec;

            int dim = height * width;

            camera_pub_->msg_.height() = 1;
            camera_pub_->msg_.width() = dim;

            camera_pub_->msg_.fields().resize(1);
            camera_pub_->msg_.fields()[0].name() = "depth";
            camera_pub_->msg_.fields()[0].offset() = 0;
            camera_pub_->msg_.fields()[0].datatype() = sensor_msgs::msg::dds_::PointField_Constants::FLOAT32_;
            camera_pub_->msg_.fields()[0].count() = 1;
            camera_pub_->msg_.is_bigendian() = false;
            camera_pub_->msg_.point_step() = 1 * sizeof(float);  // 1 floats * 4 bytes
            camera_pub_->msg_.row_step() = camera_pub_->msg_.point_step() * dim;
            camera_pub_->msg_.is_dense() = true;

            depth_buffer_.resize(dim);

            for (int i = 0; i < dim; ++i) {
                float data = static_cast<float>(sensordata[offset + i]);
                if (data == -1) {
                    data = camera_cutoff;
                }
                depth_buffer_[i] = data;
            }

            std::vector<uint8_t> data(depth_buffer_.size() * sizeof(float));

            std::memcpy(data.data(), depth_buffer_.data(), depth_buffer_.size() * sizeof(float));

            camera_pub_->msg_.data().swap(data);

            camera_pub_->unlockAndPublish();
        }

    }

    void publishPointCloud(const double* sensordata, int offset, int32_t time_sec, int32_t time_nanosec) {
        if (pointcloud_pub_->trylock()) {
            // Calculate focal length
            float vfov_rad = vfov_deg * M_PI / 180.0f;
            float f = (height / 2.0f) / std::tan(vfov_rad / 2.0f);  // focal length in pixels

            // Principal point (assuming square pixels, fx = fy = f)
            float cx = width / 2.0f;
            float cy = height / 2.0f;

            // First pass: count valid points (depth > 0)
            int valid_count = 0;
            for (int i = 0; i < height * width; ++i) {
                if (sensordata[offset + i] > 0.0f) {
                    ++valid_count;
                }
            }

            if (valid_count == 0) {
                pointcloud_pub_->unlockAndPublish();
                return;
            }

            points_buffer_.clear();

            // Allocate memory for points
            // std::vector<float> points(valid_count * 3);
            int point_idx = 0;

            // Second pass: compute 3D points for valid depth values
            for (int v = 0; v < height; ++v) {
                for (int u = 0; u < width; ++u) {
                    int idx = v * width + u;
                    float d = sensordata[offset + idx];

                    if (d <= 0.0f) {
                        continue;  // Skip invalid points
                    }

                    // Convert to 3D coordinates
                    float x = (u - cx) * d / f;
                    float y = (v - cy) * d / f;
                    float z = d;

                    // Apply -135 degree rotation around +Y
                    double rad = -135 * M_PI / 180.0;
                    double cosA = cos(rad);
                    double sinA = sin(rad);
                    float final_x = x * cosA + z * sinA;
                    float final_y = y;
                    float final_z = -x * sinA + z * cosA;

                    // Store the point
                    points_buffer_.push_back(final_x);
                    points_buffer_.push_back(final_y);
                    points_buffer_.push_back(final_z);
                    ++point_idx;
                }
            }

            pointcloud_pub_->msg_.header().frame_id() = std::string("camera");
            pointcloud_pub_->msg_.header().stamp().sec() = time_sec;
            pointcloud_pub_->msg_.header().stamp().nanosec() = time_nanosec;

            pointcloud_pub_->msg_.height() = 1;
            pointcloud_pub_->msg_.width() = valid_count;

            pointcloud_pub_->msg_.fields().resize(3);
            pointcloud_pub_->msg_.fields()[0].name() = "x";
            pointcloud_pub_->msg_.fields()[0].offset() = 0;
            pointcloud_pub_->msg_.fields()[0].datatype() = sensor_msgs::msg::dds_::PointField_Constants::FLOAT32_;
            pointcloud_pub_->msg_.fields()[0].count() = 1;

            pointcloud_pub_->msg_.fields()[1].name() = "y";
            pointcloud_pub_->msg_.fields()[1].offset() = 4;
            pointcloud_pub_->msg_.fields()[1].datatype() = sensor_msgs::msg::dds_::PointField_Constants::FLOAT32_;
            pointcloud_pub_->msg_.fields()[1].count() = 1;

            pointcloud_pub_->msg_.fields()[2].name() = "z";
            pointcloud_pub_->msg_.fields()[2].offset() = 8;
            pointcloud_pub_->msg_.fields()[2].datatype() = sensor_msgs::msg::dds_::PointField_Constants::FLOAT32_;
            pointcloud_pub_->msg_.fields()[2].count() = 1;

            pointcloud_pub_->msg_.is_bigendian() = false;
            pointcloud_pub_->msg_.point_step() = 3 * sizeof(float);  // 3 floats * 4 bytes
            pointcloud_pub_->msg_.row_step() = pointcloud_pub_->msg_.point_step() * pointcloud_pub_->msg_.width();
            pointcloud_pub_->msg_.is_dense() = true;

            std::vector<uint8_t> data(points_buffer_.size() * sizeof(float));

            std::memcpy(data.data(), points_buffer_.data(), points_buffer_.size() * sizeof(float));

            pointcloud_pub_->msg_.data().swap(data);

            pointcloud_pub_->unlockAndPublish();
        }


    }

public:
    int width = 64;
    int height = 36;
    float vfov_deg = 58.0f;
    float camera_cutoff = 10.0f;

private:
    std::unique_ptr<RealTimePublisher<sensor_msgs::msg::dds_::PointCloud2_>> camera_pub_;
    std::unique_ptr<RealTimePublisher<sensor_msgs::msg::dds_::PointCloud2_>> pointcloud_pub_;

    std::vector<float> depth_buffer_;
    std::vector<float> points_buffer_;
};

};
};
};
};
