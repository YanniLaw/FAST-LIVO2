/* 
This file is part of FAST-LIVO2: Fast, Direct LiDAR-Inertial-Visual Odometry.

Developer: Chunran Zheng <zhengcr@connect.hku.hk>

For commercial use, please contact me at <zhengcr@connect.hku.hk> or
Prof. Fu Zhang at <fuzhang@hku.hk>.

This file is subject to the terms and conditions outlined in the 'LICENSE' file,
which is included as part of this source code package.
*/

#ifndef VOXEL_MAP_H_
#define VOXEL_MAP_H_

#include "common_lib.h"
#include <Eigen/Dense>
#include <fstream>
#include <math.h>
#include <mutex>
#include <omp.h>
#include <pcl/common/io.h>
#include <ros/ros.h>
#include <thread>
#include <unistd.h>
#include <unordered_map>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#define VOXELMAP_HASH_P 116101
#define VOXELMAP_MAX_N 10000000000

static int voxel_plane_id = 0;

// 体素地图配置结构体，用于存储体素地图的各种参数设置
typedef struct VoxelMapConfig
{
  double max_voxel_size_;             // 根体素大小
  int max_layer_;                     // 八叉树最大层数
  int max_iterations_;                // 最大迭代次数
  std::vector<int> layer_init_num_;   // 每一层的初始点数
  int max_points_num_;                // 每个体素的最大点数
  double planner_threshold_;          // 平面拟合判定阈值
  double beam_err_;                   // 激光束误差
  double dept_err_;                   // 深度误差
  double sigma_num_;                  // 不确定性倍数
  bool is_pub_plane_map_;             // 是否发布平面地图

  // config of local map sliding
  double sliding_thresh;              // 滑动阈值
  bool map_sliding_en;                // 是否启用地图滑动
  int half_map_size;                  // 地图半尺寸
} VoxelMapConfig;

// 点到平面残差结构体，用于存储点到平面的匹配信息
typedef struct PointToPlane
{
  Eigen::Vector3d point_b_; // 点在机体坐标系下的坐标
  Eigen::Vector3d point_w_; // 点在世界坐标系下的坐标
  Eigen::Vector3d normal_;  // 法向量
  Eigen::Vector3d center_;  // 平面中心点
  Eigen::Matrix<double, 6, 6> plane_var_; // 平面参数的协方差矩阵
  M3D body_cov_;        // 点在机体坐标系下的协方差矩阵
  int layer_;           // 八叉树层数
  double d_;            // 平面方程中的d参数
  double eigen_value_;  // 特征值
  bool is_valid_;       // 是否有效
  float dis_to_plane_;  // 点到平面的距离(带符号，用于 EKF)
} PointToPlane;

// 体素平面结构体，用于存储体素平面的各种属性
// 平面参数可近似看成 π = [n; q]，其中 n 为法向量 normal，q 为平面中心点 center
// 所以 plane_var_ 为∑_π = [∑nn  ∑nq; ∑qn  ∑qq]
typedef struct VoxelPlane
{
  Eigen::Vector3d center_;      // 八叉树体素平面中心点
  Eigen::Vector3d normal_;      // 法向量
  Eigen::Vector3d y_normal_;    // y方向法向量
  Eigen::Vector3d x_normal_;    // x方向法向量
  Eigen::Matrix3d covariance_;  // 协方差矩阵
  Eigen::Matrix<double, 6, 6> plane_var_; // 平面参数的协方差矩阵 
  float radius_ = 0;  // 平面的半径
  float min_eigen_value_ = 1;   // 最小特征值
  float mid_eigen_value_ = 1;   // 中间特征值
  float max_eigen_value_ = 1;   // 最大特征值
  float d_ = 0;                  // 平面方程中的d参数
  int points_size_ = 0;          // 八叉树节点中点的数量
  bool is_plane_ = false;        // 是否为平面
  bool is_init_ = false;         // 是否初始化
  int id_ = 0;                   // 平面ID
  bool is_update_ = false;       // 是否更新
  VoxelPlane()
  {
    plane_var_ = Eigen::Matrix<double, 6, 6>::Zero();
    covariance_ = Eigen::Matrix3d::Zero();
    center_ = Eigen::Vector3d::Zero();
    normal_ = Eigen::Vector3d::Zero();
  }
} VoxelPlane;

// 根体素在三维空间中的整数坐标，用于哈希定位体素
class VOXEL_LOCATION
{
public:
  int64_t x, y, z; // 根体素在三维空间中的整数坐标

  VOXEL_LOCATION(int64_t vx = 0, int64_t vy = 0, int64_t vz = 0) : x(vx), y(vy), z(vz) {}

  bool operator==(const VOXEL_LOCATION &other) const { return (x == other.x && y == other.y && z == other.z); }
};

// Hash value
namespace std
{
template <> struct hash<VOXEL_LOCATION>
{
  int64_t operator()(const VOXEL_LOCATION &s) const
  {
    using std::hash;
    using std::size_t;
    return ((((s.z) * VOXELMAP_HASH_P) % VOXELMAP_MAX_N + (s.y)) * VOXELMAP_HASH_P) % VOXELMAP_MAX_N + (s.x);
  }
};
} // namespace std

struct DS_POINT
{
  float xyz[3];
  float intensity;
  int count = 0;
};

void calcBodyCov(Eigen::Vector3d &pb, const float range_inc, const float degree_inc, Eigen::Matrix3d &cov);

class VoxelOctoTree
{

public:
  VoxelOctoTree() = default;
  std::vector<pointWithVar> temp_points_; // 八叉树节点的临时点云，用于存储当前节点积累的点
  VoxelPlane *plane_ptr_;
  int layer_; // 当前八叉树节点所在的层级
  int octo_state_; // 八叉树节点状态，0表示八叉树的末端，1表示不是末端
  VoxelOctoTree *leaves_[8]; // 八叉树节点的子节点指针数组，最多有8个子节点
  double voxel_center_[3];   // 八叉树节点的中心坐标 x, y, z
  std::vector<int> layer_init_num_; // 每一层八叉树初始化的点数(每一层至少积累多少点以后，才开始判断这个节点是否能够形成平面)
  // 这里是为了方便计算后面的child center offset, 即每个子节点的中心相对于父节点的偏移量可以通过 quater_length_ 来计算
  // 假设根体素 size = s , 那么一个child的size为 s/2 , child center 距离 parent center 的偏移量为 s/4，即 quater_length_
  float quater_length_; // 八叉树节点的四分之一边长，用于判断子节点的空间范围
  float planer_threshold_;  // 八叉树节点的平面拟合阈值
  int points_size_threshold_; // 八叉树节点的点数阈值，当节点积累的点数达到该阈值时，才开始判断是否形成平面
  int update_size_threshold_; // 八叉树节点的更新点数阈值，当节点积累的更新点数达到该阈值时，才触发更新
  int max_points_num_;  // 八叉树节点允许的最大点数
  int max_layer_;       // 最大八叉树层数  每一层的voxel大小与层数有关 s_l = s_0 / 2^l
  int new_points_;      // 八叉树节点中新加入的点数
  bool init_octo_;      // 八叉树节点是否已经初始化
  bool update_enable_;  // 八叉树节点是否允许更新

  VoxelOctoTree(int max_layer, int layer, int points_size_threshold, int max_points_num, float planer_threshold)
      : max_layer_(max_layer), layer_(layer), points_size_threshold_(points_size_threshold), max_points_num_(max_points_num),
        planer_threshold_(planer_threshold)
  {
    temp_points_.clear();
    octo_state_ = 0;
    new_points_ = 0;
    update_size_threshold_ = 5;
    init_octo_ = false;
    update_enable_ = true;
    for (int i = 0; i < 8; i++)
    {
      leaves_[i] = nullptr;
    }
    plane_ptr_ = new VoxelPlane;
  }

  ~VoxelOctoTree()
  {
    for (int i = 0; i < 8; i++)
    {
      delete leaves_[i];
    }
    delete plane_ptr_;
  }
  void init_plane(const std::vector<pointWithVar> &points, VoxelPlane *plane);
  void init_octo_tree();
  void cut_octo_tree();
  void UpdateOctoTree(const pointWithVar &pv);

  VoxelOctoTree *find_correspond(Eigen::Vector3d pw);
  VoxelOctoTree *Insert(const pointWithVar &pv);
};

void loadVoxelConfig(ros::NodeHandle &nh, VoxelMapConfig &voxel_config);

class VoxelMapManager
{
public:
  VoxelMapManager() = default;
  VoxelMapConfig config_setting_;
  int current_frame_id_ = 0;
  ros::Publisher voxel_map_pub_;
  std::unordered_map<VOXEL_LOCATION, VoxelOctoTree *> voxel_map_; // 八叉树体素地图，键为体素位置，值为对应的八叉树节点指针

  PointCloudXYZI::Ptr feats_undistort_;
  PointCloudXYZI::Ptr feats_down_body_;
  PointCloudXYZI::Ptr feats_down_world_;

  M3D extR_;
  V3D extT_;
  float build_residual_time, ekf_time;
  float ave_build_residual_time = 0.0;
  float ave_ekf_time = 0.0;
  int scan_count = 0;
  StatesGroup state_;
  V3D position_last_;

  V3D last_slide_position = {0,0,0};

  geometry_msgs::Quaternion geoQuat_;

  int feats_down_size_;
  int effct_feat_num_;
  std::vector<M3D> cross_mat_list_;
  std::vector<M3D> body_cov_list_;
  std::vector<pointWithVar> pv_list_;
  std::vector<PointToPlane> ptpl_list_;

  VoxelMapManager(VoxelMapConfig &config_setting, std::unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &voxel_map)
      : config_setting_(config_setting), voxel_map_(voxel_map)
  {
    current_frame_id_ = 0;
    feats_undistort_.reset(new PointCloudXYZI());
    feats_down_body_.reset(new PointCloudXYZI());
    feats_down_world_.reset(new PointCloudXYZI());
  };

  void StateEstimation(StatesGroup &state_propagat);
  void TransformLidar(const Eigen::Matrix3d rot, const Eigen::Vector3d t, const PointCloudXYZI::Ptr &input_cloud,
                      pcl::PointCloud<pcl::PointXYZI>::Ptr &trans_cloud);

  void BuildVoxelMap();
  V3F RGBFromVoxel(const V3D &input_point);

  void UpdateVoxelMap(const std::vector<pointWithVar> &input_points);

  void BuildResidualListOMP(std::vector<pointWithVar> &pv_list, std::vector<PointToPlane> &ptpl_list);

  void build_single_residual(pointWithVar &pv, const VoxelOctoTree *current_octo, const int current_layer, bool &is_sucess, double &prob,
                             PointToPlane &single_ptpl);

  void pubVoxelMap();

  void mapSliding();
  void clearMemOutOfMap(const int& x_max,const int& x_min,const int& y_max,const int& y_min,const int& z_max,const int& z_min );

private:
  void GetUpdatePlane(const VoxelOctoTree *current_octo, const int pub_max_voxel_layer, std::vector<VoxelPlane> &plane_list);

  void pubSinglePlane(visualization_msgs::MarkerArray &plane_pub, const std::string plane_ns, const VoxelPlane &single_plane, const float alpha,
                      const Eigen::Vector3d rgb);
  void CalcVectQuation(const Eigen::Vector3d &x_vec, const Eigen::Vector3d &y_vec, const Eigen::Vector3d &z_vec, geometry_msgs::Quaternion &q);

  void mapJet(double v, double vmin, double vmax, uint8_t &r, uint8_t &g, uint8_t &b);
};
typedef std::shared_ptr<VoxelMapManager> VoxelMapManagerPtr;

#endif // VOXEL_MAP_H_