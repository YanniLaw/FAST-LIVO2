/* 
This file is part of FAST-LIVO2: Fast, Direct LiDAR-Inertial-Visual Odometry.

Developer: Chunran Zheng <zhengcr@connect.hku.hk>

For commercial use, please contact me at <zhengcr@connect.hku.hk> or
Prof. Fu Zhang at <fuzhang@hku.hk>.

This file is subject to the terms and conditions outlined in the 'LICENSE' file,
which is included as part of this source code package.
*/

#include "voxel_map.h"

/**
 * @brief 计算激光点在机体坐标系下的协方差矩阵，考虑激光测距误差和角度误差
 * 测量不是一个完美确定的点，而是 p ~ N(p, cov) 的高斯分布。其中误差主要来源于: 
 * range /depth error 和 beam angular error
 * calcBodyCov 依据激光雷达的距离分辨率 dept_err_ 和角度分辨率 beam_err_，
 * 在机体坐标系里估计每个点的测量噪声协方差（径向为测距误差、切向为角度误差）
 * @param pb  激光点在机体坐标系下的坐标
 * @param range_inc 激光测距误差
 * @param degree_inc 激光角度误差
 * @param cov  输出的协方差矩阵
 */
void calcBodyCov(Eigen::Vector3d &pb, const float range_inc, const float degree_inc, Eigen::Matrix3d &cov)
{
  if (pb[2] == 0) pb[2] = 0.0001;
  float range = sqrt(pb[0] * pb[0] + pb[1] * pb[1] + pb[2] * pb[2]);
  float range_var = range_inc * range_inc;
  Eigen::Matrix2d direction_var;
  direction_var << pow(sin(DEG2RAD(degree_inc)), 2), 0, 0, pow(sin(DEG2RAD(degree_inc)), 2);
  Eigen::Vector3d direction(pb);
  direction.normalize();
  Eigen::Matrix3d direction_hat;
  direction_hat << 0, -direction(2), direction(1), direction(2), 0, -direction(0), -direction(1), direction(0), 0;
  Eigen::Vector3d base_vector1(1, 1, -(direction(0) + direction(1)) / direction(2));
  base_vector1.normalize();
  Eigen::Vector3d base_vector2 = base_vector1.cross(direction);
  base_vector2.normalize();
  Eigen::Matrix<double, 3, 2> N;
  N << base_vector1(0), base_vector2(0), base_vector1(1), base_vector2(1), base_vector1(2), base_vector2(2);
  Eigen::Matrix<double, 3, 2> A = range * direction_hat * N;
  cov = direction * range_var * direction.transpose() + A * direction_var * A.transpose();
}

void loadVoxelConfig(ros::NodeHandle &nh, VoxelMapConfig &voxel_config)
{
  nh.param<bool>("publish/pub_plane_en", voxel_config.is_pub_plane_map_, false);
  
  nh.param<int>("lio/max_layer", voxel_config.max_layer_, 1);
  nh.param<double>("lio/voxel_size", voxel_config.max_voxel_size_, 0.5);
  nh.param<double>("lio/min_eigen_value", voxel_config.planner_threshold_, 0.01);
  nh.param<double>("lio/sigma_num", voxel_config.sigma_num_, 3);
  nh.param<double>("lio/beam_err", voxel_config.beam_err_, 0.02);
  nh.param<double>("lio/dept_err", voxel_config.dept_err_, 0.05);
  nh.param<vector<int>>("lio/layer_init_num", voxel_config.layer_init_num_, vector<int>{5,5,5,5,5});
  nh.param<int>("lio/max_points_num", voxel_config.max_points_num_, 50);
  nh.param<int>("lio/max_iterations", voxel_config.max_iterations_, 5);

  nh.param<bool>("local_map/map_sliding_en", voxel_config.map_sliding_en, false);
  nh.param<int>("local_map/half_map_size", voxel_config.half_map_size, 100);
  nh.param<double>("local_map/sliding_thresh", voxel_config.sliding_thresh, 8);
}

/**
 * @brief 初始化八叉树节点的平面信息
 * 给定八叉树节点里积累的一批世界系点（每个点自带协方差 var），用 PCA（协方差矩阵特征分解） 
 * 判断它们能否构成一个平面；如果能，就顺便用误差传播公式算出平面参数自身的协方差 plane_var
 * 调用点: 
 * 1. init_octo_tree() —— 初始化建图阶段，点数超过阈值时尝试拟合平面;
 * 2. cut_octo_tree() —— 父节点不是平面、递归细分到子节点后再对子节点拟合;
 * 3. UpdateOctoTree() —— 地图更新阶段，新点积累超过 update_size_threshold_ 时重新拟合.
 * @param points  八叉树节点积累的点云
 * @param plane  八叉树节点的平面信息结构体
 */
void VoxelOctoTree::init_plane(const std::vector<pointWithVar> &points, VoxelPlane *plane)
{
  plane->plane_var_ = Eigen::Matrix<double, 6, 6>::Zero();
  plane->covariance_ = Eigen::Matrix3d::Zero();
  plane->center_ = Eigen::Vector3d::Zero();
  plane->normal_ = Eigen::Vector3d::Zero();
  plane->points_size_ = points.size();
  plane->radius_ = 0;
  for (auto pv : points)
  {
    plane->covariance_ += pv.point_w * pv.point_w.transpose();
    plane->center_ += pv.point_w;
  }
  // 计算该八叉树体素中p oints 的均值
  plane->center_ = plane->center_ / plane->points_size_;
  // 计算八叉树体素中 points 协方差矩阵 注意用的是无偏置的1/N
  plane->covariance_ = plane->covariance_ / plane->points_size_ - plane->center_ * plane->center_.transpose();
  // 对协方差矩阵进行特征分解，得到特征值和特征向量
  Eigen::EigenSolver<Eigen::Matrix3d> es(plane->covariance_);
  Eigen::Matrix3cd evecs = es.eigenvectors();
  Eigen::Vector3cd evals = es.eigenvalues();
  Eigen::Vector3d evalsReal;
  evalsReal = evals.real(); // EigenSolver计算的特征值和特征向量是复数
  Eigen::Matrix3f::Index evalsMin, evalsMax;
  evalsReal.rowwise().sum().minCoeff(&evalsMin);
  evalsReal.rowwise().sum().maxCoeff(&evalsMax);
  // 小技巧： 因为下表集合是 {0, 1, 2}，三者之和恒为3
  int evalsMid = 3 - evalsMin - evalsMax;
  Eigen::Vector3d evecMin = evecs.real().col(evalsMin); // 法向量
  Eigen::Vector3d evecMid = evecs.real().col(evalsMid); // 切平面次主轴
  Eigen::Vector3d evecMax = evecs.real().col(evalsMax); // 切平面内主轴
  Eigen::Matrix3d J_Q;
  J_Q << 1.0 / plane->points_size_, 0, 0, 0, 1.0 / plane->points_size_, 0, 0, 0, 1.0 / plane->points_size_;
  // && evalsReal(evalsMid) > 0.05
  //&& evalsReal(evalsMid) > 0.01
  /*
  这是整个函数最值得理解的部分。目标：每个点 pi有一个微扰 δpi，求它对平面参数的影响。
  平面参数取向量6维 [n; c]，其中 n 为法向量 normal，q 为平面中心点 center
  */
  if (evalsReal(evalsMin) < planer_threshold_)
  {
    for (int i = 0; i < points.size(); i++)
    {
      Eigen::Matrix<double, 6, 3> J;
      Eigen::Matrix3d F;
      for (int m = 0; m < 3; m++)
      {
        if (m != (int)evalsMin)
        {
          Eigen::Matrix<double, 1, 3> F_m =
              (points[i].point_w - plane->center_).transpose() / ((plane->points_size_) * (evalsReal[evalsMin] - evalsReal[m])) *
              (evecs.real().col(m) * evecs.real().col(evalsMin).transpose() + evecs.real().col(evalsMin) * evecs.real().col(m).transpose());
          F.row(m) = F_m;
        }
        else
        {
          Eigen::Matrix<double, 1, 3> F_m;
          F_m << 0, 0, 0;
          F.row(m) = F_m;
        }
      }
      J.block<3, 3>(0, 0) = evecs.real() * F;
      J.block<3, 3>(3, 0) = J_Q;
      plane->plane_var_ += J * points[i].var * J.transpose();
    }

    plane->normal_ << evecs.real()(0, evalsMin), evecs.real()(1, evalsMin), evecs.real()(2, evalsMin);
    plane->y_normal_ << evecs.real()(0, evalsMid), evecs.real()(1, evalsMid), evecs.real()(2, evalsMid);
    plane->x_normal_ << evecs.real()(0, evalsMax), evecs.real()(1, evalsMax), evecs.real()(2, evalsMax);
    plane->min_eigen_value_ = evalsReal(evalsMin);
    plane->mid_eigen_value_ = evalsReal(evalsMid);
    plane->max_eigen_value_ = evalsReal(evalsMax);
    plane->radius_ = sqrt(evalsReal(evalsMax));
    plane->d_ = -(plane->normal_(0) * plane->center_(0) + plane->normal_(1) * plane->center_(1) + plane->normal_(2) * plane->center_(2));
    plane->is_plane_ = true;
    plane->is_update_ = true;
    if (!plane->is_init_)
    {
      plane->id_ = voxel_plane_id;
      voxel_plane_id++;
      plane->is_init_ = true;
    }
  }
  else
  {
    plane->is_update_ = true;
    plane->is_plane_ = false;
  }
}

// 初始化八叉树
// 它负责决定当前这个 voxel 节点，是“直接作为平面叶子节点使用”，还是“继续切成 8 个子 voxel”
void VoxelOctoTree::init_octo_tree()
{
  // 如果积累的点数超过阈值，则尝试拟合平面(一般这个阈值比冻结平面的阈值小很多)
  if (temp_points_.size() > points_size_threshold_)
  {
    init_plane(temp_points_, plane_ptr_);
    if (plane_ptr_->is_plane_ == true) // 能构成平面
    {
      octo_state_ = 0; // 本节点成为叶子节点
      // new added
      // 如果这个 voxel 已经是稳定平面，而且样本已经很多，就不再持续保存所有历史点
      if (temp_points_.size() > max_points_num_)
      {
        update_enable_ = false; // 这个plane已经很成熟，冻结，不再接受更新
        std::vector<pointWithVar>().swap(temp_points_); // 真正释放内存
        new_points_ = 0;
      }
    }
    else // 不能构成平面
    {
      octo_state_ = 1; // 本节点成为内部节点，还可继续往下细分
      cut_octo_tree(); // 细分到 8 个子节点
    }
    init_octo_ = true; // 标记已完成初始化
    new_points_ = 0;   // 清空新增点计数
  }
}

void VoxelOctoTree::cut_octo_tree()
{
  if (layer_ >= max_layer_) // 最多只能细分多少层
  {
    octo_state_ = 0; // 达到最大层数，当前节点成为叶子节点
    return;
  }
  for (size_t i = 0; i < temp_points_.size(); i++)
  {
    int xyz[3] = {0, 0, 0};
    if (temp_points_[i].point_w[0] > voxel_center_[0]) { xyz[0] = 1; }
    if (temp_points_[i].point_w[1] > voxel_center_[1]) { xyz[1] = 1; }
    if (temp_points_[i].point_w[2] > voxel_center_[2]) { xyz[2] = 1; }
    int leafnum = 4 * xyz[0] + 2 * xyz[1] + xyz[2];
    if (leaves_[leafnum] == nullptr)
    {
      leaves_[leafnum] = new VoxelOctoTree(max_layer_, layer_ + 1, layer_init_num_[layer_ + 1], max_points_num_, planer_threshold_);
      leaves_[leafnum]->layer_init_num_ = layer_init_num_;
      leaves_[leafnum]->voxel_center_[0] = voxel_center_[0] + (2 * xyz[0] - 1) * quater_length_;
      leaves_[leafnum]->voxel_center_[1] = voxel_center_[1] + (2 * xyz[1] - 1) * quater_length_;
      leaves_[leafnum]->voxel_center_[2] = voxel_center_[2] + (2 * xyz[2] - 1) * quater_length_;
      leaves_[leafnum]->quater_length_ = quater_length_ / 2;
    }
    leaves_[leafnum]->temp_points_.push_back(temp_points_[i]);
    leaves_[leafnum]->new_points_++;
  }
  for (uint i = 0; i < 8; i++)
  {
    if (leaves_[i] != nullptr)
    {
      if (leaves_[i]->temp_points_.size() > leaves_[i]->points_size_threshold_)
      {
        init_plane(leaves_[i]->temp_points_, leaves_[i]->plane_ptr_);
        if (leaves_[i]->plane_ptr_->is_plane_)
        {
          leaves_[i]->octo_state_ = 0;
          // new added
          if (leaves_[i]->temp_points_.size() > leaves_[i]->max_points_num_)
          {
            leaves_[i]->update_enable_ = false;
            std::vector<pointWithVar>().swap(leaves_[i]->temp_points_);
            new_points_ = 0;
          }
        }
        else
        {
          leaves_[i]->octo_state_ = 1;
          leaves_[i]->cut_octo_tree();
        }
        leaves_[i]->init_octo_ = true;
        leaves_[i]->new_points_ = 0;
      }
    }
  }
}

/**
 * @brief 更新八叉树节点，将新的点加入到体素地图中（递归地插入到对应的子节点或更新当前节点的平面信息）
 *        这是 LIO 每帧结束后对体素地图进行增量更新的核心函数
 * 
 * @param pv 输入的点结构体（带方差），用于更新体素地图
 */
void VoxelOctoTree::UpdateOctoTree(const pointWithVar &pv)
{
  if (!init_octo_)  // 情形①：节点还没初始化过
  {
    new_points_++;
    temp_points_.push_back(pv);
    // 待积累足够的点数后再初始化八叉树节点(计算平面参数等等)
    if (temp_points_.size() > points_size_threshold_) { init_octo_tree(); }
  }
  else  // 节点已经初始化过
  {
    if (plane_ptr_->is_plane_) // 情形②：节点已经是一个平面叶子(根体素节点)
    {
      if (update_enable_) // 还未冻结该平面叶子，允许更新
      {
        new_points_++; // 该轮新增的点数
        temp_points_.push_back(pv); // 节点内总点数
        if (new_points_ > update_size_threshold_)
        {
          // 增量式刷新平面估计(利用已积累的所有点)，让平面随新观测更准
          init_plane(temp_points_, plane_ptr_); // 重新拟合平面参数和协方差
          new_points_ = 0; // 增量更新平面参数后，重置新点计数
        }
        if (temp_points_.size() >= max_points_num_)
        {
          update_enable_ = false;
          std::vector<pointWithVar>().swap(temp_points_);
          new_points_ = 0;
        }
      }
    }
    else // 情形③/④：节点已初始化但不是平面（内部节点）
    { // 这种情况下根体素节点已经不是平面，需要将点插入到对应的子节点中
      if (layer_ < max_layer_) // 当前层还未达到最大层，可以继续生成子节点
      {
        // 根据点相对 voxel_center_ 的位置算出所属子象限 leafnum（0~7）
        int xyz[3] = {0, 0, 0};
        if (pv.point_w[0] > voxel_center_[0]) { xyz[0] = 1; }
        if (pv.point_w[1] > voxel_center_[1]) { xyz[1] = 1; }
        if (pv.point_w[2] > voxel_center_[2]) { xyz[2] = 1; }
        int leafnum = 4 * xyz[0] + 2 * xyz[1] + xyz[2];
        if (leaves_[leafnum] != nullptr) { leaves_[leafnum]->UpdateOctoTree(pv); }
        else
        {
          leaves_[leafnum] = new VoxelOctoTree(max_layer_, layer_ + 1, layer_init_num_[layer_ + 1], max_points_num_, planer_threshold_);
          leaves_[leafnum]->layer_init_num_ = layer_init_num_;
          leaves_[leafnum]->voxel_center_[0] = voxel_center_[0] + (2 * xyz[0] - 1) * quater_length_;
          leaves_[leafnum]->voxel_center_[1] = voxel_center_[1] + (2 * xyz[1] - 1) * quater_length_;
          leaves_[leafnum]->voxel_center_[2] = voxel_center_[2] + (2 * xyz[2] - 1) * quater_length_;
          leaves_[leafnum]->quater_length_ = quater_length_ / 2;
          leaves_[leafnum]->UpdateOctoTree(pv);
        }
      }
      else // 当前已经是最大层了
      {
        if (update_enable_) // 还允许更新该最大层的节点
        {
          new_points_++;
          temp_points_.push_back(pv);
          if (new_points_ > update_size_threshold_)
          {
            init_plane(temp_points_, plane_ptr_);
            new_points_ = 0;
          }
          if (temp_points_.size() > max_points_num_)
          {
            update_enable_ = false;
            std::vector<pointWithVar>().swap(temp_points_);
            new_points_ = 0;
          }
        }
      }
    }
  }
}

VoxelOctoTree *VoxelOctoTree::find_correspond(Eigen::Vector3d pw)
{
  if (!init_octo_ || plane_ptr_->is_plane_ || (layer_ >= max_layer_)) return this;

  int xyz[3] = {0, 0, 0};
  xyz[0] = pw[0] > voxel_center_[0] ? 1 : 0;
  xyz[1] = pw[1] > voxel_center_[1] ? 1 : 0;
  xyz[2] = pw[2] > voxel_center_[2] ? 1 : 0;
  int leafnum = 4 * xyz[0] + 2 * xyz[1] + xyz[2];

  // printf("leafnum: %d. \n", leafnum);

  return (leaves_[leafnum] != nullptr) ? leaves_[leafnum]->find_correspond(pw) : this;
}

VoxelOctoTree *VoxelOctoTree::Insert(const pointWithVar &pv)
{
  if ((!init_octo_) || (init_octo_ && plane_ptr_->is_plane_) || (init_octo_ && (!plane_ptr_->is_plane_) && (layer_ >= max_layer_)))
  {
    new_points_++;
    temp_points_.push_back(pv);
    return this;
  }

  if (init_octo_ && (!plane_ptr_->is_plane_) && (layer_ < max_layer_))
  {
    int xyz[3] = {0, 0, 0};
    xyz[0] = pv.point_w[0] > voxel_center_[0] ? 1 : 0;
    xyz[1] = pv.point_w[1] > voxel_center_[1] ? 1 : 0;
    xyz[2] = pv.point_w[2] > voxel_center_[2] ? 1 : 0;
    int leafnum = 4 * xyz[0] + 2 * xyz[1] + xyz[2];
    if (leaves_[leafnum] != nullptr) { return leaves_[leafnum]->Insert(pv); }
    else
    {
      leaves_[leafnum] = new VoxelOctoTree(max_layer_, layer_ + 1, layer_init_num_[layer_ + 1], max_points_num_, planer_threshold_);
      leaves_[leafnum]->layer_init_num_ = layer_init_num_;
      leaves_[leafnum]->voxel_center_[0] = voxel_center_[0] + (2 * xyz[0] - 1) * quater_length_;
      leaves_[leafnum]->voxel_center_[1] = voxel_center_[1] + (2 * xyz[1] - 1) * quater_length_;
      leaves_[leafnum]->voxel_center_[2] = voxel_center_[2] + (2 * xyz[2] - 1) * quater_length_;
      leaves_[leafnum]->quater_length_ = quater_length_ / 2;
      return leaves_[leafnum]->Insert(pv);
    }
  }
  return nullptr;
}

/**
 * @brief 把「IMU 传播得到的预测状态」和「当前帧激光点云与体素地图的点到面残差」做一次 迭代误差状态卡尔曼滤波（ESIKF / iterated EKF），输出修正后的位姿与协方差。
 * 
 * @param state_propagat 由 processImu() 完成 IMU 预积分/传播后的预测状态，作为迭代的「先验锚点」
 */
void VoxelMapManager::StateEstimation(StatesGroup &state_propagat)
{
  cross_mat_list_.clear();
  cross_mat_list_.reserve(feats_down_size_);
  body_cov_list_.clear();
  body_cov_list_.reserve(feats_down_size_);

  // build_residual_time = 0.0;
  // ekf_time = 0.0;
  // double t0 = omp_get_wtime();
  // 1. 预计算点的协方差和反对称矩阵
  for (size_t i = 0; i < feats_down_body_->size(); i++)
  {
    V3D point_this(feats_down_body_->points[i].x, feats_down_body_->points[i].y, feats_down_body_->points[i].z);
    if (point_this[2] == 0) { point_this[2] = 0.001; }
    M3D var;
    // calcBodyCov 把每个激光点建模为高斯分布 p ~ N(p, ∑_lidar)，其中 ∑_lidar 由激光测距误差和光束方向误差决定
    calcBodyCov(point_this, config_setting_.dept_err_, config_setting_.beam_err_, var);
    body_cov_list_.push_back(var);
    point_this = extR_ * point_this + extT_;
    M3D point_crossmat;
    // cross_mat_list_ 保存 IMU 系下点的反对称矩阵, 用于计算雅可比，也可用于后续协方差传播(姿态扰动对点的影响)以及地图更新
    point_crossmat << SKEW_SYM_MATRX(point_this);
    cross_mat_list_.push_back(point_crossmat);
  }

  vector<pointWithVar>().swap(pv_list_);
  pv_list_.resize(feats_down_size_);

  int rematch_num = 0;
  MD(DIM_STATE, DIM_STATE) G, H_T_H, I_STATE;
  G.setZero();
  H_T_H.setZero();
  I_STATE.setIdentity();

  bool flg_EKF_inited, flg_EKF_converged, EKF_stop_flg = 0;
  // 2. 主循环：迭代 ESIKF
  // 每轮迭代都用最新的 state_.rot_end / pos_end 重新投影，这就是「迭代」的意义
  for (int iterCount = 0; iterCount < config_setting_.max_iterations_; iterCount++)
  {
    double total_residual = 0.0;
    // 2.1 重投影 + 点协方差传播
    // 将当前帧激光点云从机体坐标系转换到世界坐标系
    pcl::PointCloud<pcl::PointXYZI>::Ptr world_lidar(new pcl::PointCloud<pcl::PointXYZI>);
    TransformLidar(state_.rot_end, state_.pos_end, feats_down_body_, world_lidar);
    M3D rot_var = state_.cov.block<3, 3>(0, 0);
    M3D t_var = state_.cov.block<3, 3>(3, 3);
    // 把点的不确定性完整传播到世界系
    for (size_t i = 0; i < feats_down_body_->size(); i++)
    {
      pointWithVar &pv = pv_list_[i];
      pv.point_b << feats_down_body_->points[i].x, feats_down_body_->points[i].y, feats_down_body_->points[i].z;
      pv.point_w << world_lidar->points[i].x, world_lidar->points[i].y, world_lidar->points[i].z;

      M3D cov = body_cov_list_[i];
      M3D point_crossmat = cross_mat_list_[i];
      // 计算世界系下点的协方差，考虑了机体系点的协方差、姿态扰动对点的影响以及平移扰动的影响
      // ∑p_w = R * ∑_lidar * R^T + (-[p]_x) * ∑_rot * (-[p]_x)^T + ∑_t
      // 第一项: 机体测量的不确定性通过旋转传播到世界系
      // 第二项: 姿态扰动对点的影响
      // 第三项: 平移扰动的影响
      cov = state_.rot_end * cov * state_.rot_end.transpose() + (-point_crossmat) * rot_var * (-point_crossmat.transpose()) + t_var;
      pv.var = cov;
      pv.body_var = body_cov_list_[i];  // 保留机体系原始协方差，用于后面残差的量测噪声
    }
    ptpl_list_.clear();

    // double t1 = omp_get_wtime();
    // 2.2 建立点到面残差
    BuildResidualListOMP(pv_list_, ptpl_list_);

    // build_residual_time += omp_get_wtime() - t1;
    // 输出的 ptpl_list_ 是有效点到面约束
    for (int i = 0; i < ptpl_list_.size(); i++)
    {
      total_residual += fabs(ptpl_list_[i].dis_to_plane_);
    }
    effct_feat_num_ = ptpl_list_.size();
    cout << "[ LIO ] Raw feature num: " << feats_undistort_->size() << ", downsampled feature num:" << feats_down_size_ 
         << " effective feature num: " << effct_feat_num_ << " average residual: " << total_residual / effct_feat_num_ << endl;
    // 2.3 构建量测雅可比与残差
    /*** Computation of Measuremnt Jacobian matrix H and measurents covarience
     * ***/
    // 为什么这里雅可比只有6列？
    // 这是因为点到面的残差只依赖位姿(R, t): 
    // 即 r = n^T * (R * p_imu + t - c)
    // 速度、零偏、重力、曝光时间都不直接出现。他们只能通过协方差P的先验项间接被更新
    // 这正式LIO的工作方式: IMU传播负责把信息再这些维度之间建立相关性
    MatrixXd Hsub(effct_feat_num_, 6);          // N×6： 每条残差对 (δθ, δt) 的雅可比
    MatrixXd Hsub_T_R_inv(6, effct_feat_num_);  // 6×N： Hᵀ R⁻¹（R 为对角，故存成向量乘）
    VectorXd R_inv(effct_feat_num_);            // N： 每条残差的 1/σ²
    VectorXd meas_vec(effct_feat_num_);         // N： 新息 z − h(x)
    meas_vec.setZero();
    // 逐条残差构建
    for (int i = 0; i < effct_feat_num_; i++)
    {
      auto &ptpl = ptpl_list_[i];
      V3D point_this(ptpl.point_b_);
      point_this = extR_ * point_this + extT_; // LiDAR 系 → IMU/body 系
      V3D point_body(ptpl.point_b_);
      M3D point_crossmat;
      point_crossmat << SKEW_SYM_MATRX(point_this); // [p_imu]_×

      /*** get the normal vector of closest surface/corner ***/
      // 注意这里用的是state_propagat
      // 用先验状态算残差方差
      V3D point_world = state_propagat.rot_end * point_this + state_propagat.pos_end;
      Eigen::Matrix<double, 1, 6> J_nq;
      J_nq.block<1, 3>(0, 0) = point_world - ptpl_list_[i].center_; // ∂dis/∂n
      J_nq.block<1, 3>(0, 3) = -ptpl_list_[i].normal_;  // ∂dis/∂c

      M3D var;
      // V3D normal_b = state_.rot_end.inverse() * ptpl_list_[i].normal_;
      // V3D point_b = ptpl_list_[i].point_b_;
      // double cos_theta = fabs(normal_b.dot(point_b) / point_b.norm());
      // ptpl_list_[i].body_cov_ = ptpl_list_[i].body_cov_ * (1.0 / cos_theta) * (1.0 / cos_theta);

      // point_w cov
      // var = state_propagat.rot_end * extR_ * ptpl_list_[i].body_cov_ * (state_propagat.rot_end * extR_).transpose() +
      //       state_propagat.cov.block<3, 3>(3, 3) + (-point_crossmat) * state_propagat.cov.block<3, 3>(0, 0) * (-point_crossmat).transpose();

      // point_w cov (another_version)
      // var = state_propagat.rot_end * extR_ * ptpl_list_[i].body_cov_ * (state_propagat.rot_end * extR_).transpose() +
      //       state_propagat.cov.block<3, 3>(3, 3) - point_crossmat * state_propagat.cov.block<3, 3>(0, 0) * point_crossmat;

      // point_body cov
      // 这里var只用了body_cov_(传感器噪声)经R_wbR_bl旋转到世界系，没有叠加state_propagat.cov 的姿态/位置项（那些行被注释掉了）。
      // 这是刻意的：状态不确定度已经通过P进入了卡尔曼增益，若再加进R就会重复计算，导致滤波不合理。
      // 注意: 这里用的是 state_propagat（IMU 先验），而下面算 A 用的是 state_（当前迭代值）。
      // 两处评价点不同，属实现上的小不一致
      var = state_propagat.rot_end * extR_ * ptpl_list_[i].body_cov_ * (state_propagat.rot_end * extR_).transpose();

      double sigma_l = J_nq * ptpl_list_[i].plane_var_ * J_nq.transpose();
      // 测量方差R 由三部分组成
      // σ_i^2 = 0.001 + J_nq * Σ_π * J_nq^T + n^T * Σ_p * n
      // 其中：
      // 1. 0.001 是一个固定的小量(下限)，对应σ_min ≈ 3.2 cm，防止除零，同时给每条残差的最大权重设了上限
      // 2. J_nq * Σ_π * J_nq^T 对应平面参数的不确定性贡献
      // 3. n^T * Σ_p * n 对应点位置不确定性贡献，即点测量噪声投影到法向量方向上的方差
      R_inv(i) = 1.0 / (0.001 + sigma_l + ptpl_list_[i].normal_.transpose() * var * ptpl_list_[i].normal_);
      // R_inv(i) = 1.0 / (sigma_l + ptpl_list_[i].normal_.transpose() * var * ptpl_list_[i].normal_);

      /*** calculate the Measuremnt Jacobian matrix H ***/
      // 测量雅可比矩阵 H 的计算
      // 由R‘ = R Exp(δθ^) 可得
      // R' p_imu = R p_imu - R [p_imu]x δθ
      // ==> ∂dis / ∂(δθ) =  [p_imu]x R^T n = A , ∂dis / ∂(δt) = n
      // Hi = [A^T, n^T] ∈ R^(1x6)
      // ------ 
      // 测量模型取 zi=0（点应落在平面上），故新息: 
      // zi - hi(x) = 0 - dis = meas_vec(i) = - dis_to_plane_
      // 符号自洽性检查：若 dis>0（点在 +n 侧），则 meas_vec <0，更新量 ∝H^⊤(z−h)=−dis[A;n]，
      // 代入后 Δdis∝−dis∥H∥^2<0，即点被拉回平面。✓ 是下降方向。
      // Hsub_T_R_inv 就是把H^T R^-1按列存下来(R对角，所以逐元素乘)
      V3D A(point_crossmat * state_.rot_end.transpose() * ptpl_list_[i].normal_);
      Hsub.row(i) << VEC_FROM_ARRAY(A), ptpl_list_[i].normal_[0], ptpl_list_[i].normal_[1], ptpl_list_[i].normal_[2];
      Hsub_T_R_inv.col(i) << A[0] * R_inv(i), A[1] * R_inv(i), A[2] * R_inv(i), ptpl_list_[i].normal_[0] * R_inv(i),
          ptpl_list_[i].normal_[1] * R_inv(i), ptpl_list_[i].normal_[2] * R_inv(i);
      meas_vec(i) = -ptpl_list_[i].dis_to_plane_;
    }
    // 2.4 迭代卡尔曼更新
    EKF_stop_flg = false;
    flg_EKF_converged = false;
    /*** Iterative Kalman Filter Update ***/
    // MAP + Gauss–Newton
    // 通过迭代卡尔曼滤波更新状态向量，使得测量残差最小化，即在高斯-牛顿框架下求解最大后验估计(MAP)
    MatrixXd K(DIM_STATE, effct_feat_num_);
    // auto &&Hsub_T = Hsub.transpose();
    auto &&HTz = Hsub_T_R_inv * meas_vec;
    // fout_dbg<<"HTz: "<<HTz<<endl;
    H_T_H.block<6, 6>(0, 0) = Hsub_T_R_inv * Hsub;
    // EigenSolver<Matrix<double, 6, 6>> es(H_T_H.block<6,6>(0,0));
    MD(DIM_STATE, DIM_STATE) &&K_1 = (H_T_H.block<DIM_STATE, DIM_STATE>(0, 0) + state_.cov.block<DIM_STATE, DIM_STATE>(0, 0).inverse()).inverse();
    G.block<DIM_STATE, 6>(0, 0) = K_1.block<DIM_STATE, 6>(0, 0) * H_T_H.block<6, 6>(0, 0);
    auto vec = state_propagat - state_;
    VD(DIM_STATE)
    solution = K_1.block<DIM_STATE, 6>(0, 0) * HTz + vec.block<DIM_STATE, 1>(0, 0) - G.block<DIM_STATE, 6>(0, 0) * vec.block<6, 1>(0, 0);
    int minRow, minCol;
    state_ += solution;
    auto rot_add = solution.block<3, 1>(0, 0);
    auto t_add = solution.block<3, 1>(3, 0);
    // 2.5 收敛判断 / Rematch
    if ((rot_add.norm() * 57.3 < 0.01) && (t_add.norm() * 100 < 0.015)) { flg_EKF_converged = true; }
    V3D euler_cur = state_.rot_end.eulerAngles(2, 1, 0);

    /*** Rematch Judgement ***/

    if (flg_EKF_converged || ((rematch_num == 0) && (iterCount == (config_setting_.max_iterations_ - 2)))) { rematch_num++; }

    /*** Convergence Judgements and Covariance Update ***/
    // 2.6 协方差更新与收尾
    // 触发条件: 连续两次收敛 或 最后一轮
    if (!EKF_stop_flg && (rematch_num >= 2 || (iterCount == config_setting_.max_iterations_ - 1)))
    {
      /*** Covariance Update ***/
      // _state.cov = (I_STATE - G) * _state.cov;
      // G只有前6列有非零值，对应状态向量的旋转和平移部分，(I-G)P实际上只改动P的前六列(其余列保持原值)
      state_.cov.block<DIM_STATE, DIM_STATE>(0, 0) =
          (I_STATE.block<DIM_STATE, DIM_STATE>(0, 0) - G.block<DIM_STATE, DIM_STATE>(0, 0)) * state_.cov.block<DIM_STATE, DIM_STATE>(0, 0);
      // total_distance += (_state.pos_end - position_last).norm();
      position_last_ = state_.pos_end;
      geoQuat_ = tf::createQuaternionMsgFromRollPitchYaw(euler_cur(0), euler_cur(1), euler_cur(2));

      // VD(DIM_STATE) K_sum  = K.rowwise().sum();
      // VD(DIM_STATE) P_diag = _state.cov.diagonal();
      EKF_stop_flg = true;
    }
    if (EKF_stop_flg) break;
  }

  // double t2 = omp_get_wtime();
  // scan_count++;
  // ekf_time = t2 - t0 - build_residual_time;

  // ave_build_residual_time = ave_build_residual_time * (scan_count - 1) / scan_count + build_residual_time / scan_count;
  // ave_ekf_time = ave_ekf_time * (scan_count - 1) / scan_count + ekf_time / scan_count;

  // cout << "[ Mapping ] ekf_time: " << ekf_time << "s, build_residual_time: " << build_residual_time << "s" << endl;
  // cout << "[ Mapping ] ave_ekf_time: " << ave_ekf_time << "s, ave_build_residual_time: " << ave_build_residual_time << "s" << endl;
}

void VoxelMapManager::TransformLidar(const Eigen::Matrix3d rot, const Eigen::Vector3d t, const PointCloudXYZI::Ptr &input_cloud,
                                     pcl::PointCloud<pcl::PointXYZI>::Ptr &trans_cloud)
{
  pcl::PointCloud<pcl::PointXYZI>().swap(*trans_cloud);
  trans_cloud->reserve(input_cloud->size());
  for (size_t i = 0; i < input_cloud->size(); i++)
  {
    pcl::PointXYZINormal p_c = input_cloud->points[i];
    Eigen::Vector3d p(p_c.x, p_c.y, p_c.z);
    p = (rot * (extR_ * p + extT_) + t);
    pcl::PointXYZI pi;
    pi.x = p(0);
    pi.y = p(1);
    pi.z = p(2);
    pi.intensity = p_c.intensity;
    trans_cloud->points.push_back(pi);
  }
}

/** 
 * @brief  把一帧降采样后的世界系点云写入体素地图
 * BuildVoxelMap() 自己并不真正完成复杂的平面拟合和 octree 递归，
 * 一句话概括就是: 
 * 把当前第一批已经变换到世界坐标系的降采样 LiDAR 点，附上每个点的不确定性，
 * 然后按大 voxel 做哈希分桶；每个 voxel 创建一个 VoxelOctoTree 根节点，
 * 最后统一调用 init_octo_tree()，把每个 voxel 递归判断为“平面”或者继续八叉树细分。
 */
void VoxelMapManager::BuildVoxelMap()
{
  // 读取voxel 参数
  float voxel_size = config_setting_.max_voxel_size_; // 根体素尺寸大小，即最外层 hash voxel 的边长
  float planer_threshold = config_setting_.planner_threshold_;
  int max_layer = config_setting_.max_layer_;
  int max_points_num = config_setting_.max_points_num_;
  std::vector<int> layer_init_num = config_setting_.layer_init_num_;
  // 为每帧点构造带协方差的数据pointWithVar
  std::vector<pointWithVar> input_points;

  for (size_t i = 0; i < feats_down_world_->size(); i++)
  {
    pointWithVar pv;
    pv.point_w << feats_down_world_->points[i].x, feats_down_world_->points[i].y, feats_down_world_->points[i].z;
    V3D point_this(feats_down_body_->points[i].x, feats_down_body_->points[i].y, feats_down_body_->points[i].z);
    M3D var;
    // 从激光点的原始测量开始构造测量协方差矩阵
    // 它从地图初始化阶段开始，就不是把点当“确定的位置”，而是把点当一个带 uncertainty 的空间观测
    calcBodyCov(point_this, config_setting_.dept_err_, config_setting_.beam_err_, var);
    M3D point_crossmat;
    point_crossmat << SKEW_SYM_MATRX(point_this);
    // 把点 covariance 转到世界系
    // 转换到世界系时额外叠加了两项：姿态不确定度导致的误差（用叉乘矩阵 (-point_crossmat) 乘旋转协方差）
    // 以及位置不确定度（state_.cov.block<3,3>(3,3)）。
    // 这样存进地图的每个点都自带统计意义上的噪声信息，后面拟合平面计算 plane_var_ 时就会用到。
    // 它实际上是把几类不确定性叠加起来: 
    // ∑p_w = R*∑lidar*R^T + J_R*∑R*J_R^T + ∑t
    // 其中:
    // R 为世界系到机体系的旋转矩阵, ∑lidar 为激光点在机体系下的测量协方差
    // J_R 为旋转对点的雅可比矩阵 (-point_crossmat), ∑rot 为姿态协方差, ∑t 为位置（平移）协方差
    // 如何推导？
    // 因为p_w = R * p_b + t，所以对p_b的协方差进行旋转变换，并叠加姿态和位置的不确定性
    // 姿态有一个很小的扰动 Rexp(δθ^)，其中 δθ ~ N(0, ∑rot)，δθ^ 为 δθ 的反对称矩阵形式
    // 对point 的影响大致 δp ≈ -[p]x * δθ
    // 所以 Jacobian   J_R = -[p]x  于是 ∑p_R = [p]x * ∑rot * [p]x^T
    var = (state_.rot_end * extR_) * var * (state_.rot_end * extR_).transpose() +
          (-point_crossmat) * state_.cov.block<3, 3>(0, 0) * (-point_crossmat).transpose() + state_.cov.block<3, 3>(3, 3);
    pv.var = var;
    input_points.push_back(pv);
  }
  // 逐点哈希定位根体素并插入
  uint plsize = input_points.size();
  for (uint i = 0; i < plsize; i++)
  {
    const pointWithVar p_v = input_points[i];
    float loc_xyz[3];
    for (int j = 0; j < 3; j++)
    {
      loc_xyz[j] = p_v.point_w[j] / voxel_size;
      // 负坐标向负无穷取整，避免 -0.5 -> 0 的边界错位
      // 为什么负数要专门减 1？ 其实是为了近似实现向下取整 floor(x/s)，防止点恰好落在体素边界时被分错格子
      // 因为在 C++ 中，整数类型的强制转换会向零取整，而我们希望负坐标向负无穷取整。
      // 例如 -0.5 / voxel_size 会得到 -0.5，强制转换为 int64_t 会得到 0，这会导致边界错位。
      // 所以对于负数，需要减 1，使其向负无穷取整。
      if (loc_xyz[j] < 0) { loc_xyz[j] -= 1.0; }
    }
    VOXEL_LOCATION position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1], (int64_t)loc_xyz[2]);
    auto iter = voxel_map_.find(position);
    if (iter != voxel_map_.end()) // 如果根体素已经存在于地图中, 则将新的点加入该体素的临时点云中
    {
      voxel_map_[position]->temp_points_.push_back(p_v);
      voxel_map_[position]->new_points_++;
    }
    else // 根体素不存在，则新建八叉树根节点 VoxelOctoTree（第 0 层）
    {
      VoxelOctoTree *octo_tree = new VoxelOctoTree(max_layer, 0, layer_init_num[0], max_points_num, planer_threshold);
      voxel_map_[position] = octo_tree;
      voxel_map_[position]->quater_length_ = voxel_size / 4; // 八叉树节点的四分之一边长(八叉树子节点步长)
      voxel_map_[position]->voxel_center_[0] = (0.5 + position.x) * voxel_size; // 根体素中心坐标
      voxel_map_[position]->voxel_center_[1] = (0.5 + position.y) * voxel_size; // 根体素中心坐标
      voxel_map_[position]->voxel_center_[2] = (0.5 + position.z) * voxel_size; // 根体素中心坐标
      voxel_map_[position]->temp_points_.push_back(p_v);
      voxel_map_[position]->new_points_++;
      voxel_map_[position]->layer_init_num_ = layer_init_num; // 让子节点能继承各层初始化阈值
    }
  }
  // 遍历整个地图，触发树的构建/划分
  for (auto iter = voxel_map_.begin(); iter != voxel_map_.end(); ++iter)
  {
    iter->second->init_octo_tree();
  }
}

V3F VoxelMapManager::RGBFromVoxel(const V3D &input_point)
{
  int64_t loc_xyz[3];
  for (int j = 0; j < 3; j++)
  {
    loc_xyz[j] = floor(input_point[j] / config_setting_.max_voxel_size_);
  }

  VOXEL_LOCATION position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1], (int64_t)loc_xyz[2]);
  int64_t ind = loc_xyz[0] + loc_xyz[1] + loc_xyz[2];
  uint k((ind + 100000) % 3);
  V3F RGB((k == 0) * 255.0, (k == 1) * 255.0, (k == 2) * 255.0);
  // cout<<"RGB: "<<RGB.transpose()<<endl;
  return RGB;
}

/**
 * @brief 更新体素地图，将新的点云信息加入对应的八叉树中
 * 把一批新的世界系点（带方差）逐点插入/更新到已有的体素地图中，是 LIO 每帧结束后"增量维护地图"的入口函数
 * @param input_points 
 */
void VoxelMapManager::UpdateVoxelMap(const std::vector<pointWithVar> &input_points)
{
  float voxel_size = config_setting_.max_voxel_size_;
  float planer_threshold = config_setting_.planner_threshold_;
  int max_layer = config_setting_.max_layer_;
  int max_points_num = config_setting_.max_points_num_;
  std::vector<int> layer_init_num = config_setting_.layer_init_num_;
  uint plsize = input_points.size();
  for (uint i = 0; i < plsize; i++)
  {
    const pointWithVar p_v = input_points[i];
    // 1. 根据点的世界坐标计算它所在的根体素(voxel)哈希位置
    //    对负坐标做 -1 修正，保证 floor 语义正确
    float loc_xyz[3];
    for (int j = 0; j < 3; j++)
    {
      loc_xyz[j] = p_v.point_w[j] / voxel_size;
      if (loc_xyz[j] < 0) { loc_xyz[j] -= 1.0; }
    }
    VOXEL_LOCATION position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1], (int64_t)loc_xyz[2]);
    auto iter = voxel_map_.find(position);
    // 如果该根体素已经存在于地图中，则将点交给已有八叉树递归更新
    if (iter != voxel_map_.end()) { voxel_map_[position]->UpdateOctoTree(p_v); }
    else  // 不存在则新建一个根节点 VoxelOctoTree(第0层)，设置体素中心/边长等
    {     // 然后同样调用 UpdateOctoTree(p_v) 把这个点插进去
      VoxelOctoTree *octo_tree = new VoxelOctoTree(max_layer, 0, layer_init_num[0], max_points_num, planer_threshold);
      voxel_map_[position] = octo_tree;
      voxel_map_[position]->layer_init_num_ = layer_init_num;
      voxel_map_[position]->quater_length_ = voxel_size / 4;
      voxel_map_[position]->voxel_center_[0] = (0.5 + position.x) * voxel_size;
      voxel_map_[position]->voxel_center_[1] = (0.5 + position.y) * voxel_size;
      voxel_map_[position]->voxel_center_[2] = (0.5 + position.z) * voxel_size;
      voxel_map_[position]->UpdateOctoTree(p_v);
    }
  }
}

/**
 * @brief 建立点到面残差
 * 对当前帧每个降采样后的世界系点，在体素地图里找到它所属的那个平面，算出一条点到面残差 PointToPlane，
 * 把所有成功匹配的残差收集起来交给 EKF。整个过程用 OpenMP 并行。
 * @param pv_list 当前帧降采样点，已变换到世界系
 * @param ptpl_list 点到面残差列表，函数会将所有成功匹配的残差存入该列表
 */
void VoxelMapManager::BuildResidualListOMP(std::vector<pointWithVar> &pv_list, std::vector<PointToPlane> &ptpl_list)
{
  int max_layer = config_setting_.max_layer_;
  double voxel_size = config_setting_.max_voxel_size_;
  double sigma_num = config_setting_.sigma_num_;
  std::mutex mylock;
  ptpl_list.clear();
  std::vector<PointToPlane> all_ptpl_list(pv_list.size());
  std::vector<bool> useful_ptpl(pv_list.size());
  std::vector<size_t> index(pv_list.size());
  for (size_t i = 0; i < index.size(); ++i)
  {
    index[i] = i;
    useful_ptpl[i] = false;
  }
  #ifdef MP_EN
    omp_set_num_threads(MP_PROC_NUM);
    #pragma omp parallel for
  #endif
  for (int i = 0; i < index.size(); i++) // 按点并行 + 邻域回退
  {
    pointWithVar &pv = pv_list[i];
    float loc_xyz[3];
    for (int j = 0; j < 3; j++)
    {
      loc_xyz[j] = pv.point_w[j] / voxel_size;
      if (loc_xyz[j] < 0) { loc_xyz[j] -= 1.0; } // 负数向下取整
    }
    // 和 BuildVoxelMap 里完全一致的哈希定位
    VOXEL_LOCATION position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1], (int64_t)loc_xyz[2]);
    auto iter = voxel_map_.find(position);
    if (iter != voxel_map_.end())
    {
      VoxelOctoTree *current_octo = iter->second; // 根体素节点
      PointToPlane single_ptpl;
      bool is_sucess = false;
      double prob = 0;
      // 第一次尝试：在自己所在的根体素里找平面
      // 单点、单树、递归匹配
      build_single_residual(pv, current_octo, 0, is_sucess, prob, single_ptpl);
      if (!is_sucess) // 失败回退：尝试相邻根体素
      {
        VOXEL_LOCATION near_position = position;
        if (loc_xyz[0] > (current_octo->voxel_center_[0] + current_octo->quater_length_)) { near_position.x = near_position.x + 1; }
        else if (loc_xyz[0] < (current_octo->voxel_center_[0] - current_octo->quater_length_)) { near_position.x = near_position.x - 1; }
        if (loc_xyz[1] > (current_octo->voxel_center_[1] + current_octo->quater_length_)) { near_position.y = near_position.y + 1; }
        else if (loc_xyz[1] < (current_octo->voxel_center_[1] - current_octo->quater_length_)) { near_position.y = near_position.y - 1; }
        if (loc_xyz[2] > (current_octo->voxel_center_[2] + current_octo->quater_length_)) { near_position.z = near_position.z + 1; }
        else if (loc_xyz[2] < (current_octo->voxel_center_[2] - current_octo->quater_length_)) { near_position.z = near_position.z - 1; }
        auto iter_near = voxel_map_.find(near_position);
        if (iter_near != voxel_map_.end()) { build_single_residual(pv, iter_near->second, 0, is_sucess, prob, single_ptpl); }
      }
      if (is_sucess)
      {
        mylock.lock(); // 上锁，保护共享资源
        useful_ptpl[i] = true;
        all_ptpl_list[i] = single_ptpl;
        mylock.unlock();
      }
      else
      {
        mylock.lock();
        useful_ptpl[i] = false;
        mylock.unlock();
      }
    }
  }
  // 串行压缩: 将所有有效的点到平面残差压缩到最终的列表中
  for (size_t i = 0; i < useful_ptpl.size(); i++)
  {
    if (useful_ptpl[i]) { ptpl_list.push_back(all_ptpl_list[i]); }
  }
}

/**
 * @brief 构建单个点到平面的残差，用于计算点到平面的距离并判断是否符合平面约束
 * 给定一个世界系点，沿八叉树递归下探；一旦遇到"平面"节点，就用几何门限 + 概率门限判断这个点能不能落在该平面上，
 * 并算出残差。整棵树走完后，保留概率最高的那个匹配。
 * @param pv 待匹配的点及其协方差
 * @param current_octo 当前八叉树节点
 * @param current_layer 当前八叉树层数
 * @param is_sucess 是否成功匹配到平面
 * @param prob 最高匹配的概率
 * @param single_ptpl 单个点到平面的残差结构体(最优匹配的残差)
 */
void VoxelMapManager::build_single_residual(pointWithVar &pv, const VoxelOctoTree *current_octo, const int current_layer, bool &is_sucess,
                                            double &prob, PointToPlane &single_ptpl)
{
  int max_layer = config_setting_.max_layer_;
  double sigma_num = config_setting_.sigma_num_;

  double radius_k = 3;
  Eigen::Vector3d p_w = pv.point_w;
  // ── 情形 A：当前节点是平面 ──
  //    A1. 面内距离门限
  //    A2. 不确定度(σ)门限
  //    A3. 概率打分，择优替换 single_ptpl
  ///  三重门限：面内距离门限 + 不确定度(σ)门限 + 概率打分
  // 1. 面内距离门限: 点投影要在平面片范围内
  // 2. 不确定度(σ)门限: 残差要在联合不确定度的 3σ 内
  // 3. 概率打分: 全树取概率最大者
  if (current_octo->plane_ptr_->is_plane_)
  {
    VoxelPlane &plane = *current_octo->plane_ptr_;
    Eigen::Vector3d p_world_to_center = p_w - plane.center_;
    // 点到平面的“法向”距离
    float dis_to_plane = fabs(plane.normal_(0) * p_w(0) + plane.normal_(1) * p_w(1) + plane.normal_(2) * p_w(2) + plane.d_);
    // 点到平面中心的距离平方
    float dis_to_center = (plane.center_(0) - p_w(0)) * (plane.center_(0) - p_w(0)) + (plane.center_(1) - p_w(1)) * (plane.center_(1) - p_w(1)) +
                          (plane.center_(2) - p_w(2)) * (plane.center_(2) - p_w(2));
    // 面内（切向）距离
    float range_dis = sqrt(dis_to_center - dis_to_plane * dis_to_plane);
    // 几何上是勾股定理
    // |pc - c|^2 = dis_to_plane^2 + range_dis^2  法向分量  与 切向分量
    /*
                 n
                 ↑c (center of the plane)
      ─ ─ ─ ─ ─ ─┼─ ─ ─ ─ ─ ─  plane
                 |\
   dis_to_plane  | \
                 |  \  |p_w - c|
                 |   \
      ─ ─ ─ ─ ─ ─+----*  p_w
                 |range_dis
    含义：即使点在法向上贴着平面，如果它的投影点跑出了这个平面片很远（比如跑到了墙外），也不该把它当成约束。
    radius_k = 3 是宽容系数（因为 radius_ = sqrt(λ_max) 只是统计尺度，偏小，见前面 radius_ 的分析）
    若 range_dis 超限 → 直接 return，不进入子节点（平面节点本来也没有子节点）
*/
    if (range_dis <= radius_k * plane.radius_)
    {
      //  不确定度（σ）门限
      // ① 平面参数不确定性
      // 点到面距离 dis = n^T (p_w - c),对平面参数[n;c] 的一阶泰勒展开
      // δdis = ∂dis/∂n * δn + ∂dis/∂c * δc
      //      = [p_w - c, -n] * [δn; δc] = J_nq * [δn; δc]
      // ====> Var1 = J_nq * ∑_π * J_nq^T 
      // ② 点自身不确定性
      // 点的世界系协方差 ∑_p 投影到法向上，得到点到平面的距离不确定性 Var2 = n^T * ∑_p * n
      // ③ 合成
      // σ_l = Var1 + Var2 = J_nq * ∑_π * J_nq^T + n^T * ∑_p * n
      // ===> 检验统计量 = |dis| / sqrt(σ_l) < σ_num
      // 即：残差必须落在 σ_num倍标准差以内（默认 3σ）
      // 这就把“拟合得好的平面”(∑_π 小) 和 “点观测精度高”(∑_p 小) 自动纳入门限，是FAST-LIVO2相比普通LOAM的关键优势
      Eigen::Matrix<double, 1, 6> J_nq;
      J_nq.block<1, 3>(0, 0) = p_w - plane.center_; // ∂dis/∂n
      J_nq.block<1, 3>(0, 3) = -plane.normal_;      // ∂dis/∂c
      double sigma_l = J_nq * plane.plane_var_ * J_nq.transpose();   // ① 平面参数不确定性的贡献
      sigma_l += plane.normal_.transpose() * pv.var * plane.normal_; // ② 点自身不确定性的贡献
      if (dis_to_plane < sigma_num * sqrt(sigma_l)) // sigma_num 默认 3
      {
        is_sucess = true;
        //  概率打分与择优
        // 打分公式就是零均值一维高斯 PDF(省去了1/sqrt(2*pi)常数，只用于比较大小): 
        // prob = 1/ sqrt(sigma_l) * exp(-dis_to_plane^2 / 2*sigma_l)
        // dis越小，prob越大(贴合越好)
        // sigma_l 越小，峰值1/ sqrt(sigma_l) 越高，衰减越快(越"自信")
        double this_prob = 1.0 / (sqrt(sigma_l)) * exp(-0.5 * dis_to_plane * dis_to_plane / sigma_l);
        if (this_prob > prob) // 不是"首次命中即返回"
        { // 因为情形 B 里父节点会遍历全部 8 个子节点（不提前 break），
          // 所以整棵树跑完后 single_ptpl 是全树概率最大的那个平面，prob 是它的分数。
          // 这是一个"八叉树内最大似然选择"。
          prob = this_prob;
          pv.normal = plane.normal_;
          single_ptpl.body_cov_ = pv.body_var;
          single_ptpl.point_b_ = pv.point_b;
          single_ptpl.point_w_ = pv.point_w;
          single_ptpl.plane_var_ = plane.plane_var_;
          single_ptpl.normal_ = plane.normal_;
          single_ptpl.center_ = plane.center_;
          single_ptpl.d_ = plane.d_;
          single_ptpl.layer_ = current_layer;
          single_ptpl.dis_to_plane_ = plane.normal_(0) * p_w(0) + plane.normal_(1) * p_w(1) + plane.normal_(2) * p_w(2) + plane.d_;
          // dis_to_plane_ ← 带符号，用于 EKF
        }
        return;
      }
      else
      {
        // is_sucess = false;
        return;
      }
    }
    else
    {
      // is_sucess = false;
      return;
    }
  }
  else // ── 情形 B：当前节点不是平面 ──
  {
    // 非平面且未到最大层 → 遍历 8 个非空子节点递归，把累加器原样传下去。
    if (current_layer < max_layer)
    { // 没有提前退出，因此是一棵完整的树遍历（代价上界 8^depth，但 max_layer_ 默认 1，实际最多访问 8 个节点）
      for (size_t leafnum = 0; leafnum < 8; leafnum++)
      { // 未初始化的节点（init_octo_ == false）其 is_plane_ 默认为 false，会走进这个分支，
        // 但 leaves_ 全为 nullptr，等价于"空转返回"
        if (current_octo->leaves_[leafnum] != nullptr)
        {

          VoxelOctoTree *leaf_octo = current_octo->leaves_[leafnum];
          build_single_residual(pv, leaf_octo, current_layer + 1, is_sucess, prob, single_ptpl);
        }
      }
      return;
    }
    else { return; } // 到最大层还不是平面 → 直接返回，该点在这棵树上匹配失败，放弃
  }
}

void VoxelMapManager::pubVoxelMap()
{
  double max_trace = 0.25;
  double pow_num = 0.2;
  ros::Rate loop(500);
  float use_alpha = 0.8;
  visualization_msgs::MarkerArray voxel_plane;
  voxel_plane.markers.reserve(1000000);
  std::vector<VoxelPlane> pub_plane_list;
  for (auto iter = voxel_map_.begin(); iter != voxel_map_.end(); iter++)
  {
    GetUpdatePlane(iter->second, config_setting_.max_layer_, pub_plane_list);
  }
  for (size_t i = 0; i < pub_plane_list.size(); i++)
  {
    V3D plane_cov = pub_plane_list[i].plane_var_.block<3, 3>(0, 0).diagonal();
    double trace = plane_cov.sum();
    if (trace >= max_trace) { trace = max_trace; }
    trace = trace * (1.0 / max_trace);
    trace = pow(trace, pow_num);
    uint8_t r, g, b;
    mapJet(trace, 0, 1, r, g, b);
    Eigen::Vector3d plane_rgb(r / 256.0, g / 256.0, b / 256.0);
    double alpha;
    if (pub_plane_list[i].is_plane_) { alpha = use_alpha; }
    else { alpha = 0; }
    pubSinglePlane(voxel_plane, "plane", pub_plane_list[i], alpha, plane_rgb);
  }
  voxel_map_pub_.publish(voxel_plane);
  loop.sleep();
}

void VoxelMapManager::GetUpdatePlane(const VoxelOctoTree *current_octo, const int pub_max_voxel_layer, std::vector<VoxelPlane> &plane_list)
{
  if (current_octo->layer_ > pub_max_voxel_layer) { return; }
  if (current_octo->plane_ptr_->is_update_) { plane_list.push_back(*current_octo->plane_ptr_); }
  if (current_octo->layer_ < current_octo->max_layer_)
  {
    if (!current_octo->plane_ptr_->is_plane_)
    {
      for (size_t i = 0; i < 8; i++)
      {
        if (current_octo->leaves_[i] != nullptr) { GetUpdatePlane(current_octo->leaves_[i], pub_max_voxel_layer, plane_list); }
      }
    }
  }
  return;
}

void VoxelMapManager::pubSinglePlane(visualization_msgs::MarkerArray &plane_pub, const std::string plane_ns, const VoxelPlane &single_plane,
                                     const float alpha, const Eigen::Vector3d rgb)
{
  visualization_msgs::Marker plane;
  plane.header.frame_id = "camera_init";
  plane.header.stamp = ros::Time();
  plane.ns = plane_ns;
  plane.id = single_plane.id_;
  plane.type = visualization_msgs::Marker::CYLINDER;
  plane.action = visualization_msgs::Marker::ADD;
  plane.pose.position.x = single_plane.center_[0];
  plane.pose.position.y = single_plane.center_[1];
  plane.pose.position.z = single_plane.center_[2];
  geometry_msgs::Quaternion q;
  CalcVectQuation(single_plane.x_normal_, single_plane.y_normal_, single_plane.normal_, q);
  plane.pose.orientation = q;
  plane.scale.x = 3 * sqrt(single_plane.max_eigen_value_);
  plane.scale.y = 3 * sqrt(single_plane.mid_eigen_value_);
  plane.scale.z = 2 * sqrt(single_plane.min_eigen_value_);
  plane.color.a = alpha;
  plane.color.r = rgb(0);
  plane.color.g = rgb(1);
  plane.color.b = rgb(2);
  plane.lifetime = ros::Duration();
  plane_pub.markers.push_back(plane);
}

void VoxelMapManager::CalcVectQuation(const Eigen::Vector3d &x_vec, const Eigen::Vector3d &y_vec, const Eigen::Vector3d &z_vec,
                                      geometry_msgs::Quaternion &q)
{
  Eigen::Matrix3d rot;
  rot << x_vec(0), x_vec(1), x_vec(2), y_vec(0), y_vec(1), y_vec(2), z_vec(0), z_vec(1), z_vec(2);
  Eigen::Matrix3d rotation = rot.transpose();
  Eigen::Quaterniond eq(rotation);
  q.w = eq.w();
  q.x = eq.x();
  q.y = eq.y();
  q.z = eq.z();
}

void VoxelMapManager::mapJet(double v, double vmin, double vmax, uint8_t &r, uint8_t &g, uint8_t &b)
{
  r = 255;
  g = 255;
  b = 255;

  if (v < vmin) { v = vmin; }

  if (v > vmax) { v = vmax; }

  double dr, dg, db;

  if (v < 0.1242)
  {
    db = 0.504 + ((1. - 0.504) / 0.1242) * v;
    dg = dr = 0.;
  }
  else if (v < 0.3747)
  {
    db = 1.;
    dr = 0.;
    dg = (v - 0.1242) * (1. / (0.3747 - 0.1242));
  }
  else if (v < 0.6253)
  {
    db = (0.6253 - v) * (1. / (0.6253 - 0.3747));
    dg = 1.;
    dr = (v - 0.3747) * (1. / (0.6253 - 0.3747));
  }
  else if (v < 0.8758)
  {
    db = 0.;
    dr = 1.;
    dg = (0.8758 - v) * (1. / (0.8758 - 0.6253));
  }
  else
  {
    db = 0.;
    dg = 0.;
    dr = 1. - (v - 0.8758) * ((1. - 0.504) / (1. - 0.8758));
  }

  r = (uint8_t)(255 * dr);
  g = (uint8_t)(255 * dg);
  b = (uint8_t)(255 * db);
}

/**
 * @brief 地图滑动操作，清理超出当前视野范围的体素节点，
 * 以机器人当前位置为中心，只保留一个固定范围（half_map_size 个根体素半径）内的地图，
 * 把超出这个范围的旧体素删除，从而控制内存占用和查找/拟合的计算量，避免地图随轨迹增长无限膨胀
 */
void VoxelMapManager::mapSliding()
{
  if((position_last_ - last_slide_position).norm() < config_setting_.sliding_thresh)
  {
    std::cout<<RED<<"[DEBUG]: Last sliding length "<<(position_last_ - last_slide_position).norm()<<RESET<<"\n";
    return;
  }

  //get global id now
  last_slide_position = position_last_;
  double t_sliding_start = omp_get_wtime();
  float loc_xyz[3];
  for (int j = 0; j < 3; j++)
  { // 将机器人当前位置转换为根体素整数坐标
    loc_xyz[j] = position_last_[j] / config_setting_.max_voxel_size_;
    if (loc_xyz[j] < 0) { loc_xyz[j] -= 1.0; }
  }
  // VOXEL_LOCATION position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1], (int64_t)loc_xyz[2]);//discrete global
  // 以当前体素坐标为中心，划出一个 [center - half_map_size, center + half_map_size] 的立方体窗口
  // 窗口外的根体素全部删除
  clearMemOutOfMap((int64_t)loc_xyz[0] + config_setting_.half_map_size, (int64_t)loc_xyz[0] - config_setting_.half_map_size,
                    (int64_t)loc_xyz[1] + config_setting_.half_map_size, (int64_t)loc_xyz[1] - config_setting_.half_map_size,
                    (int64_t)loc_xyz[2] + config_setting_.half_map_size, (int64_t)loc_xyz[2] - config_setting_.half_map_size);
  double t_sliding_end = omp_get_wtime();
  std::cout<<RED<<"[DEBUG]: Map sliding using "<<t_sliding_end - t_sliding_start<<" secs"<<RESET<<"\n";
  return;
}

void VoxelMapManager::clearMemOutOfMap(const int& x_max,const int& x_min,const int& y_max,const int& y_min,const int& z_max,const int& z_min )
{
  int delete_voxel_cout = 0;
  // double delete_time = 0;
  // double last_delete_time = 0;
  for (auto it = voxel_map_.begin(); it != voxel_map_.end(); )
  {
    const VOXEL_LOCATION& loc = it->first;
    bool should_remove = loc.x > x_max || loc.x < x_min || loc.y > y_max || loc.y < y_min || loc.z > z_max || loc.z < z_min;
    if (should_remove){
      // last_delete_time = omp_get_wtime();
      delete it->second;
      it = voxel_map_.erase(it);
      // delete_time += omp_get_wtime() - last_delete_time;
      delete_voxel_cout++;
    } else {
      ++it;
    }
  }
  std::cout<<RED<<"[DEBUG]: Delete "<<delete_voxel_cout<<" root voxels"<<RESET<<"\n";
  // std::cout<<RED<<"[DEBUG]: Delete "<<delete_voxel_cout<<" voxels using "<<delete_time<<" s"<<RESET<<"\n";
}