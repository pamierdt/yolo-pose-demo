/**
 * JumpRopeCounter.h
 *
 * 跳绳计数器 - 基于姿态估计的跳跃检测算法
 * Jump Rope Counter - Jump detection algorithm based on pose estimation
 *
 * 算法特点 / Algorithm Features:
 * - 三状态机设计 (Three-state machine)
 * - 动态阈值自适应 (Dynamic threshold adaptation)
 * - 地面基准自校准 (Ground baseline auto-calibration)
 * - 防作弊检测 (Anti-cheating detection)
 * - 冷却时间保护 (Cooldown protection)
 *
 * 使用方法 / Usage:
 * ```cpp
 * JumpRopeCounter counter(300.0f);  // 300ms minimum interval
 * int count = counter.update(shoulderY, hipY, ankleY, timestampMs);
 * ```
 */

#pragma once

#ifndef JUMP_ROPE_COUNTER_H
#define JUMP_ROPE_COUNTER_H

#include <algorithm>
#include <cmath>
#include <deque>
#include <vector>

class JumpRopeCounter {
public:
  /**
   * 状态机定义 / State Machine Definition
   *
   * STATE_GROUND    (0): 地面状态，等待起跳 / On ground, waiting for jump
   * STATE_ASCENDING (1): 上升状态，正在起跳 / Ascending, jumping up
   * STATE_DESCENDING(2): 下降状态，正在落地 / Descending, landing
   *
   * 状态转换 / State Transitions:
   * GROUND -> AIR: 髋部抬升超过自适应波峰的60% / Lift > 60% of Adaptive Peak
   * AIR -> GROUND: 髋部回落低于自适应波峰的35% / Lift < 35% of Adaptive Peak
   */
  enum State {
    STATE_CALIBRATING = 0, // 校准中 / Calibrating
    STATE_GROUND = 1,      // 地面 / Ground (Valley)
    STATE_AIR = 2          // 腾空 / Air (Peak)
  };

  /**
   * 构造函数 / Constructor
   * @param minIntervalMs 两次跳跃之间的最小间隔时间（毫秒）/ Minimum interval
   * between two jumps (ms) 默认值: 300ms / Default: 300ms
   */
  JumpRopeCounter(float minIntervalMs = 300.0f);

  /**
   * 重置计数器状态 / Reset counter state
   *
   * 清除所有状态和计数，用于开始新的跳绳会话
   * Clears all states and counts, used to start a new jump rope session
   */
  void reset();

  /**
   * 更新状态并返回当前计数 / Update state and return current count
   *
   * 这是核心算法入口，每帧调用一次
   * This is the core algorithm entry point, called once per frame
   *
   * @param shoulderY 肩部Y坐标（归一化，0-1范围）
   *                  Shoulder Y coordinate (normalized, 0-1 range)
   *                  用途：计算躯干长度，确定动态阈值
   *                  Usage: Calculate trunk length, determine dynamic threshold
   *
   * @param hipY      髋部Y坐标（归一化，0-1范围）
   *                  Hip Y coordinate (normalized, 0-1 range)
   *                  用途：主要判定指标，跳跃检测的核心
   *                  Usage: Primary indicator, core of jump detection
   *
   * @param ankleY    踝部Y坐标（归一化，0-1范围）
   *                  Ankle Y coordinate (normalized, 0-1 range)
   *                  用途：防作弊检测，验证真实跳跃
   *                  Usage: Anti-cheating detection, verify real jumps
   *
   * @param timestampMs 当前时间戳（毫秒）
   *                    Current timestamp (milliseconds)
   *                    用途：计算跳跃间隔，冷却时间判定
   *                    Usage: Calculate jump interval, cooldown determination
   *
   * @return 当前累计跳跃次数 / Current accumulated jump count
   *
   * 注意事项 / Notes:
   * - Y坐标系向下为正（图像坐标系）/ Y-axis points downward (image coordinates)
   * - 建议使用左右关键点的平均值 / Recommend using average of left/right
   * keypoints
   * - 确保关键点置信度足够高 / Ensure keypoint confidence is high enough
   */
  int update(float shoulderY, float hipY, float ankleY, double timestampMs);

  /**
   * 获取当前计数 / Get current count
   * @return 累计跳跃次数 / Accumulated jump count
   */
  int getCount() const;

  /**
   * 获取地面基准Y坐标 / Get ground baseline Y coordinate
   * @return 髋部地面基准值（归一化）/ Hip ground baseline (normalized)
   *
   * 用途：可视化调试，显示基准线位置
   * Usage: Visualization debugging, display baseline position
   */
  float getGroundY() const;

  /**
   * 设置起跳和落地阈值比例 / Set jump up and land down threshold ratios
   * @param upRatio 起跳阈值比例 (默认0.6) / Jump up threshold ratio (default
   * 0.6)
   * @param downRatio 落地阈值比例 (默认0.35) / Land down threshold ratio
   * (default 0.35)
   */
  void setThresholds(float upRatio, float downRatio);

  /**
   * 获取当前状态 / Get current state
   * @return 状态值 (0=GROUND, 1=ASCENDING, 2=DESCENDING)
   *         State value (0=GROUND, 1=ASCENDING, 2=DESCENDING)
   *
   * 用途：UI显示，调试分析
   * Usage: UI display, debugging analysis
   */
  int getState() const;

private:
  // ========== 状态变量 / State Variables ==========
  State state; // 当前状态 / Current state
  int count;   // 跳跃计数 / Jump count

  // ========== 阈值参数 / Threshold Parameters ==========
  float upThresholdRatio = 0.60f;   // 起跳阈值比例 / Jump up threshold ratio
  float downThresholdRatio = 0.35f; // 落地阈值比例 / Land down threshold ratio

  // ========== 基准线 / Baselines ==========
  float groundY; // 髋部地面基准（动态更新）/ Hip ground baseline (dynamically
                 // updated)
  float ankleGroundY;         // 踝部地面基准（动态更新）/ Ankle ground baseline
                              // (dynamically updated)
  float prevShoulderY = 0.0f; // 上一帧肩中心Y / Previous frame shoulder Y
  bool hasPrevShoulder =
      false; // 是否已有上一帧数据 / Whether previous frame exists

  // ========== 状态机变量 / State Machine Variables ==========
  double lastJumpTime;   // 上次计数时间戳（毫秒）/ Last count timestamp (ms)
  double airStartTime;   // 腾空开始时间戳 / Air start timestamp
  float currentJumpPeak; // 当前跳跃高度自适应包络 / Adaptive jump peak envelope
  float maxAnkleLiftInAir; // 腾空期间最大踝部抬升 / Max ankle lift during air
                           // state
  bool isJumpValid;        // 本次跳跃是否有效 / Is current jump valid

  // ========== 时间控制 / Time Control ==========
  // double lastJumpTime; // 上次计数时间戳（毫秒）/ Last count timestamp (ms)
  // // Moved to State Machine Variables
  double minInterval; // 最小跳跃间隔（毫秒）/ Minimum jump interval (ms)
  std::deque<double> recentIntervals; // 最近5次跳跃间隔(用于计算平均节奏) /
                                      // Recent 5 jump intervals

  // ========== 跳跃追踪 / Jump Tracking ==========
  // float maxJumpY; // 本次跳跃最高点（Y最小值）/ Current jump peak (min Y
  // value) // Removed bool isJumpValid; // 本次跳跃是否有效（踝部验证通过）/ Is
  // current jump valid // Moved to State Machine Variables (ankle verified)

  // ========== 校准参数 / Calibration ==========
  static const int CALIBRATION_FRAMES = 3; // 快速启动：取前3帧作为初始基准
  static const int POST_CALIBRATION_FRAMES =
      10; // 校准后额外补偿帧，基准快速收敛
  int calibrationCounter;
  float calibrationHipSum;
  float calibrationAnkleSum;
  int postCalibCounter;

  // ========== 优化：多帧平滑 / Optimization: Multi-frame Smoothing ==========
  static const int SMOOTH_WINDOW_SIZE = 3;
  struct WeightedMovingAverage {
    float buffer[SMOOTH_WINDOW_SIZE];
    int index = 0;
    int count = 0;
    const float weights[3] = {0.2f, 0.3f, 0.5f};

    void push(float value) {
      buffer[index] = value;
      index = (index + 1) % SMOOTH_WINDOW_SIZE;
      if (count < SMOOTH_WINDOW_SIZE)
        count++;
    }

    float get() const {
      if (count == 0)
        return 0.0f;
      if (count < SMOOTH_WINDOW_SIZE) {
        float sum = 0.0f;
        for (int i = 0; i < count; i++)
          sum += buffer[i];
        return sum / count;
      }

      float weightedSum = 0.0f;
      float wSum = 0.0f;

      int idx_0 = (index - 1 + SMOOTH_WINDOW_SIZE) % SMOOTH_WINDOW_SIZE;
      weightedSum += buffer[idx_0] * weights[2];
      wSum += weights[2];

      int idx_1 = (index - 2 + SMOOTH_WINDOW_SIZE) % SMOOTH_WINDOW_SIZE;
      weightedSum += buffer[idx_1] * weights[1];
      wSum += weights[1];

      int idx_2 = (index - 3 + SMOOTH_WINDOW_SIZE) % SMOOTH_WINDOW_SIZE;
      weightedSum += buffer[idx_2] * weights[0];
      wSum += weights[0];

      return weightedSum / wSum;
    }

    void reset() {
      index = 0;
      count = 0;
    }
  };

  WeightedMovingAverage hipSmoother;
  WeightedMovingAverage shoulderSmoother;
  WeightedMovingAverage ankleSmoother;

  // ========== 优化：自适应阈值 / Optimization: Adaptive Threshold ==========
  static const int HISTORY_SIZE = 5;
  float jumpHeightHistory[HISTORY_SIZE];
  int historyIndex = 0;
  int historyCount = 0;

  void addJumpHeight(float height);
  float getAdaptiveThresholdCoefficient() const;

  // ========== 优化：姿态验证 / Optimization: Pose Validation ==========
  bool checkPoseValidity(float shoulderY, float hipY, float ankleY);
};

#endif // JUMP_ROPE_COUNTER_H
