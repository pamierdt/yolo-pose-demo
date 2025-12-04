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

#include <algorithm>
#include <cmath>

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
   * GROUND -> ASCENDING: 髋部抬升超过阈值 / Hip lift exceeds threshold
   * ASCENDING -> DESCENDING: 开始下降 / Start descending
   * DESCENDING -> GROUND: 回到地面基准 / Return to ground baseline
   */
  enum State { STATE_GROUND = 0, STATE_ASCENDING = 1, STATE_DESCENDING = 2 };

  /**
   * 构造函数 / Constructor
   *
   * @param minIntervalMs 两次跳跃之间的最小间隔时间（毫秒）
   *                      Minimum interval between two jumps (milliseconds)
   *                      默认值 / Default: 300ms
   *                      推荐范围 / Recommended range: 200-500ms
   *                      - 200ms: 适合快速跳绳 / For fast jumping
   *                      - 300ms: 标准设置 / Standard setting
   *                      - 500ms: 适合慢速/儿童 / For slow/children
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

  // ========== 基准线 / Baselines ==========
  float groundY; // 髋部地面基准（动态更新）/ Hip ground baseline (dynamically
                 // updated)
  float ankleGroundY; // 踝部地面基准（动态更新）/ Ankle ground baseline
                      // (dynamically updated)

  // ========== 时间控制 / Time Control ==========
  double lastJumpTime; // 上次计数时间戳（毫秒）/ Last count timestamp (ms)
  double minInterval;  // 最小跳跃间隔（毫秒）/ Minimum jump interval (ms)

  // ========== 跳跃追踪 / Jump Tracking ==========
  float maxJumpY; // 本次跳跃最高点（Y最小值）/ Current jump peak (min Y value)
  bool isJumpValid; // 本次跳跃是否有效（踝部验证通过）/ Is current jump valid
                    // (ankle verified)

  // ========== 初始化标志 / Initialization Flag ==========
  bool initialized; // 是否已初始化 / Is initialized

  // ========== 优化：多帧平滑 / Optimization: Multi-frame Smoothing ==========
  static const int SMOOTH_WINDOW_SIZE = 3;
  struct MovingAverage {
    float buffer[SMOOTH_WINDOW_SIZE];
    int index = 0;
    int count = 0;

    void push(float value) {
      buffer[index] = value;
      index = (index + 1) % SMOOTH_WINDOW_SIZE;
      if (count < SMOOTH_WINDOW_SIZE)
        count++;
    }

    float get() const {
      if (count == 0)
        return 0.0f;
      float sum = 0.0f;
      for (int i = 0; i < count; i++)
        sum += buffer[i];
      return sum / count;
    }

    void reset() {
      index = 0;
      count = 0;
    }
  };

  MovingAverage hipSmoother;
  MovingAverage shoulderSmoother;
  MovingAverage ankleSmoother;

  // ========== 优化：自适应阈值 / Optimization: Adaptive Threshold ==========
  static const int HISTORY_SIZE = 5;
  float jumpHeightHistory[HISTORY_SIZE];
  int historyIndex = 0;
  int historyCount = 0;

  void addJumpHeight(float height);
  float getAdaptiveThresholdCoefficient() const;

  // ========== 优化：姿态验证 / Optimization: Pose Validation ==========
  bool isValidPose(float shoulderY, float hipY, float ankleY) const;
};
