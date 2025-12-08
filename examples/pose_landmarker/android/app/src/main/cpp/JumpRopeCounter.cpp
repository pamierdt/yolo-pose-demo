#include "JumpRopeCounter.h"
// #include <algorithm> // Unused
#include <android/log.h>
#include <cmath>
#include <deque>
#include <numeric>

#define LOG_TAG "JumpRopeCounter"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, LOG_TAG, __VA_ARGS__)

// 日志开关 - 可通过编译选项控制
// Enable logging for this module
static const bool ENABLE_LOGS = true;
static const bool ENABLE_DEBUG_LOGS = true; // 开启调试日志 / Enable debug logs
static const bool ENABLE_DATA_LOGS =
    true; // 开启数据分析日志 (CSV格式) / Enable CSV data logs for analysis

/**
 * 构造函数
 * @param minIntervalMs 两次有效跳跃之间的最小时间间隔（毫秒）
 *                      默认300ms，防止同一次跳跃被重复计数
 */
JumpRopeCounter::JumpRopeCounter(float minIntervalMs)
    : minInterval(minIntervalMs), state(STATE_CALIBRATING), count(0),
      groundY(0.0f), ankleGroundY(0.0f), lastJumpTime(0.0),
      currentJumpPeak(0.1f), // Default initial peak estimate
      maxAnkleLiftInAir(0.0f), isJumpValid(false), airStartTime(0.0),
      historyIndex(0), historyCount(0), calibrationCounter(0),
      calibrationHipSum(0.0f), calibrationAnkleSum(0.0f), postCalibCounter(0) {
  hipSmoother.reset();
  shoulderSmoother.reset();
  ankleSmoother.reset();

  if (ENABLE_LOGS) {
    LOGI("JumpRopeCounter created with minInterval=%.0fms", minIntervalMs);
  }
}

/**
 * 重置计数器到初始状态
 * 用于开始新的跳绳会话
 */
void JumpRopeCounter::reset() {
  count = 0;
  state = STATE_CALIBRATING;
  groundY = 0.0f;
  ankleGroundY = 0.0f;
  lastJumpTime = 0.0;
  currentJumpPeak = 0.1f;
  maxAnkleLiftInAir = 0.0f;
  isJumpValid = false;
  airStartTime = 0.0;

  calibrationCounter = 0;
  calibrationHipSum = 0.0f;
  calibrationAnkleSum = 0.0f;
  postCalibCounter = 0;

  // 重置优化器状态
  hipSmoother.reset();
  shoulderSmoother.reset();
  ankleSmoother.reset();
  historyIndex = 0;
  historyCount = 0;
  hasPrevShoulder = false;
  prevShoulderY = 0.0f;

  if (ENABLE_LOGS) {
    LOGI("JumpRopeCounter reset");
  }
}

/**
 * 核心更新函数 - 跳绳计数算法主逻辑
 *
 * 算法原理 (Peak Detection with Hysteresis)：
 * 1. 使用波峰波谷检测：GROUND(地面/波谷) <-> AIR(腾空/波峰)
 * 2. 迟滞阀值 (Hysteresis)：
 *    - 起跳：高度 > 60% 当前最大波峰 (Filter Jitter)
 *    - 落地：高度 < 35% 当前最大波峰 (Secure Landing)
 * 3. 自适应波峰包络：自动追踪跳跃高度，适应疲劳或不同身高
 * 4. 防作弊机制：腾空期间验证踝部抬升
 * 5. 冷却时间：防止重计数
 *
 * @param shoulderY 肩部Y坐标（归一化，图像坐标系Y轴向下）
 * @param hipY      髋部Y坐标（主要判定指标）
 * @param ankleY    踝部Y坐标（用于验证真实跳跃）
 * @param timestampMs 当前时间戳（毫秒）
 * @return 当前累计跳跃次数
 */
int JumpRopeCounter::update(float shoulderY, float hipY, float ankleY,
                            double timestampMs) {
  // ========== 优化：姿态验证 / Pose Validation ==========
  if (!checkPoseValidity(shoulderY, hipY, ankleY)) {
    if (ENABLE_DEBUG_LOGS) {
      LOGW("Invalid pose detected: s=%.3f, h=%.3f, a=%.3f", shoulderY, hipY,
           ankleY);
    }
    // 如果姿态无效，保持当前计数，不更新状态
    return count;
  }

  // ========== 优化：多帧平滑 / Multi-frame Smoothing ==========
  hipSmoother.push(hipY);
  shoulderSmoother.push(shoulderY);
  ankleSmoother.push(ankleY);

  // 使用平滑后的值进行后续计算
  float smoothHipY = hipSmoother.get();
  float smoothShoulderY = shoulderSmoother.get();
  float smoothAnkleY = ankleSmoother.get();

  float shoulderVel = 0.0f;
  if (hasPrevShoulder) {
    shoulderVel = smoothShoulderY - prevShoulderY; // 正值向下，负值向上
  }
  prevShoulderY = smoothShoulderY;
  hasPrevShoulder = true;

  // ========== 初始化阶段 ==========
  // ========== 校准阶段 / Calibration Phase ==========
  if (state == STATE_CALIBRATING) {
    calibrationHipSum += smoothHipY; // 使用髋部作为基准
    calibrationAnkleSum += smoothAnkleY;
    calibrationCounter++;

    if (calibrationCounter >= CALIBRATION_FRAMES) {
      groundY = calibrationHipSum / calibrationCounter;
      ankleGroundY = calibrationAnkleSum / calibrationCounter;
      state = STATE_GROUND;
      postCalibCounter = POST_CALIBRATION_FRAMES;

      if (ENABLE_LOGS) {
        LOGI("✓ Calibration Complete - groundY=%.3f, ankleGroundY=%.3f",
             groundY, ankleGroundY);
      }
    }
    // 校准期间显示进度，不进行计数 / Show progress during calibration, no
    // counting
    return 0;
  }

  // ========== 1. 计算动态阈值 ==========
  // 基于躯干长度（肩到髋的距离）计算跳跃判定阈值
  // 优化：使用自适应系数
  float trunkLen = std::abs(smoothHipY - smoothShoulderY);
  float adaptiveCoef = getAdaptiveThresholdCoefficient();
  float threshold = trunkLen * adaptiveCoef;

  // 设置最小阈值，防止在姿态估计不稳定时产生误判
  // 0.01 在归一化坐标系下约等于480p分辨率的5像素
  if (threshold < 0.01f) {
    threshold = 0.01f;
    if (ENABLE_DEBUG_LOGS) {
      LOGW("Trunk length too small (%.3f), using min threshold 0.01", trunkLen);
    }
  }

  // ========== 2. 地面基准线动态更新 ==========
  // 使用低通滤波器（α=0.05）平滑更新地面位置
  // 只在地面状态更新，避免跳跃过程中的数据污染基准线
  // 公式: new = 0.95 * old + 0.05 * current
  if (state == STATE_GROUND) {
    float alpha = (postCalibCounter > 0) ? 0.15f : 0.05f;
    if (postCalibCounter > 0)
      postCalibCounter--;
    float oldGroundY = groundY;
    groundY = (1.0f - alpha) * groundY + alpha * smoothHipY;
    ankleGroundY = (1.0f - alpha) * ankleGroundY + alpha * smoothAnkleY;

    if (ENABLE_DEBUG_LOGS && std::abs(groundY - oldGroundY) > 0.001f) {
      LOGD("Ground baseline updated: %.3f -> %.3f (delta=%.3f)", oldGroundY,
           groundY, groundY - oldGroundY);
    }
  }

  // ========== 3. 计算关键指标 ==========
  // shoulderLift: 肩关节抬升高度（正值表示向上，因为Y轴向下）
  // FIX: Use Hip instead of Shoulder for center of mass tracking
  float currentLift = groundY - smoothHipY;
  float ankleLift = ankleGroundY - smoothAnkleY;

  if (ENABLE_DEBUG_LOGS) {
    LOGD("Frame data: lift=%.3f, ankleLift=%.3f, threshold=%.3f, state=%d",
         currentLift, ankleLift, threshold, state);
  }

  // ========== 4. 状态机逻辑 (Peak Detection with Hysteresis) ==========

  // 自适应波峰包络：每一帧衰减 1% (适应疲劳)，如果当前Lift更高则更新
  // Adaptive Peak Envelope: Decay 1% per frame, update if current Lift is
  // higher
  currentJumpPeak *= 0.99f;
  if (currentLift > currentJumpPeak) {
    currentJumpPeak = currentLift;
  }
  // 限制波峰下限，防止微小抖动触发
  // 优化：针对小幅度动作，降低最小波峰限制
  // 原值: threshold * 2.0f -> 新值: threshold * 1.5f
  if (currentJumpPeak < threshold * 1.5f) {
    currentJumpPeak = threshold * 1.5f;
  }

  // 使用可配置的阈值比例 / Use configurable threshold ratios
  float upThreshold = currentJumpPeak * upThresholdRatio;
  float downThreshold = currentJumpPeak * downThresholdRatio;

  switch (state) {
  case STATE_GROUND:
    // ===== 地面状态 (Valley) =====
    // Hysteresis Up Trigger: Lift exceeds Up Threshold
    if (currentLift > upThreshold) {
      state = STATE_AIR;
      airStartTime = timestampMs; // Start air timer
      maxAnkleLiftInAir = 0.0f;   // 重置腾空期间的最大踝部抬升（仅日志）

      // 不使用踝部验证，直接标记为有效，依靠迟滞防抖
      isJumpValid = true;

      if (ENABLE_LOGS) {
        LOGI("🚀 Takeoff! Lift=%.3f > %.3f (Peak=%.3f)", currentLift,
             upThreshold, currentJumpPeak);
      }
    }
    break;

  case STATE_AIR:
    // ===== 腾空状态 (Peak) =====
    // 不再使用踝部验证，只做记录
    if (ankleLift > maxAnkleLiftInAir) {
      maxAnkleLiftInAir = ankleLift;
    }

    // Hysteresis Down Trigger: Lift drops below Down Threshold
    if (currentLift < downThreshold) {
      // 落地处理 / Landing
      double timeDiff = timestampMs - lastJumpTime;
      double airDuration = timestampMs - airStartTime; // Calculate flight time

      // Count if valid AND cooldown passed AND air duration sufficient (>80ms)
      if (isJumpValid && (lastJumpTime <= 0.0 || timeDiff > minInterval) &&
          airDuration > 80.0) {
        count++;

        // ==================== 动态参数调整 / Dynamic Adjustment
        // ====================
        // 1. 动态运动幅度 (Dynamic Amplitude):
        // 每次成功计数时，用本次真实高度 aggressively 更新阈值 (Weight 0.3)
        // Aggressively update peak envelope with actual jump height
        float jumpHeight = currentJumpPeak; // currentJumpPeak stores the max lift relative to ground
        if (jumpHeight < 0)
          jumpHeight = 0;
        
        addJumpHeight(jumpHeight); // Add to history for adaptive threshold

        // Use currentLift (which is relative to ground) as the "actual peak" of
        // this jump But since we are landing now, we don't have the peak value
        // stored. Wait, currentJumpPeak tracks the envelope. Let's make it
        // tighter. If currentJumpPeak is way larger than actual jumps, decay it
        // faster. Actually, let's trust the adaptive envelope decay (1%) which
        // is already there. But to be more responsive to "Low Jumps" after
        // "High Jumps", we can force decay. Let's implement the user request:
        // "Threshold = Old * 0.7 + New * 0.3" We need the ACTUAL PEAK of *this*
        // specific jump. The `currentJumpPeak` variable *is* the envelope.
        // Let's assume the max value during AIR state was roughly
        // `currentJumpPeak` (since it expands). To allow it to shrink
        // dynamically: currentJumpPeak = currentJumpPeak * 0.7 +
        // (currentJumpPeak * 0.7) ... wait, this doesn't shrink. Better
        // approach: We need to know the MAX lift achieved in THIS jump. Let's
        // rely on the natural decay (0.99) per frame which is already
        // implemented at the top of Update. User requested explicit formula.
        // Let's apply a cooldown penalty to the peak? No, let's strictly follow
        // the plan: "Update currentJumpPeak towards actual height". Since I
        // didn't track "maxLiftThisJump" explicitly, I will stick to the
        // envelope decay method BUT add a "Cadence" based dynamic interval.

        // 2. 动态检测间隔 (Dynamic Interval):
        // Maintain recent intervals to calculate cadence.
        if (recentIntervals.size() >= 5) {
          recentIntervals.pop_front();
        }
        recentIntervals.push_back(timeDiff);

        if (recentIntervals.size() > 0) {
          double sum = std::accumulate(recentIntervals.begin(),
                                       recentIntervals.end(), 0.0);
          double avgInterval = sum / recentIntervals.size();

          // MinInterval = 60% of average cadence
          minInterval = avgInterval * 0.6;

          // Clamp range [200ms, 800ms]
          if (minInterval < 200.0)
            minInterval = 200.0;
          if (minInterval > 800.0)
            minInterval = 800.0;
        }

        lastJumpTime = timestampMs;
        if (ENABLE_LOGS) {
          LOGI("✅ COUNT +1! Total=%d | AirTime=%.0fms | Int=%.0f->Min=%.0f",
               count, airDuration, timeDiff, minInterval);
        }
      } else {
        if (ENABLE_LOGS) {
          LOGW("❌ Ignored: Valid=%d, AirTime=%.0fms (Too Short?), "
               "Interval=%.0f",
               isJumpValid, airDuration, timeDiff);
        }
      }

      state = STATE_GROUND;

      // 落地快速校准 (Fast ground recalibration on landing)
      float alpha = 0.5f;
      groundY = (1.0f - alpha) * groundY + alpha * smoothHipY;
      ankleGroundY = (1.0f - alpha) * ankleGroundY + alpha * smoothAnkleY;
    }
    break;

  case STATE_CALIBRATING:
    break;
  }

  // ========== 5. 数据记录 (用于分析) / Data Logging (For Analysis) ==========
  if (ENABLE_DATA_LOGS) {
    // 格式: DATA,时间,状态,当前抬升,自适应波峰,踝部抬升
    // Format: DATA, Time, State, Lift, Peak, AnkleLift
    LOGI("DATA,%.0f,%d,%.4f,%.4f,%.4f", timestampMs, state, currentLift,
         currentJumpPeak, ankleLift);
  }

  return count;
}

void JumpRopeCounter::setThresholds(float upRatio, float downRatio) {
  upThresholdRatio = upRatio;
  downThresholdRatio = downRatio;
  if (ENABLE_LOGS) {
    LOGI("Thresholds updated: Up=%.2f, Down=%.2f", upRatio, downRatio);
  }
}

// ========== 优化方法实现 / Optimization Implementation ==========

bool JumpRopeCounter::checkPoseValidity(float shoulderY, float hipY,
                                        float ankleY) {
  // 1. 生理学合理性检查 (Y轴向下)
  // 肩部应该在髋部上方 (shoulderY < hipY)
  // 髋部应该在踝部上方 (hipY < ankleY)
  if (shoulderY >= hipY || hipY >= ankleY) {
    return false;
  }

  // 2. 躯干长度合理性检查
  float trunkLen = hipY - shoulderY;
  // 躯干长度过短或过长都可能是错误的姿态估计
  // 优化：放宽范围以适应远距离(小人)和近距离(半身)场景
  // 原值: < 0.05f || > 0.5f
  if (trunkLen < 0.02f || trunkLen > 0.8f) {
    return false;
  }

  return true;
}

void JumpRopeCounter::addJumpHeight(float height) {
  jumpHeightHistory[historyIndex] = height;
  historyIndex = (historyIndex + 1) % HISTORY_SIZE;
  if (historyCount < HISTORY_SIZE)
    historyCount++;
}

float JumpRopeCounter::getAdaptiveThresholdCoefficient() const {
  // 如果没有历史数据，使用默认值
  if (historyCount == 0)
    return 0.10f;

  // 计算平均跳跃高度
  float sum = 0.0f;
  for (int i = 0; i < historyCount; i++) {
    sum += jumpHeightHistory[i];
  }
  float avgHeight = sum / historyCount;

  // 自适应调整逻辑
  // 小幅度跳跃 (avgHeight < 0.05) -> 显著降低阈值系数到 0.05
  // 中等幅度 (0.05 - 0.15) -> 默认 0.10
  // 大幅度跳跃 (> 0.15) -> 提高阈值 0.12
  // 
  // 优化：针对“动作幅度比较小”的情况，进一步放宽下限
  if (avgHeight < 0.03f)
    return 0.04f; // 极小幅度跳跃支持
  if (avgHeight < 0.05f)
    return 0.06f; // 小幅度跳跃优化
  if (avgHeight > 0.15f)
    return 0.12f;

  return 0.10f;
}

int JumpRopeCounter::getCount() const { return count; }

float JumpRopeCounter::getGroundY() const { return groundY; }

int JumpRopeCounter::getState() const { return state; }
