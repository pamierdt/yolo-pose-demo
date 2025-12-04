#include "JumpRopeCounter.h"
#include <algorithm>
#include <android/log.h>
#include <cmath>

#define LOG_TAG "JumpRopeCounter"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, LOG_TAG, __VA_ARGS__)

// 日志开关 - 可通过编译选项控制
// Enable logging for this module
static const bool ENABLE_LOGS = true;
static const bool ENABLE_DEBUG_LOGS = false; // 详细调试日志

/**
 * 构造函数
 * @param minIntervalMs 两次有效跳跃之间的最小时间间隔（毫秒）
 *                      默认300ms，防止同一次跳跃被重复计数
 */
JumpRopeCounter::JumpRopeCounter(float minIntervalMs)
    : minInterval(minIntervalMs), state(STATE_GROUND), count(0), groundY(0.0f),
      ankleGroundY(0.0f), lastJumpTime(0.0), maxJumpY(0.0f), isJumpValid(false),
      initialized(false), historyIndex(0), historyCount(0) {
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
  state = STATE_GROUND;
  groundY = 0.0f;
  ankleGroundY = 0.0f;
  lastJumpTime = 0.0;
  maxJumpY = 0.0f;
  isJumpValid = false;
  initialized = false;

  // 重置优化器状态
  hipSmoother.reset();
  shoulderSmoother.reset();
  ankleSmoother.reset();
  historyIndex = 0;
  historyCount = 0;

  if (ENABLE_LOGS) {
    LOGI("JumpRopeCounter reset");
  }
}

/**
 * 核心更新函数 - 跳绳计数算法主逻辑
 *
 * 算法原理：
 * 1. 使用三状态机：GROUND(地面) -> ASCENDING(上升) -> DESCENDING(下降) ->
 * GROUND
 * 2. 动态阈值：基于躯干长度自适应调整，适应不同身高用户
 * 3. 地面基准自适应：使用低通滤波器动态更新地面位置
 * 4. 防作弊机制：验证踝部抬升，过滤"假跳"（只弯腰不起跳）
 * 5. 冷却时间：防止同一次跳跃被重复计数
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
  if (!isValidPose(shoulderY, hipY, ankleY)) {
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

  // ========== 初始化阶段 ==========
  // 首次调用时，将当前位置设为地面基准
  if (!initialized) {
    groundY = smoothHipY;
    ankleGroundY = smoothAnkleY;
    initialized = true;
    if (ENABLE_LOGS) {
      LOGI("✓ Initialized - groundY=%.3f, ankleGroundY=%.3f", groundY,
           ankleGroundY);
    }
    return count;
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
    float oldGroundY = groundY;
    groundY = 0.95f * groundY + 0.05f * smoothHipY;
    ankleGroundY = 0.95f * ankleGroundY + 0.05f * smoothAnkleY;

    if (ENABLE_DEBUG_LOGS && std::abs(groundY - oldGroundY) > 0.001f) {
      LOGD("Ground baseline updated: %.3f -> %.3f (delta=%.3f)", oldGroundY,
           groundY, groundY - oldGroundY);
    }
  }

  // ========== 3. 计算关键指标 ==========
  // hipLift: 髋部抬升高度（正值表示向上，因为Y轴向下）
  float hipLift = groundY - smoothHipY;
  float ankleLift = ankleGroundY - smoothAnkleY;

  if (ENABLE_DEBUG_LOGS) {
    LOGD("Frame data: hipLift=%.3f, ankleLift=%.3f, threshold=%.3f, state=%d",
         hipLift, ankleLift, threshold, state);
  }

  // ========== 4. 状态机逻辑 ==========
  switch (state) {
  case STATE_GROUND:
    // ===== 地面状态 =====
    // 等待起跳信号：髋部抬升超过阈值
    if (hipLift > threshold) {
      state = STATE_ASCENDING;
      maxJumpY = smoothHipY; // 记录当前位置为最高点（后续会更新）
      isJumpValid = false;   // 重置有效性标志，等待踝部验证

      if (ENABLE_LOGS) {
        LOGI("🚀 Jump Started! hipLift=%.3f, threshold=%.3f (coef=%.2f), "
             "timestamp=%.0fms",
             hipLift, threshold, adaptiveCoef, timestampMs);
      }
    }
    break;

  case STATE_ASCENDING:
    // ===== 上升状态 =====
    // 持续追踪最高点（Y坐标最小值）
    if (smoothHipY < maxJumpY) {
      maxJumpY = smoothHipY;
      if (ENABLE_DEBUG_LOGS) {
        LOGD("↑ New peak: maxJumpY=%.3f", maxJumpY);
      }
    }

    // 防作弊检测：验证踝部是否真实抬升
    // 要求踝部抬升超过阈值的50%（相对宽松，避免漏检）
    if ((ankleGroundY - smoothAnkleY) > (threshold * 0.5f)) {
      if (!isJumpValid && ENABLE_DEBUG_LOGS) {
        LOGD("✓ Ankle validation passed: ankleLift=%.3f", ankleLift);
      }
      isJumpValid = true;
    }

    // 检测下降开始：使用滞后判定防止抖动
    // 当髋部位置比最高点下降超过 20% 阈值时，认为开始下降
    if (smoothHipY > (maxJumpY + threshold * 0.2f)) {
      state = STATE_DESCENDING;
      float jumpHeight = groundY - maxJumpY;
      if (ENABLE_LOGS) {
        LOGI("⬇ Descending started - jumpHeight=%.3f, ankleValid=%d",
             jumpHeight, isJumpValid);
      }
    }
    break;

  case STATE_DESCENDING:
    // ===== 下降状态 =====
    // 继续检查踝部（补充验证机会）
    if ((ankleGroundY - smoothAnkleY) > (threshold * 0.5f)) {
      isJumpValid = true;
    }

    // 落地判定：髋部回到接近地面基准线
    // 使用 50% 阈值作为落地容差，避免过于严格
    if (hipLift < (threshold * 0.5f)) {
      // 计算与上次跳跃的时间间隔
      double timeDiff = timestampMs - lastJumpTime;
      float jumpHeight = groundY - maxJumpY;

      // 判定是否计数：需同时满足踝部验证和冷却时间
      if (isJumpValid && timeDiff > minInterval) {
        count++;
        lastJumpTime = timestampMs;

        // 优化：记录跳跃高度用于自适应阈值
        addJumpHeight(jumpHeight);

        if (ENABLE_LOGS) {
          LOGI("✅ COUNT +1! Total=%d | height=%.3f, interval=%.0fms, "
               "ankle=%.3f",
               count, jumpHeight, timeDiff, ankleLift);
        }
      } else {
        // 记录未计数原因，用于调试
        if (ENABLE_LOGS) {
          if (!isJumpValid) {
            LOGW("❌ Invalid jump (ankle not lifted enough: %.3f < %.3f)",
                 ankleLift, threshold * 0.5f);
          } else {
            LOGW("❌ Cooldown period (interval=%.0fms < %.0fms)", timeDiff,
                 minInterval);
          }
        }
      }

      // 状态转换回地面
      state = STATE_GROUND;

      // 落地后快速校准基准线（α=0.5，比正常更新更激进）
      // 这样可以快速适应用户位置变化（如向前/后移动）
      groundY = 0.5f * groundY + 0.5f * smoothHipY;
      ankleGroundY = 0.5f * ankleGroundY + 0.5f * smoothAnkleY;

      if (ENABLE_DEBUG_LOGS) {
        LOGD("⬛ Landed - ground recalibrated to %.3f", groundY);
      }
    }
    break;
  }

  return count;
}

// ========== 优化方法实现 / Optimization Implementation ==========

bool JumpRopeCounter::isValidPose(float shoulderY, float hipY,
                                  float ankleY) const {
  // 1. 生理学合理性检查 (Y轴向下)
  // 肩部应该在髋部上方 (shoulderY < hipY)
  // 髋部应该在踝部上方 (hipY < ankleY)
  if (shoulderY >= hipY || hipY >= ankleY) {
    return false;
  }

  // 2. 躯干长度合理性检查
  float trunkLen = hipY - shoulderY;
  // 躯干长度过短或过长都可能是错误的姿态估计
  if (trunkLen < 0.05f || trunkLen > 0.5f) {
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
  // 小幅度跳跃 -> 降低阈值 (0.08)
  // 大幅度跳跃 -> 提高阈值 (0.12)
  // 默认 -> 0.10
  if (avgHeight < 0.05f)
    return 0.08f;
  if (avgHeight > 0.15f)
    return 0.12f;

  return 0.10f;
}

int JumpRopeCounter::getCount() const { return count; }

float JumpRopeCounter::getGroundY() const { return groundY; }

int JumpRopeCounter::getState() const { return state; }
