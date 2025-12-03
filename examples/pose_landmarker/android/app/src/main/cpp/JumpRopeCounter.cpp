#include "JumpRopeCounter.h"
#include <android/log.h>
#include <algorithm>
#include <cmath>

#define LOG_TAG "JumpRopeCounter"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)

// Enable logging for this module
static const bool ENABLE_LOGS = true;

JumpRopeCounter::JumpRopeCounter(float minIntervalMs)
    : minInterval(minIntervalMs),
      state(STATE_GROUND),
      count(0),
      groundY(0.0f),
      ankleGroundY(0.0f),
      lastJumpTime(0.0),
      maxJumpY(0.0f),
      isJumpValid(false),
      initialized(false) {
}

void JumpRopeCounter::reset() {
    count = 0;
    state = STATE_GROUND;
    groundY = 0.0f;
    ankleGroundY = 0.0f;
    lastJumpTime = 0.0;
    maxJumpY = 0.0f;
    isJumpValid = false;
    initialized = false;
}

int JumpRopeCounter::update(float shoulderY, float hipY, float ankleY, double timestampMs) {
    if (!initialized) {
        groundY = hipY;
        ankleGroundY = ankleY;
        initialized = true;
        if (ENABLE_LOGS) LOGI("JumpRopeCounter initialized: groundY=%.2f", groundY);
        return count;
    }

    // 1. 计算动态阈值 (躯干长度 * 0.1) - 降低系数以适应小朋友
    float trunkLen = std::abs(hipY - shoulderY);
    float threshold = trunkLen * 0.10f;
    // 限制最小阈值，防止误判 (归一化坐标下设为 0.01, 约等于 480p 下的 5px)
    if (threshold < 0.01f) threshold = 0.01f;

    // 2. 地面基准线动态更新 (低通滤波)
    // 只有在地面状态才更新，避免将跳跃过程计入基准
    if (state == STATE_GROUND) {
        groundY = 0.95f * groundY + 0.05f * hipY;
        ankleGroundY = 0.95f * ankleGroundY + 0.05f * ankleY;
    }

    // 3. 状态机逻辑
    float hipLift = groundY - hipY; // 向上为正 (Y轴向下)
    
    switch (state) {
        case STATE_GROUND:
            // 触发起跳: 髋部抬升超过阈值
            if (hipLift > threshold) {
                state = STATE_ASCENDING;
                maxJumpY = hipY; // 记录最高点 (Y最小值)
                isJumpValid = false; // 重置有效性标志
                if (ENABLE_LOGS) LOGI("Jump Start! Lift=%.2f, Thr=%.2f", hipLift, threshold);
            }
            break;

        case STATE_ASCENDING:
            // 更新最高点 (Min Y)
            if (hipY < maxJumpY) {
                maxJumpY = hipY;
            }

            // 检查假跳 (踝部验证): 踝部也必须抬升超过 50% 阈值
            if ((ankleGroundY - ankleY) > (threshold * 0.5f)) {
                isJumpValid = true;
            }

            // 转入下降: 
            // 简单判定: 当前高度比最高点下降了一定幅度 (防止抖动)
            // 或者直接使用 pseudo-code 的 "开始回落"
            // 这里使用 hysteresis: 当前 Y > maxJumpY + 0.2 * threshold
            if (hipY > (maxJumpY + threshold * 0.2f)) {
                state = STATE_DESCENDING;
            }
            break;

        case STATE_DESCENDING:
            // 检查踝部 (下降过程中也可以补充验证)
            if ((ankleGroundY - ankleY) > (threshold * 0.5f)) {
                isJumpValid = true;
            }

            // 落地判定: 回到基准线附近
            if (hipLift < (threshold * 0.5f)) {
                // 冷却时间检查
                double timeDiff = timestampMs - lastJumpTime;
                
                if (isJumpValid && timeDiff > minInterval) {
                    count++;
                    lastJumpTime = timestampMs;
                    if (ENABLE_LOGS) LOGI("Count +1! Total=%d, TimeDiff=%.0fms", count, timeDiff);
                } else {
                    if (ENABLE_LOGS) {
                         if (!isJumpValid) LOGI("Invalid Jump (Ankle not lifted)");
                         else LOGI("Cooldown ignored (Diff=%.0fms)", timeDiff);
                    }
                }
                
                state = STATE_GROUND;
                // 落地后快速校准基准线
                groundY = 0.5f * groundY + 0.5f * hipY;
            }
            break;
    }

    return count;
}

int JumpRopeCounter::getCount() const {
    return count;
}

float JumpRopeCounter::getGroundY() const {
    return groundY;
}

int JumpRopeCounter::getState() const {
    return state;
}
