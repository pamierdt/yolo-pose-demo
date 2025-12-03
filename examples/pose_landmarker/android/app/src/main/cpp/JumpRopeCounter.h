#pragma once

#include <algorithm>
#include <cmath>

class JumpRopeCounter {
public:
    // 状态机定义：地面(GROUND), 上升(ASCENDING), 下降(DESCENDING)
    enum State {
        STATE_GROUND = 0,
        STATE_ASCENDING = 1,
        STATE_DESCENDING = 2
    };

    // 构造函数
    // minIntervalMs: 两次跳跃之间的最小间隔时间（毫秒）
    JumpRopeCounter(float minIntervalMs = 300.0f);

    // 重置计数器状态
    void reset();

    // 更新状态并返回当前计数
    // shoulderY: 肩部Y坐标 (用于计算躯干长度)
    // hipY: 髋部Y坐标 (核心判定指标)
    // ankleY: 踝部Y坐标 (用于防作弊/假跳检测)
    // timestampMs: 当前时间戳（毫秒）
    int update(float shoulderY, float hipY, float ankleY, double timestampMs);

    int getCount() const;
    float getGroundY() const;
    int getState() const;

private:
    State state;
    int count;
    
    float groundY;      // 髋部地面基准
    float ankleGroundY; // 踝部地面基准
    
    double lastJumpTime; // 上次计数时间
    double minInterval;  // 最小跳跃间隔
    
    float maxJumpY;     // 本次跳跃最高点 (Min Y value in image coords)
    bool isJumpValid;   // 本次跳跃是否有效 (踝部检测通过)
    
    bool initialized;
};
