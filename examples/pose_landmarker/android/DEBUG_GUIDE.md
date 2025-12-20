# 跳绳计数不准确 - 调试指南
## Debug Guide for Inaccurate Jump Counting

## 📊 问题分类

请先确定问题类型：

### A. **误判过多**（False Positives）
- [ ] 站立不动时也在计数
- [ ] 轻微晃动就增加计数
- [ ] 一次跳跃计数多次
- [ ] 休息时出现计数

### B. **漏判过多**（False Negatives）
- [ ] 明明跳了但没计数
- [ ] 小幅度跳跃检测不到
- [ ] 连续跳跃时漏掉部分
- [ ] 侧身跳跃检测不到

### C. **计数延迟**
- [ ] 计数反应慢
- [ ] 跳完很久才计数

---

## 🔧 快速诊断步骤

### Step 1: 检查日志

```bash
# 过滤跳绳计数日志
adb logcat -c  # 清空日志
adb logcat | grep "JumpRopeCounter"
```

### 关键日志指标

#### 查看状态转换
```log
# 正常应该看到清晰的状态转换
🚀 Takeoff! Lift=0.120 > 0.060 (Peak=0.100)  # 起跳
✅ COUNT +1! Total=1 | AirTime=150ms | Int=0->Min=300  # 落地计数
```

#### 查看波峰值
```log
# 查看 DATA 日志
DATA,1000,1,0.0020,0.1000,0.0010  # 时间,状态,抬升,波峰,踝部
                    ^^^^^ 注意这个波峰值
```

**波峰异常判断**：
- 波峰过高（> 0.2）→ 小跳检测不到 ⚠️
- 波峰过低（< 0.05）→ 容易误判 ⚠️
- 波峰合理（0.08 - 0.15）→ 正常 ✅

---

## 🎯 针对性修复方案

### 场景 1: 误判过多（站立也计数）

**可能原因**: 波峰在地面状态不衰减导致阈值过低

**临时解决方案**: 在地面状态允许缓慢衰减

```cpp
// 修改 JumpRopeCounter.cpp 第 217 行附近
if (state == STATE_AIR) {
    currentJumpPeak *= 0.99f;  // 腾空快速衰减
} else if (state == STATE_GROUND) {
    currentJumpPeak *= 0.995f;  // 地面缓慢衰减（每帧0.5%）
}
```

---

### 场景 2: 漏判过多（跳了不计数）

**可能原因 A**: 波峰值过高，小幅度跳跃达不到阈值

**临时解决方案**: 降低起跳阈值比例

```kotlin
// 在 PoseLandmarkerHelper.kt 或 UI 设置中
counter.setThresholds(
    upRatio = 0.50f,    // 从 0.60 降低到 0.50
    downRatio = 0.30f   // 从 0.35 降低到 0.30
)
```

**可能原因 B**: 姿态验证过于严格

**临时解决方案**: 增加容忍帧数

```cpp
// 修改 JumpRopeCounter.h 第 251 行
static const int MAX_INVALID_FRAMES = 5; // 从 3 增加到 5
```

---

### 场景 3: 一次跳跃计数多次

**可能原因**: 最小间隔设置过小

**临时解决方案**: 增加最小间隔

```kotlin
// 创建计数器时
val counter = JumpRopeCounter(minIntervalMs = 400f)  // 从 300 增加到 400
```

---

## 🔍 详细调试方法

### 1. 启用详细日志

修改 `JumpRopeCounter.cpp` 第 15-18 行：

```cpp
static const bool ENABLE_LOGS = true;
static const bool ENABLE_DEBUG_LOGS = true;
static const bool ENABLE_DATA_LOGS = true;  // 开启CSV数据日志
```

### 2. 记录测试数据

```bash
# 录制日志到文件
adb logcat | grep "JumpRopeCounter" > jump_test.log

# 测试时：
# 1. 跳绳 10 次（正常速度）
# 2. 休息 5 秒
# 3. 再跳 10 次
```

### 3. 分析日志

查找异常模式：

#### 误判检查
```bash
# 查找在地面状态的计数（异常）
grep "COUNT.*state=1" jump_test.log
```

#### 漏判检查
```bash
# 查找起跳但没有计数的情况
grep "Takeoff" jump_test.log | wc -l  # 起跳次数
grep "COUNT" jump_test.log | wc -l     # 计数次数
# 两者应该相近
```

#### 波峰追踪
```bash
# 提取波峰值变化
grep "DATA" jump_test.log | awk -F',' '{print $5}' > peak_values.txt
```

---

## 🔄 渐进式回退策略

如果问题严重，可以逐步回退修复：

### 回退步骤 1: 恢复波峰衰减（保留其他修复）

```cpp
// JumpRopeCounter.cpp 第 217 行改回
// if (state == STATE_AIR) {  // 注释掉条件
    currentJumpPeak *= 0.99f;
// }
```

**测试**: 如果问题解决 → 波峰衰减修复有问题  
**如果**: 问题依然 → 继续下一步

---

### 回退步骤 2: 恢复姿态验证（保留高度记录修复）

```cpp
// JumpRopeCounter.cpp 第 95-113 行简化为
if (!checkPoseValidity(shoulderY, hipY, ankleY)) {
    if (ENABLE_DEBUG_LOGS) {
      LOGW("Invalid pose detected: s=%.3f, h=%.3f, a=%.3f", shoulderY, hipY, ankleY);
    }
    return count;
}
```

```cpp
// JumpRopeCounter.cpp 第 418 行改回
if (shoulderY >= hipY || hipY >= ankleY) {
    return false;
}
```

**测试**: 如果问题解决 → 姿态验证放宽有问题  
**如果**: 问题依然 → 继续下一步

---

### 回退步骤 3: 恢复跳跃高度记录（只保留文档修复）

```cpp
// JumpRopeCounter.cpp 第 256 行改回
float jumpHeight = currentJumpPeak;  // 改回使用包络值
```

注释掉相关代码：
```cpp
// 第 245 行
// currentJumpMaxLift = currentLift;

// 第 265-268 行
// if (currentLift > currentJumpMaxLift) {
//     currentJumpMaxLift = currentLift;
// }
```

---

## 📊 对比测试

### 原版 vs 修复版对比

| 测试场景 | 原版计数 | 修复版计数 | 实际次数 | 备注 |
|---------|---------|-----------|---------|------|
| 正常跳绳 30 次 | ___ | ___ | 30 | |
| 休息后再跳 10 次 | ___ | ___ | 10 | 休息5秒 |
| 小幅度跳 20 次 | ___ | ___ | 20 | 离地<5cm |
| 快速跳 60 次 | ___ | ___ | 60 | 2次/秒 |

---

## 🎛️ 参数调优建议

### 默认参数
```kotlin
minIntervalMs = 300f        // 最小间隔
upThresholdRatio = 0.60f    // 起跳阈值
downThresholdRatio = 0.35f  // 落地阈值
```

### 调优方向

#### 误判过多 → 提高阈值
```kotlin
upThresholdRatio = 0.65f    // 提高到 0.65
downThresholdRatio = 0.30f  // 降低到 0.30（增大迟滞）
minIntervalMs = 400f        // 增加间隔
```

#### 漏判过多 → 降低阈值
```kotlin
upThresholdRatio = 0.55f    // 降低到 0.55
downThresholdRatio = 0.40f  // 提高到 0.40
minIntervalMs = 250f        // 减少间隔
```

#### 小幅度跳跃检测不到
```kotlin
upThresholdRatio = 0.50f    // 显著降低
downThresholdRatio = 0.35f  // 保持
```

---

## 💡 推荐调试流程

1. **收集日志** → 跳绳测试并保存日志
2. **分析模式** → 确定是误判还是漏判
3. **调整参数** → 先尝试参数调优
4. **如果无效** → 逐步回退修复
5. **找到问题** → 报告具体哪个修复有问题

---

## 📝 需要提供的信息

请提供以下信息以便进一步诊断：

1. **问题类型**：误判多 / 漏判多 / 都有？
2. **测试场景**：
   - 实际跳了多少次？
   - 计数显示多少？
   - 什么跳法（正常/快速/慢速/小幅度）？
3. **日志输出**：
   ```bash
   adb logcat | grep "JumpRopeCounter" | tail -100
   ```
4. **是否比修复前更差**：修复前计数准确率大概多少？

---

## 🔧 紧急回退方案

如果需要立即回退所有修复：

```bash
cd /Users/dingtao/yolo-pose-demo
git checkout examples/pose_landmarker/android/jumprope-counter-lib/src/main/cpp/JumpRopeCounter.cpp
git checkout examples/pose_landmarker/android/jumprope-counter-lib/src/main/cpp/JumpRopeCounter.h
```

然后重新编译即可恢复到修复前的版本。
