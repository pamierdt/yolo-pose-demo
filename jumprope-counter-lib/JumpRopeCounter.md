# Jump Rope Counter Library

跳绳计数器Android库 - 基于姿态估计的智能跳跃检测算法  
Jump Rope Counter Android Library - Intelligent jump detection algorithm based on pose estimation

## Features / 特性

- ✅ **高精度计数** - 基于波峰波谷检测的状态机算法 / High-precision counting with peak-valley detection
- ✅ **自适应阈值** - 动态调整检测参数，适应不同跳跃幅度 / Adaptive thresholds for different jump amplitudes
- ✅ **防作弊机制** - 多重验证确保真实跳跃 / Anti-cheating with multiple validations
- ✅ **零依赖** - 纯C++实现，无外部依赖 / Zero dependencies, pure C++ implementation
- ✅ **轻量级** - SO文件仅~40KB (arm64-v8a) / Lightweight, only ~40KB SO file
- ✅ **跨平台** - 支持arm64-v8a和armeabi-v7a / Cross-platform support

## Integration / 集成方式

### Method 1: Gradle Module Dependency (推荐)

在项目的 `settings.gradle` 中添加：
```gradle
include ':jumprope-counter-lib'
project(':jumprope-counter-lib').projectDir = new File('/path/to/jumprope-counter-lib')
```

在 app 的 `build.gradle` 中添加依赖：
```gradle
dependencies {
    implementation project(':jumprope-counter-lib')
}
```

### Method 2: AAR File

1. 将 `jumprope-counter-lib-release.aar` 复制到 `app/libs/` 目录
2. 在 `build.gradle` 中添加：
```gradle
dependencies {
    implementation files('libs/jumprope-counter-lib-release.aar')
}
```

## Usage / 使用方法

### Kotlin Example

```kotlin
import com.yolo.jumprope.JumpRopeCounter

// 创建计数器 (最小跳跃间隔300ms)
val counter = JumpRopeCounter(minIntervalMs = 300f)

// 在姿态检测回调中更新
fun onPoseDetected(landmarks: List<Landmark>, timestampMs: Double) {
    // 获取关键点坐标 (归一化 0-1)
    val shoulderY = (landmarks[11].y + landmarks[12].y) / 2f  // 肩部中点
    val hipY = (landmarks[23].y + landmarks[24].y) / 2f       // 髋部中点
    val ankleY = (landmarks[27].y + landmarks[28].y) / 2f     // 踝部中点
    
    // 更新计数器
    val jumpCount = counter.update(shoulderY, hipY, ankleY, timestampMs)
    
    // 显示结果
    textView.text = "Jump Count: $jumpCount"
}

// 重置计数
counter.reset()

// 释放资源
counter.close()

// 或使用 use {} 自动释放
JumpRopeCounter().use { counter ->
    // 使用计数器
}
```

### Advanced Configuration

```kotlin
// 设置自定义阈值
counter.setThresholds(
    upRatio = 0.60f,    // 起跳阈值 (默认0.60)
    downRatio = 0.35f   // 落地阈值 (默认0.35)
)

// 获取当前状态
val state = counter.getState()  // 0=地面, 1=上升, 2=下降

// 获取地面基准线
val groundY = counter.getGroundY()
```

## API Reference

### Constructor

```kotlin
JumpRopeCounter(minIntervalMs: Float = 300f)
```
- `minIntervalMs`: 两次跳跃之间的最小间隔（毫秒），防止重复计数

### Methods

| Method | Description | Return |
|--------|-------------|--------|
| `update(shoulderY, hipY, ankleY, timestampMs)` | 更新状态并返回计数 | Int |
| `getCount()` | 获取当前计数 | Int |
| `getState()` | 获取当前状态 (0/1/2) | Int |
| `getGroundY()` | 获取地面基准Y坐标 | Float |
| `setThresholds(upRatio, downRatio)` | 设置检测阈值 | Unit |
| `reset()` | 重置计数器 | Unit |
| `close()` | 释放资源 | Unit |

## Algorithm Details / 算法详情

### State Machine / 状态机

```
STATE_CALIBRATING (0) → STATE_GROUND (1) ⇄ STATE_AIR (2)
```

1. **CALIBRATING**: 初始校准阶段，采集前3帧建立地面基准
2. **GROUND**: 地面状态，等待起跳
3. **AIR**: 腾空状态，等待落地

### Detection Logic / 检测逻辑

- **起跳检测**: 髋部抬升 > 60% 自适应波峰
- **落地检测**: 髋部回落 < 35% 自适应波峰
- **有效性验证**: 腾空时间 > 80ms
- **冷却时间**: 防止同一跳跃重复计数

## Performance / 性能

- **CPU占用**: < 1% (单线程)
- **内存占用**: < 1MB
- **延迟**: < 1ms per frame
- **准确率**: > 99% (正常跳绳场景)

## Requirements / 系统要求

- Android API Level 24+ (Android 7.0+)
- arm64-v8a or armeabi-v7a architecture
- 姿态估计输入 (归一化坐标 0-1)

## License

Apache License 2.0

## Support / 支持

For issues and questions, please create an issue in the repository.
