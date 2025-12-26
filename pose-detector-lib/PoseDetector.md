# Pose Detector Library

YOLO姿态检测Android库 - 基于RKNN的高性能姿态估计  
YOLO Pose Detection Android Library - High-performance pose estimation based on RKNN

## Features / 特性

- ✅ **硬件加速** - 支持Rockchip NPU和RGA硬件加速 / Hardware acceleration with Rockchip NPU & RGA
- ✅ **高性能** - 零拷贝优化，FP16推理 / Zero-copy optimization, FP16 inference
- ✅ **完整集成** - 包含RKNN运行时和RGA库 / Complete integration with RKNN runtime and RGA
- ✅ **NMS后处理** - 内置非极大值抑制 / Built-in Non-Maximum Suppression
- ✅ **多架构支持** - arm64-v8a和armeabi-v7a / Multi-architecture support
- ✅ **易于使用** - 简洁的Kotlin API / Easy-to-use Kotlin API

## Integration / 集成方式

### Method 1: Gradle Module Dependency (推荐)

在项目的 `settings.gradle` 中添加：
```gradle
include ':pose-detector-lib'
project(':pose-detector-lib').projectDir = new File('/path/to/pose-detector-lib')
```

在 app 的 `build.gradle` 中添加依赖：
```gradle
dependencies {
    implementation project(':pose-detector-lib')
}
```

### Method 2: AAR File

1. 将 `pose-detector-lib-release.aar` 复制到 `app/libs/` 目录
2. 在 `build.gradle` 中添加：
```gradle
dependencies {
    implementation files('libs/pose-detector-lib-release.aar')
}
```

## Usage / 使用方法

### Kotlin Example

```kotlin
import com.yolo.pose.detector.RknnRunner
import java.nio.ByteBuffer

// 1. 加载RKNN模型
val modelBuffer = loadModelFromAssets("yolov8n-pose.rknn")
val runner = RknnRunner(modelBuffer)

// 2. 运行推理 (使用Bitmap + NMS)
val bitmap: Bitmap = // 从相机或文件获取
val results = runner.runBitmapWithNms(
    bitmap = bitmap,
    detectThreshold = 0.5f,  // 检测阈值
    nmsThreshold = 0.5f      // NMS阈值
)

// 3. 解析结果
// results 是一个 FloatArray，格式：
// [num_poses, pose1_data..., pose2_data...]
// 每个pose: [score, x, y, w, h, kp1_x, kp1_y, kp1_score, ...]

// 4. 释放资源
runner.close()
```

### Loading Model from Assets

```kotlin
fun loadModelFromAssets(fileName: String): ByteBuffer {
    val assetManager = context.assets
    val inputStream = assetManager.open(fileName)
    val modelSize = inputStream.available()
    val buffer = ByteBuffer.allocateDirect(modelSize)
    
    val channel = Channels.newChannel(inputStream)
    channel.read(buffer)
    buffer.rewind()
    
    return buffer
}
```

### Performance Optimization

```kotlin
// 设置性能选项
runner.setPerfOptions(
    useQuantOutput = false,    // 使用量化输出 (更快但精度略低)
    cacheableInput = true      // 使用可缓存输入缓冲
)
```

## API Reference

### Constructor

```kotlin
RknnRunner(modelBuffer: ByteBuffer)
```
- `modelBuffer`: RKNN模型文件的DirectByteBuffer

### Methods

| Method | Description | Return |
|--------|-------------|--------|
| `runBitmapWithNms(bitmap, detectThresh, nmsThresh)` | 运行推理并执行NMS | FloatArray |
| `run(inputBuffer, inputSize)` | 运行推理(预处理的输入) | FloatArray |
| `runPixels(pixels)` | 运行推理(ARGB像素数组) | FloatArray |
| `setPerfOptions(useQuantOutput, cacheableInput)` | 设置性能选项 | Unit |
| `close()` | 释放资源 | Unit |

### Output Format

`runBitmapWithNms` 返回的 FloatArray 格式：

```
[num_poses, pose1_data, pose2_data, ...]
```

每个 pose 的数据格式 (56个float):
```
[score, x, y, w, h,  // 5个: 置信度和边界框
 kp0_x, kp0_y, kp0_score,  // 3个: 关键点0 (鼻子)
 kp1_x, kp1_y, kp1_score,  // 3个: 关键点1 (左眼)
 ...                        // 共17个关键点
 kp16_x, kp16_y, kp16_score] // 3个: 关键点16 (右脚踝)
```

### Keypoint Indices / 关键点索引

| Index | Name | 中文名称 |
|-------|------|---------|
| 0 | Nose | 鼻子 |
| 1 | Left Eye | 左眼 |
| 2 | Right Eye | 右眼 |
| 3 | Left Ear | 左耳 |
| 4 | Right Ear | 右耳 |
| 5 | Left Shoulder | 左肩 |
| 6 | Right Shoulder | 右肩 |
| 7 | Left Elbow | 左肘 |
| 8 | Right Elbow | 右肘 |
| 9 | Left Wrist | 左腕 |
| 10 | Right Wrist | 右腕 |
| 11 | Left Hip | 左髋 |
| 12 | Right Hip | 右髋 |
| 13 | Left Knee | 左膝 |
| 14 | Right Knee | 右膝 |
| 15 | Left Ankle | 左踝 |
| 16 | Right Ankle | 右踝 |

## Model Requirements / 模型要求

### Supported Models

- YOLOv8n-pose.rknn
- YOLOv8s-pose.rknn
- YOLOv8m-pose.rknn
- 其他YOLO-pose RKNN模型

### Model Conversion

使用RKNN-Toolkit2将ONNX模型转换为RKNN格式：

```python
from rknn.api import RKNN

rknn = RKNN()
rknn.config(target_platform='rk3588')
rknn.load_onnx('yolov8n-pose.onnx')
rknn.build(do_quantization=False)
rknn.export_rknn('yolov8n-pose.rknn')
```

## Performance / 性能

### RK3588 (NPU)
- YOLOv8n-pose: ~30 FPS @ 640x640
- YOLOv8s-pose: ~20 FPS @ 640x640

### CPU Fallback
- YOLOv8n-pose: ~5 FPS @ 640x640

## Dependencies / 依赖项

库已包含以下依赖（无需额外安装）:

- `librknnrt.so` - RKNN运行时 (~8.4MB arm64, ~5.3MB armv7)
- `librga.so` - RGA硬件加速库 (~1.1MB arm64, ~723KB armv7)
- `libc++_shared.so` - C++标准库

## Requirements / 系统要求

- Android API Level 24+ (Android 7.0+)
- arm64-v8a or armeabi-v7a architecture
- Rockchip SoC (推荐，用于NPU加速)
  - RK3588/RK3588S
  - RK3566/RK3568
  - RK3562
  - 其他支持RKNN的芯片

## Troubleshooting / 故障排除

### 模型加载失败
- 确保模型文件是RKNN格式 (.rknn)
- 检查模型是否与目标平台匹配
- 确认ByteBuffer是DirectByteBuffer

### 推理速度慢
- 启用性能选项: `setPerfOptions(useQuantOutput=true, cacheableInput=false)`
- 使用量化模型
- 降低输入分辨率

### NPU不可用
- 检查设备是否支持RKNN
- 确认librknnrt.so版本与芯片匹配
- 查看logcat日志获取详细错误信息

## License

Apache License 2.0

## Support / 支持

For issues and questions, please create an issue in the repository.
