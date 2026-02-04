# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a YOLO Pose-based Android application for jump rope counting, optimized for Rockchip hardware with NPU acceleration. The project consists of a main Android app and two native libraries that work together to detect poses and count jump rope repetitions in real-time.

## Architecture

### Three-Module Structure

1. **app** - Main Android application
   - Camera capture and UI (CameraFragment, GalleryFragment)
   - Integrates both native libraries
   - Handles video processing and display

2. **pose-detector-lib** - Native pose detection library
   - C++ library wrapping RKNN runtime for hardware-accelerated pose estimation
   - Processes camera frames and outputs 17 keypoint coordinates per person
   - Uses Rockchip NPU (RK3588/RK3568) for inference acceleration
   - Includes prebuilt binaries: `librknnrt.so`, `librga.so`

3. **jumprope-counter-lib** - Native jump counting algorithm
   - Pure C++ state machine for jump detection
   - Analyzes pose keypoints (shoulder, hip, ankle) to count jumps
   - Implements adaptive threshold algorithm with peak-valley detection
   - Zero external dependencies

### Data Flow

```
Camera → PoseLandmarkerHelper → RknnRunner (pose-detector-lib) →
Keypoints → JumpRopeCounter (jumprope-counter-lib) → Count Display
```

### Key Integration Points

- **PoseLandmarkerHelper.kt**: Main integration point that orchestrates pose detection and jump counting
- **JumpRopeVideoProcessor.kt**: Handles video file processing
- **OverlayView.kt**: Renders pose landmarks and jump count on screen

## Build Commands

### Build the entire project
```bash
./gradlew build
```

### Build specific modules
```bash
./gradlew :app:assembleDebug
./gradlew :pose-detector-lib:assembleDebug
./gradlew :jumprope-counter-lib:assembleDebug
```

### Build and install to device
```bash
./gradlew installDebug
```

### Build release AAR files for libraries
```bash
./gradlew :pose-detector-lib:assembleRelease
./gradlew :jumprope-counter-lib:assembleRelease
```

### Clean build
```bash
./gradlew clean
```

### Run tests
```bash
./gradlew test
./gradlew connectedAndroidTest  # Requires connected device
```

### Build native libraries only
```bash
./gradlew :pose-detector-lib:externalNativeBuildDebug
./gradlew :jumprope-counter-lib:externalNativeBuildDebug
```

## Native Library Development

### Building C++ Code

Both native libraries use CMake. When you modify C++ files:

1. Changes to `.cpp` or `.h` files trigger automatic CMake rebuild
2. Changes to `CMakeLists.txt` require Gradle sync in Android Studio
3. Native libraries are built for both `arm64-v8a` and `armeabi-v7a`

### Debugging Native Code

Enable detailed logging in C++ files:
```cpp
// In JumpRopeCounter.cpp (lines 15-18)
static const bool ENABLE_LOGS = true;
static const bool ENABLE_DEBUG_LOGS = true;
static const bool ENABLE_DATA_LOGS = true;  // CSV format logs
```

View native logs:
```bash
adb logcat | grep "JumpRopeCounter"
adb logcat | grep "RknnRunner"
```

### Library Dependency Strategy

The app uses a fallback mechanism for native libraries (see `app/build.gradle:112-131`):
1. First tries to load prebuilt AAR from `app/libs/`
2. Falls back to source module if AAR not found

This allows distributing prebuilt binaries without exposing source code.

## Jump Rope Algorithm

### Recent Algorithm Fixes (2025-12-19)

The jump counting algorithm has undergone significant fixes documented in `ALGORITHM_FIX_SUMMARY.md`. Key improvements:

1. **Adaptive peak decay** - Only decays during air state, preventing false positives after rest periods
2. **Accurate jump height recording** - Tracks actual max lift per jump instead of envelope value
3. **Relaxed pose validation** - Tolerates 3 consecutive invalid frames before rejecting data
4. **Boundary checking** - Allows 10% screen-out-of-bounds tolerance

### Algorithm Parameters

Default thresholds in `JumpRopeCounter.cpp`:
- `minIntervalMs`: 300ms (minimum time between jumps)
- `upThresholdRatio`: 0.60 (takeoff threshold, 60% of adaptive peak)
- `downThresholdRatio`: 0.35 (landing threshold, 35% of adaptive peak)

Adjust via Kotlin API:
```kotlin
counter.setThresholds(upRatio = 0.60f, downRatio = 0.35f)
```

### State Machine

- `STATE_CALIBRATING (0)`: Initial 3-frame calibration
- `STATE_GROUND (1)`: On ground, waiting for takeoff
- `STATE_AIR (2)`: In air, waiting for landing

### Debugging Jump Counting Issues

See `DEBUG_GUIDE.md` for comprehensive troubleshooting steps. Quick checks:

- **False positives** (counting when not jumping): Increase `upThresholdRatio` or `minIntervalMs`
- **False negatives** (missing jumps): Decrease `upThresholdRatio`
- **Small jumps not detected**: Lower thresholds or check if peak decay is working correctly

## Model Files

### YOLO Pose Models

Place RKNN models in `app/src/main/assets/`:
- `yolo11n-pose-rk3568.rknn` (recommended)
- `yolov8n-pose-rk3568.rknn`
- Various quantization variants (fp16, int8, hybrid, mixed)

The Gradle build checks for model presence but does not auto-download.

### Model Conversion

Convert ONNX to RKNN format using RKNN-Toolkit2:
```python
from rknn.api import RKNN
rknn = RKNN()
rknn.config(target_platform='rk3588')
rkonnx('yolov8n-pose.onnx')
rknn.build(do_quantization=False)
rknn.export_rknn('yolov8n-pose.rknn')
```

## Testing

### Manual Testing Checklist

See `TEST_CHECKLIST.md` for comprehensive test scenarios. Key tests:

1. Normal jump rope counting (30 jumps)
2. Rest period handling (jump, rest 5s, jump again)
3. Small amplitude jumps
4. Fast jumping (2 jumps/second)
5. Partial body out of frame

### Test Script

```bash
./test_build.sh  # Automated build verification
```

## Common Issues

### Build Failures

**NDK not found**: Ensure Android NDK is installed via Android Studio SDK Mer

**CMake version**: Requires CMake 3.18.1 or higher

**Missing model files**: Check `app/src/main/assets/` for `.rknn` files

### Runtime Issues

**NPU not available**: Library falls back to CPU inference (slower)
- Check device has Rockchip SoC (RK3588/RK3568)
- Verify `librknnrt.so` matches chip version

**Inaccurate counting**:
- Check logcat for algorithm state transitions
- Verify pose detection quality (keypoint confidence > 0.3)
- Review `DEBUG_GUIDE.md` for parameter tuning

## Project-Specific Conventions

### Coordinate System

All pose coordinates are normalized to [0, 1]:
- (0, 0) = top-left corner
- (1, 1) = bottom-right corner
- Y-axis increases downward

### Keypoint Indices (YOLO Pose format)

Critical keypoints for jump counting:
- 5, 6: Left/Right Shoulder
- 11, 12: Left/Right Hip
- 15, 16: Left/Right Ankle

Full 17-keypoint layout documented in `pose-detector-lib/PoseDetector.md`

### Git Workflow

Recent commits show algorithm optimization focus:
- `8ea489a`: Temporary commit
- `9a22aa9`: Directory structure modification
- `0e54eaa`: Algorithm optimization
- `64c3c65`: Jump rope counting algorithm optimization

## Performance Targets

- **Pose detection**: ~30 FPS @ 640x640 on RK3588 NPU
- **Jump counting**: < 1ms per frame, < 1% CPU
- **Accuracy**: > 99% in normal jump rope scenarios

## Documentation Files

- `ALGORITHM_FIX_SUMMARY.md`: Detailed algorithm fix documentation
- `QUICK_REFERENCE.md`: Quick reference for algorithm fixes
- `DEBUG_GUIDE.md`: Troubleshooting guide for counting issues
- `TEST_CHECKLIST.md`: Comprehensive testing scenarios
- `SMALL_AMPLITUDE_FIX.md`: Specific fix for small jump detection
- `pose-detector-lib/PoseDetector.md`: Pose detection library API
- `jumprope-counter-lib/JumpRopeCounter.md`: Jump counter library API

## Hardware Requirements

- Android 7.0+ (API Level 24)
- arm64-v8a or armeabi-v7a architecture
- Rockchip SoC recommended for NPU acceleration (RK3588/RK3568/RK3566/RK3562)
- Camera with minimum 30 FPS capability
