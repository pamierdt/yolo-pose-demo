#include <jni.h>
#include <android/log.h>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>
#include <algorithm>
#include <utility>
#include <cmath>

#include "rknn_api.h"
#include "JumpRopeCounter.h"

#include <chrono>

#define LOG_TAG "rknn_jni"
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
static bool ENABLE_LOGS = true;

class Timer {
public:
    Timer(const char* name) : name_(name) {
        start_ = std::chrono::high_resolution_clock::now();
    }
    ~Timer() {
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start_).count();
        if (ENABLE_LOGS) LOGI("%s took %lld us", name_, duration);
    }
    long long get_duration_us() {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::microseconds>(end - start_).count();
    }
private:
    const char* name_;
    std::chrono::time_point<std::chrono::high_resolution_clock> start_;
};

namespace {

struct RknnHolder {
  rknn_context ctx = 0;
  rknn_tensor_attr input_attr{};
  std::vector<rknn_tensor_attr> output_attrs;
  bool logged_shape = false;
  
  // Pre-allocated buffers to avoid repeated allocation
  std::vector<uint16_t> fp16_lut;
  std::vector<uint16_t> fp16_input_buffer;
  std::vector<uint8_t> uint8_input_buffer;
};

// Simple float to half conversion for 0-255 integers
uint16_t FloatToHalf(float x) {
    uint32_t f = *((uint32_t*)&x);
    return ((f >> 16) & 0x8000) | ((((f & 0x7f800000) - 0x38000000) >> 13) & 0x7c00) | ((f >> 13) & 0x03ff);
}

void InitFp16Lut(RknnHolder* holder) {
  holder->fp16_lut.resize(256);
  for (int i = 0; i < 256; ++i) {
    holder->fp16_lut[i] = FloatToHalf(static_cast<float>(i));
  }
}

void ThrowIllegalState(JNIEnv* env, const std::string& msg) {
  jclass exClass = env->FindClass("java/lang/IllegalStateException");
  if (exClass) {
    env->ThrowNew(exClass, msg.c_str());
  }
}

RknnHolder* FromHandle(jlong handle) {
  return reinterpret_cast<RknnHolder*>(handle);
}

size_t ElemSize(rknn_tensor_type type) {
  switch (type) {
    case RKNN_TENSOR_FLOAT32:
    case RKNN_TENSOR_INT32:
    case RKNN_TENSOR_UINT32:
      return 4;
    case RKNN_TENSOR_FLOAT16:
    case RKNN_TENSOR_INT16:
    case RKNN_TENSOR_UINT16:
      return 2;
    case RKNN_TENSOR_INT8:
    case RKNN_TENSOR_UINT8:
    case RKNN_TENSOR_BOOL:
      return 1;
    default:
      return 4;
  }
}


struct PoseKeypoint {
    float x, y, score;
};

struct Pose {
    float score;
    float x, y, w, h;
    std::vector<PoseKeypoint> keypoints;
};

inline float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

float ComputeIoU(const Pose& a, const Pose& b) {
    float x1 = std::max(a.x - a.w / 2.0f, b.x - b.w / 2.0f);
    float y1 = std::max(a.y - a.h / 2.0f, b.y - b.h / 2.0f);
    float x2 = std::min(a.x + a.w / 2.0f, b.x + b.w / 2.0f);
    float y2 = std::min(a.y + a.h / 2.0f, b.y + b.h / 2.0f);

    if (x2 < x1 || y2 < y1) return 0.0f;

    float intersection = (x2 - x1) * (y2 - y1);
    float area_a = a.w * a.h;
    float area_b = b.w * b.h;

    return intersection / (area_a + area_b - intersection);
}

void NMS(std::vector<Pose>& poses, float iou_threshold) {
    std::sort(poses.begin(), poses.end(), [](const Pose& a, const Pose& b) {
        return a.score > b.score;
    });

    for (size_t i = 0; i < poses.size(); ++i) {
        if (poses[i].score == 0.0f) continue;
        for (size_t j = i + 1; j < poses.size(); ++j) {
            if (poses[j].score == 0.0f) continue;
            if (ComputeIoU(poses[i], poses[j]) > iou_threshold) {
                poses[j].score = 0.0f;
            }
        }
    }

    poses.erase(std::remove_if(poses.begin(), poses.end(), [](const Pose& p) {
        return p.score == 0.0f;
    }), poses.end());
}

}  // namespace

// Lightweight rope counter: EMA baseline + velocity gate to avoid per-frame Kalman math.
// 轻量级跳绳计数器：使用指数移动平均(EMA)基准线 + 速度门控，避免每帧都进行复杂的卡尔曼滤波计算。
// JumpRopeCounter logic moved to JumpRopeCounter.h/cpp

extern "C" JNIEXPORT jlong JNICALL
Java_com_google_mediapipe_examples_poselandmarker_JumpRopeCounter_nativeCreate(
    JNIEnv* env, jobject, jfloat minIntervalMs) {
  if (ENABLE_LOGS) LOGI("nativeCreate: minInt=%.2f", minIntervalMs);
  auto* obj = new JumpRopeCounter(minIntervalMs);
  return reinterpret_cast<jlong>(obj);
}

extern "C" JNIEXPORT void JNICALL
Java_com_google_mediapipe_examples_poselandmarker_JumpRopeCounter_nativeReset(
    JNIEnv* env, jobject, jlong handle) {
  auto* obj = reinterpret_cast<JumpRopeCounter*>(handle);
  if (!obj) return;
  if (ENABLE_LOGS) LOGI("nativeReset called for handle %lld", (long long)handle);
  obj->reset();
}

extern "C" JNIEXPORT jint JNICALL
Java_com_google_mediapipe_examples_poselandmarker_JumpRopeCounter_nativeUpdate(
    JNIEnv* env, jobject, jlong handle, jfloat shoulderY, jfloat hipY, jfloat ankleY, jdouble timestampMs) {
  auto* obj = reinterpret_cast<JumpRopeCounter*>(handle);
  if (!obj) {
    ThrowIllegalState(env, "JumpRopeCounter handle is null");
    return 0;
  }
  return static_cast<jint>(obj->update(shoulderY, hipY, ankleY, timestampMs));
}

extern "C" JNIEXPORT jint JNICALL
Java_com_google_mediapipe_examples_poselandmarker_JumpRopeCounter_nativeGetCount(
    JNIEnv* env, jobject, jlong handle) {
  auto* obj = reinterpret_cast<JumpRopeCounter*>(handle);
  if (!obj) {
    ThrowIllegalState(env, "JumpRopeCounter handle is null");
    return 0;
  }
  int count = obj->getCount();
  return static_cast<jint>(count);
}

extern "C" JNIEXPORT jfloat JNICALL
Java_com_google_mediapipe_examples_poselandmarker_JumpRopeCounter_nativeGetGroundY(
    JNIEnv* env, jobject, jlong handle) {
  auto* obj = reinterpret_cast<JumpRopeCounter*>(handle);
  if (!obj) {
    ThrowIllegalState(env, "JumpRopeCounter handle is null");
    return 0.f;
  }
  return static_cast<jfloat>(obj->getGroundY());
}

extern "C" JNIEXPORT jint JNICALL
Java_com_google_mediapipe_examples_poselandmarker_JumpRopeCounter_nativeGetState(
    JNIEnv* env, jobject, jlong handle) {
  auto* obj = reinterpret_cast<JumpRopeCounter*>(handle);
  if (!obj) {
    ThrowIllegalState(env, "JumpRopeCounter handle is null");
    return 0;
  }
  return static_cast<jint>(obj->getState());
}

extern "C" JNIEXPORT void JNICALL
Java_com_google_mediapipe_examples_poselandmarker_JumpRopeCounter_nativeRelease(
    JNIEnv* env, jobject, jlong handle) {
  auto* obj = reinterpret_cast<JumpRopeCounter*>(handle);
  if (!obj) return;
  delete obj;
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_google_mediapipe_examples_poselandmarker_RknnRunner_nativeInit(
    JNIEnv* env, jobject /*thiz*/, jobject model_buffer, jint model_size) {
  if (!model_buffer) {
    ThrowIllegalState(env, "Model buffer is null");
    return 0;
  }

  auto* buffer_ptr =
      static_cast<uint8_t*>(env->GetDirectBufferAddress(model_buffer));
  if (!buffer_ptr) {
    ThrowIllegalState(env, "Model buffer must be a direct ByteBuffer");
    return 0;
  }

  auto holder = std::make_unique<RknnHolder>();

  // Initialize FP16 LUT
  InitFp16Lut(holder.get());

  int ret = rknn_init(&holder->ctx, buffer_ptr, model_size, 0, nullptr);
  if (ret != RKNN_SUCC) {
    LOGE("rknn_init failed: %d", ret);
    ThrowIllegalState(env, "rknn_init failed: " + std::to_string(ret));
    return 0;
  }

  // Query input attr
  std::memset(&holder->input_attr, 0, sizeof(holder->input_attr));
  holder->input_attr.index = 0;
  ret = rknn_query(holder->ctx, RKNN_QUERY_INPUT_ATTR, &holder->input_attr,
                   sizeof(holder->input_attr));
  if (ret != RKNN_SUCC) {
    LOGE("rknn_query INPUT_ATTR failed: %d", ret);
    rknn_destroy(holder->ctx);
    ThrowIllegalState(env,
                      "rknn_query INPUT_ATTR failed: " + std::to_string(ret));
    return 0;
  }

  // Log input attr
  {
    std::string dims_str;
    for (uint32_t i = 0; i < holder->input_attr.n_dims; ++i) {
      dims_str.append(std::to_string(holder->input_attr.dims[i]));
      if (i + 1 < holder->input_attr.n_dims) dims_str.append("x");
    }
    if (ENABLE_LOGS) LOGI("RKNN input attr: n_dims=%u dims=%s n_elems=%u size=%u type=%d fmt=%d qnt_type=%d scale=%f zp=%d",
         holder->input_attr.n_dims, dims_str.c_str(), holder->input_attr.n_elems,
         holder->input_attr.size, holder->input_attr.type, holder->input_attr.fmt,
         holder->input_attr.qnt_type, holder->input_attr.scale, holder->input_attr.zp);
  }

  // Query input/output count
  rknn_input_output_num in_out_num{};
  ret = rknn_query(holder->ctx, RKNN_QUERY_IN_OUT_NUM, &in_out_num,
                   sizeof(in_out_num));
  if (ret != RKNN_SUCC || in_out_num.n_output == 0) {
    LOGE("rknn_query IN_OUT_NUM failed: %d", ret);
    rknn_destroy(holder->ctx);
    ThrowIllegalState(env, "rknn_query IN_OUT_NUM failed: " +
                                std::to_string(ret));
    return 0;
  }

  holder->output_attrs.resize(in_out_num.n_output);
  for (uint32_t i = 0; i < in_out_num.n_output; ++i) {
    std::memset(&holder->output_attrs[i], 0, sizeof(rknn_tensor_attr));
    holder->output_attrs[i].index = i;
    ret = rknn_query(holder->ctx, RKNN_QUERY_OUTPUT_ATTR,
                     &holder->output_attrs[i],
                     sizeof(holder->output_attrs[i]));
    if (ret != RKNN_SUCC) {
      LOGE("rknn_query OUTPUT_ATTR[%u] failed: %d", i, ret);
      rknn_destroy(holder->ctx);
      ThrowIllegalState(env, "rknn_query OUTPUT_ATTR failed: " +
                                  std::to_string(ret));
      return 0;
    }
  }

  if (ENABLE_LOGS) LOGI("RKNN init ok. Input dims: %u %u %u %u, output num: %u",
       holder->input_attr.dims[0], holder->input_attr.dims[1],
       holder->input_attr.dims[2], holder->input_attr.dims[3],
       in_out_num.n_output);

  return reinterpret_cast<jlong>(holder.release());
}

extern "C" JNIEXPORT jintArray JNICALL
Java_com_google_mediapipe_examples_poselandmarker_RknnRunner_nativeGetOutputShape(
    JNIEnv* env, jobject /*thiz*/, jlong handle) {
  auto* holder = FromHandle(handle);
  if (!holder || holder->output_attrs.empty()) {
    ThrowIllegalState(env, "RKNN handle is null or not initialized");
    return nullptr;
  }

  const rknn_tensor_attr& out_attr = holder->output_attrs[0];
  jintArray shape_array = env->NewIntArray(out_attr.n_dims);
  if (!shape_array) {
    ThrowIllegalState(env, "Failed to allocate shape array");
    return nullptr;
  }
  env->SetIntArrayRegion(shape_array, 0, out_attr.n_dims,
                         reinterpret_cast<const jint*>(out_attr.dims));
  return shape_array;
}

extern "C" JNIEXPORT jfloatArray JNICALL
Java_com_google_mediapipe_examples_poselandmarker_RknnRunner_nativeRun(
    JNIEnv* env, jobject /*thiz*/, jlong handle, jobject input_buffer,
    jint input_size) {
  auto* holder = FromHandle(handle);
  if (!holder) {
    ThrowIllegalState(env, "RKNN handle is null");
    return nullptr;
  }

  auto* input_ptr =
      static_cast<float*>(env->GetDirectBufferAddress(input_buffer));
  if (!input_ptr) {
    ThrowIllegalState(env, "Input buffer must be a direct ByteBuffer");
    return nullptr;
  }

  rknn_input inputs[1];
  std::memset(inputs, 0, sizeof(inputs));
  inputs[0].index = 0;
  // Match model expectation: use model's dtype/format (typically uint8 NHWC for RKNN quantized)
  inputs[0].type = holder->input_attr.type;
  inputs[0].fmt = holder->input_attr.fmt;
  inputs[0].size = input_size;
  inputs[0].pass_through = 0;
  inputs[0].buf = input_ptr;

  const size_t expected_size_bytes =
      holder->input_attr.n_elems * ElemSize(inputs[0].type);
  if (expected_size_bytes != static_cast<size_t>(input_size)) {
    LOGE("Input size mismatch: expected %zu (type %d), got %d",
         expected_size_bytes, inputs[0].type, input_size);
  }

  int ret = rknn_inputs_set(holder->ctx, 1, inputs);
  if (ret != RKNN_SUCC) {
    LOGE("rknn_inputs_set failed: %d", ret);
    ThrowIllegalState(env, "rknn_inputs_set failed: " + std::to_string(ret));
    return nullptr;
  }

  ret = rknn_run(holder->ctx, nullptr);
  if (ret != RKNN_SUCC) {
    LOGE("rknn_run failed: %d", ret);
    ThrowIllegalState(env, "rknn_run failed: " + std::to_string(ret));
    return nullptr;
  }

  const rknn_tensor_attr& out_attr = holder->output_attrs[0];
  const size_t out_elems = out_attr.n_elems;
  std::vector<float> output(out_elems, 0.f);

  std::vector<rknn_output> outputs(holder->output_attrs.size());
  for (size_t i = 0; i < outputs.size(); ++i) {
    outputs[i].want_float = 1;
    if (i == 0) {
      outputs[i].is_prealloc = 1;
      outputs[i].buf = output.data();
      outputs[i].size = out_elems * sizeof(float);
    } else {
      outputs[i].is_prealloc = 0;
      outputs[i].buf = nullptr;
      outputs[i].size = 0;
    }
  }

  ret = rknn_outputs_get(holder->ctx, static_cast<uint32_t>(outputs.size()),
                         outputs.data(), nullptr);
  if (ret != RKNN_SUCC) {
    LOGE("rknn_outputs_get failed: %d", ret);
    ThrowIllegalState(env,
                      "rknn_outputs_get failed: " + std::to_string(ret));
    return nullptr;
  }

  // Release non-prealloc outputs if any
  rknn_outputs_release(holder->ctx, static_cast<uint32_t>(outputs.size()),
                       outputs.data());

  if (ENABLE_LOGS && !output.empty()) {
    const size_t sample = std::min<size_t>(output.size(), 16);
    std::string buf;
    buf.reserve(sample * 8);
    for (size_t i = 0; i < sample; ++i) {
      char tmp[64];
      std::snprintf(tmp, sizeof(tmp), "%.4f,", output[i]);
      buf.append(tmp);
    }
    if (!holder->logged_shape) {
      const rknn_tensor_attr& attr0 = holder->output_attrs[0];
      std::string dims_str;
      for (uint32_t i = 0; i < attr0.n_dims; ++i) {
        dims_str.append(std::to_string(attr0.dims[i]));
        if (i + 1 < attr0.n_dims) dims_str.append("x");
      }
    if (ENABLE_LOGS) LOGI("RKNN output[0] attr: n_dims=%u dims=%s n_elems=%u size=%u type=%d qnt_type=%d",
           attr0.n_dims, dims_str.c_str(), attr0.n_elems, attr0.size,
           attr0.type, attr0.qnt_type);
      holder->logged_shape = true;
    }
    if (ENABLE_LOGS) LOGI("RKNN output[0] size=%zu sample(first %zu): %s", output.size(), sample, buf.c_str());
  }

  jfloatArray result = env->NewFloatArray(static_cast<jsize>(out_elems));
  if (!result) {
    ThrowIllegalState(env, "Failed to allocate output array");
    return nullptr;
  }
  env->SetFloatArrayRegion(result, 0, static_cast<jsize>(out_elems),
                           output.data());
  return result;
}

extern "C" JNIEXPORT jfloatArray JNICALL
Java_com_google_mediapipe_examples_poselandmarker_RknnRunner_nativeRunPixels(
    JNIEnv* env, jobject /*thiz*/, jlong handle, jintArray pixels) {
  auto* holder = FromHandle(handle);
  if (!holder) {
    ThrowIllegalState(env, "RKNN handle is null");
    return nullptr;
  }

  jsize pixel_count = env->GetArrayLength(pixels);
  jint* pixel_ptr = env->GetIntArrayElements(pixels, nullptr);
  if (!pixel_ptr) {
    ThrowIllegalState(env, "Failed to get pixel array elements");
    return nullptr;
  }

  // Determine target input type and resize buffer
  rknn_tensor_type input_type = holder->input_attr.type;
  size_t num_elements = pixel_count * 3; // RGB
  
  void* input_buf = nullptr;
  size_t input_size_bytes = 0;

  if (input_type == RKNN_TENSOR_FLOAT16) {
      if (holder->fp16_input_buffer.size() != num_elements) {
          holder->fp16_input_buffer.resize(num_elements);
      }
      
      const uint16_t* lut = holder->fp16_lut.data();
      uint16_t* dst = holder->fp16_input_buffer.data();
      
      // Convert ARGB int to FP16 RGB using LUT
      for (int i = 0; i < pixel_count; ++i) {
          jint p = pixel_ptr[i];
          dst[i * 3 + 0] = lut[(p >> 16) & 0xFF]; // R
          dst[i * 3 + 1] = lut[(p >> 8) & 0xFF];  // G
          dst[i * 3 + 2] = lut[p & 0xFF];         // B
      }
      input_buf = holder->fp16_input_buffer.data();
      input_size_bytes = num_elements * 2;
  } else if (input_type == RKNN_TENSOR_UINT8) {
      // Direct UINT8 pass-through for quantized models
      // pixels are ARGB_8888 (int), need to extract RGB bytes
      if (holder->uint8_input_buffer.size() != num_elements) {
          holder->uint8_input_buffer.resize(num_elements);
      }
      uint8_t* dst = holder->uint8_input_buffer.data();
      for (int i = 0; i < pixel_count; ++i) {
          jint p = pixel_ptr[i];
          dst[i * 3 + 0] = static_cast<uint8_t>((p >> 16) & 0xFF);
          dst[i * 3 + 1] = static_cast<uint8_t>((p >> 8) & 0xFF);
          dst[i * 3 + 2] = static_cast<uint8_t>(p & 0xFF);
      }
      input_buf = holder->uint8_input_buffer.data();
      input_size_bytes = num_elements;
  } else {
       ThrowIllegalState(env, "Unsupported input type for runPixels: " + std::to_string(input_type));
       env->ReleaseIntArrayElements(pixels, pixel_ptr, JNI_ABORT);
       return nullptr;
  }

  env->ReleaseIntArrayElements(pixels, pixel_ptr, JNI_ABORT);

  // Set inputs
  rknn_input inputs[1];
  std::memset(inputs, 0, sizeof(inputs));
  inputs[0].index = 0;
  inputs[0].type = input_type;
  inputs[0].fmt = holder->input_attr.fmt;
  inputs[0].size = static_cast<uint32_t>(input_size_bytes);
  inputs[0].pass_through = 0;
  inputs[0].buf = input_buf;

  int ret = rknn_inputs_set(holder->ctx, 1, inputs);
  if (ret != RKNN_SUCC) {
    LOGE("rknn_inputs_set failed: %d", ret);
    ThrowIllegalState(env, "rknn_inputs_set failed: " + std::to_string(ret));
    return nullptr;
  }

  ret = rknn_run(holder->ctx, nullptr);
  if (ret != RKNN_SUCC) {
    LOGE("rknn_run failed: %d", ret);
    ThrowIllegalState(env, "rknn_run failed: " + std::to_string(ret));
    return nullptr;
  }

  // Get outputs
  const rknn_tensor_attr& out_attr = holder->output_attrs[0];
  const size_t out_elems = out_attr.n_elems;
  std::vector<float> output(out_elems, 0.f);

  std::vector<rknn_output> outputs(holder->output_attrs.size());
  for (size_t i = 0; i < outputs.size(); ++i) {
    outputs[i].want_float = 1;
    if (i == 0) {
      outputs[i].is_prealloc = 1;
      outputs[i].buf = output.data();
      outputs[i].size = out_elems * sizeof(float);
    } else {
      outputs[i].is_prealloc = 0;
      outputs[i].buf = nullptr;
      outputs[i].size = 0;
    }
  }

  ret = rknn_outputs_get(holder->ctx, static_cast<uint32_t>(outputs.size()),
                         outputs.data(), nullptr);
  if (ret != RKNN_SUCC) {
    LOGE("rknn_outputs_get failed: %d", ret);
    ThrowIllegalState(env,
                      "rknn_outputs_get failed: " + std::to_string(ret));
    return nullptr;
  }

  // Release non-prealloc outputs if any
  rknn_outputs_release(holder->ctx, static_cast<uint32_t>(outputs.size()),
                       outputs.data());

  jfloatArray result = env->NewFloatArray(static_cast<jsize>(out_elems));
  if (!result) {
    ThrowIllegalState(env, "Failed to allocate output array");
    return nullptr;
  }
  env->SetFloatArrayRegion(result, 0, static_cast<jsize>(out_elems),
                           output.data());
  return result;
}

extern "C" JNIEXPORT void JNICALL
Java_com_google_mediapipe_examples_poselandmarker_RknnRunner_nativeRelease(
    JNIEnv* /*env*/, jobject /*thiz*/, jlong handle) {
  auto* holder = FromHandle(handle);
  if (!holder) return;
  if (holder->ctx != 0) {
    rknn_destroy(holder->ctx);
  }
  delete holder;
}

#include <android/bitmap.h>

// ...

inline float inverse_sigmoid(float y) {
    if (y <= 0.0001f) return -10.0f;
    if (y >= 0.9999f) return 10.0f;
    return -std::log(1.0f/y - 1.0f);
}

extern "C" JNIEXPORT jfloatArray JNICALL
Java_com_google_mediapipe_examples_poselandmarker_RknnRunner_nativeRunBitmapWithNms(
    JNIEnv* env, jobject /*thiz*/, jlong handle, jobject bitmap, jfloat detectThresh, jfloat nmsThresh) {
  auto* holder = FromHandle(handle);
  if (!holder) {
    ThrowIllegalState(env, "RKNN handle is null");
    return nullptr;
  }

  auto t_start = std::chrono::high_resolution_clock::now();

  AndroidBitmapInfo info;
  if (AndroidBitmap_getInfo(env, bitmap, &info) < 0) {
    ThrowIllegalState(env, "AndroidBitmap_getInfo failed");
    return nullptr;
  }
  if (ENABLE_LOGS) LOGI("Bitmap info: %u x %u stride=%u format=%d", info.width, info.height, info.stride, info.format);
  
  if (info.format != ANDROID_BITMAP_FORMAT_RGBA_8888) {
    ThrowIllegalState(env, "Bitmap format must be ARGB_8888");
    return nullptr;
  }

  void* pixels_ptr = nullptr;
  if (AndroidBitmap_lockPixels(env, bitmap, &pixels_ptr) < 0) {
    ThrowIllegalState(env, "AndroidBitmap_lockPixels failed");
    return nullptr;
  }

  auto t_lock = std::chrono::high_resolution_clock::now();

  rknn_tensor_type input_type = holder->input_attr.type;
  size_t num_pixels = info.width * info.height;
  size_t num_elements = num_pixels * 3; 
  
  void* input_buf = nullptr;
  size_t input_size_bytes = 0;

  // Assuming input size matches bitmap size, otherwise we need scaling (not handled here)
  
  if (input_type == RKNN_TENSOR_FLOAT16) {
      if (holder->fp16_input_buffer.size() != num_elements) {
          holder->fp16_input_buffer.resize(num_elements);
      }
      const uint16_t* lut = holder->fp16_lut.data();
      uint16_t* dst = holder->fp16_input_buffer.data();
      
      uint8_t* src_rows = static_cast<uint8_t*>(pixels_ptr);
      int dst_idx = 0;
      
      for (int y = 0; y < info.height; ++y) {
          uint8_t* src_pixel = src_rows + y * info.stride;
          for (int x = 0; x < info.width; ++x) {
             // Memory: B G R A
             uint8_t b = src_pixel[0];
             uint8_t g = src_pixel[1];
             uint8_t r = src_pixel[2];
             
             dst[dst_idx++] = lut[r];
             dst[dst_idx++] = lut[g];
             dst[dst_idx++] = lut[b];
             
             src_pixel += 4;
          }
      }
      input_buf = holder->fp16_input_buffer.data();
      input_size_bytes = num_elements * 2;
  } else if (input_type == RKNN_TENSOR_UINT8) {
      if (holder->uint8_input_buffer.size() != num_elements) {
          holder->uint8_input_buffer.resize(num_elements);
      }
      uint8_t* dst = holder->uint8_input_buffer.data();
      
      uint8_t* src_rows = static_cast<uint8_t*>(pixels_ptr);
      int dst_idx = 0;
      
      for (int y = 0; y < info.height; ++y) {
          uint8_t* src_pixel = src_rows + y * info.stride;
          for (int x = 0; x < info.width; ++x) {
             // Memory: B G R A
             uint8_t b = src_pixel[0];
             uint8_t g = src_pixel[1];
             uint8_t r = src_pixel[2];
             
             dst[dst_idx++] = r;
             dst[dst_idx++] = g;
             dst[dst_idx++] = b;
             
             src_pixel += 4;
          }
      }
      input_buf = holder->uint8_input_buffer.data();
      input_size_bytes = num_elements;
  } else {
       AndroidBitmap_unlockPixels(env, bitmap);
       ThrowIllegalState(env, "Unsupported input type for runPixels: " + std::to_string(input_type));
       return nullptr;
  }
  
  AndroidBitmap_unlockPixels(env, bitmap);

  auto t_prep = std::chrono::high_resolution_clock::now();
  if (ENABLE_LOGS) LOGI("Prep done: input type=%d elements=%zu bytes=%zu", input_type, num_elements, input_size_bytes);

  rknn_input inputs[1];
  std::memset(inputs, 0, sizeof(inputs));
  inputs[0].index = 0;
  inputs[0].type = input_type;
  inputs[0].fmt = holder->input_attr.fmt;
  inputs[0].size = static_cast<uint32_t>(input_size_bytes);
  inputs[0].buf = input_buf;

  int ret = rknn_inputs_set(holder->ctx, 1, inputs);
  if (ret != RKNN_SUCC) {
    ThrowIllegalState(env, "rknn_inputs_set failed: " + std::to_string(ret));
    return nullptr;
  }

  auto t_set = std::chrono::high_resolution_clock::now();

  ret = rknn_run(holder->ctx, nullptr);
  if (ret != RKNN_SUCC) {
    ThrowIllegalState(env, "rknn_run failed: " + std::to_string(ret));
    return nullptr;
  }

  auto t_run = std::chrono::high_resolution_clock::now();

  const rknn_tensor_attr& out_attr = holder->output_attrs[0];
  const size_t out_elems = out_attr.n_elems;
  std::vector<float> output(out_elems);
  rknn_output rknn_outputs[1];
  std::memset(rknn_outputs, 0, sizeof(rknn_outputs));
  rknn_outputs[0].want_float = 1;
  rknn_outputs[0].is_prealloc = 1;
  rknn_outputs[0].buf = output.data();
  rknn_outputs[0].size = out_elems * sizeof(float);

  ret = rknn_outputs_get(holder->ctx, 1, rknn_outputs, nullptr);
  if (ret != RKNN_SUCC) {
    ThrowIllegalState(env, "rknn_outputs_get failed: " + std::to_string(ret));
    return nullptr;
  }
  rknn_outputs_release(holder->ctx, 1, rknn_outputs);

  auto t_get = std::chrono::high_resolution_clock::now();

  // --- End of inference logic ---

  // --- Start of Post-Processing ---
  // outputShape: [1, 56, 8400] or [1, 8400, 56]
  int dims[4] = {0};
  for(int i=0; i<out_attr.n_dims; ++i) dims[i] = out_attr.dims[i];

  bool channelFirst = false;
  int candidateCount = 0;
  int valuesPerCandidate = 0;

  if (out_attr.n_dims == 3) {
      if (dims[1] < dims[2]) {
          channelFirst = true;
          valuesPerCandidate = dims[1];
          candidateCount = dims[2];
      } else {
          channelFirst = false;
          candidateCount = dims[1];
          valuesPerCandidate = dims[2];
      }
  } else if (out_attr.n_dims == 2) {
       if (dims[0] > dims[1]) {
           candidateCount = dims[0];
           valuesPerCandidate = dims[1];
           channelFirst = false; 
       } else {
           valuesPerCandidate = dims[0];
           candidateCount = dims[1];
           channelFirst = true; 
       }
  }

  if (candidateCount == 0 || valuesPerCandidate < 5) {
       return env->NewFloatArray(0);
  }
  if (ENABLE_LOGS) LOGI("Output: n_dims=%u dims=%d %d %d %d channelFirst=%d candidates=%d values=%d", out_attr.n_dims, dims[0], dims[1], dims[2], dims[3], channelFirst ? 1 : 0, candidateCount, valuesPerCandidate);

  int numKeypoints = (valuesPerCandidate - 5) / 3;
  std::vector<Pose> candidates;
  candidates.reserve(100); 

  // Assuming NHWC input layout for width/height retrieval if needed
  // Using dims[2] as width (commonly 1, H, W, C)
  float inputSize = static_cast<float>(holder->input_attr.dims[2]);
  if (inputSize == 0) inputSize = 640.0f; 
  
  float rawDetectThresh = inverse_sigmoid(detectThresh);

  std::vector<std::pair<float,int>> passed;
  passed.reserve(candidateCount);
  for (int i = 0; i < candidateCount; ++i) {
      float rawScore = channelFirst ? output[4 * candidateCount + i] : output[i * valuesPerCandidate + 4];
      if (rawScore >= rawDetectThresh) passed.emplace_back(rawScore, i);
  }
  int topk = static_cast<int>(passed.size());
  if (topk > 300) {
      topk = 300;
      std::nth_element(passed.begin(), passed.begin() + topk, passed.end(), [](const std::pair<float,int>& a, const std::pair<float,int>& b){ return a.first > b.first; });
      passed.resize(topk);
  }

  if (ENABLE_LOGS) LOGI("Decode: candidates=%d, passed=%zu, topk=%d", candidateCount, passed.size(), topk);

  for (int idx = 0; idx < topk; ++idx) {
      int i = passed[idx].second;
      float rawScore = passed[idx].first;
      float score = sigmoid(rawScore);

      Pose pose;
      pose.score = score;

      float cx = channelFirst ? output[0 * candidateCount + i] : output[i * valuesPerCandidate + 0];
      float cy = channelFirst ? output[1 * candidateCount + i] : output[i * valuesPerCandidate + 1];
      float w = channelFirst ? output[2 * candidateCount + i] : output[i * valuesPerCandidate + 2];
      float h = channelFirst ? output[3 * candidateCount + i] : output[i * valuesPerCandidate + 3];

      if (w > 2.0f) {
          cx /= inputSize;
          cy /= inputSize;
          w /= inputSize;
          h /= inputSize;
      }

      pose.x = cx;
      pose.y = cy;
      pose.w = w;
      pose.h = h;

      for (int k = 0; k < numKeypoints; ++k) {
          float kx = channelFirst ? output[(5 + k*3) * candidateCount + i] : output[i * valuesPerCandidate + (5 + k*3)];
          float ky = channelFirst ? output[(5 + k*3 + 1) * candidateCount + i] : output[i * valuesPerCandidate + (5 + k*3 + 1)];
          float ks = channelFirst ? output[(5 + k*3 + 2) * candidateCount + i] : output[i * valuesPerCandidate + (5 + k*3 + 2)];

          if (kx > 2.0f) {
              kx /= inputSize;
              ky /= inputSize;
          }

          pose.keypoints.push_back({kx, ky, sigmoid(ks)});
      }
      candidates.push_back(pose);
  }

  if (ENABLE_LOGS) LOGI("Before NMS: %zu candidates", candidates.size());
  NMS(candidates, nmsThresh);
  if (ENABLE_LOGS) LOGI("After NMS: %zu", candidates.size());

  // Serialize results
  // Format: [num_poses, score, x, y, w, h, kp1_x, kp1_y, kp1_s, ..., score2...]
  int poseSize = 5 + numKeypoints * 3;
  std::vector<float> resultData;
  resultData.reserve(1 + candidates.size() * poseSize);
  
  resultData.push_back(static_cast<float>(candidates.size()));
  
  for (const auto& p : candidates) {
      resultData.push_back(p.score);
      resultData.push_back(p.x);
      resultData.push_back(p.y);
      resultData.push_back(p.w);
      resultData.push_back(p.h);
      for (const auto& kp : p.keypoints) {
          resultData.push_back(kp.x);
          resultData.push_back(kp.y);
          resultData.push_back(kp.score);
      }
  }

  jfloatArray resultArray = env->NewFloatArray(resultData.size());
  env->SetFloatArrayRegion(resultArray, 0, resultData.size(), resultData.data());
  
  auto t_post = std::chrono::high_resolution_clock::now();

  long long us_prep = std::chrono::duration_cast<std::chrono::microseconds>(t_prep - t_start).count();
  long long us_set = std::chrono::duration_cast<std::chrono::microseconds>(t_set - t_prep).count();
  long long us_run = std::chrono::duration_cast<std::chrono::microseconds>(t_run - t_set).count();
  long long us_get = std::chrono::duration_cast<std::chrono::microseconds>(t_get - t_run).count();
  long long us_post = std::chrono::duration_cast<std::chrono::microseconds>(t_post - t_get).count();
  long long us_total = std::chrono::duration_cast<std::chrono::microseconds>(t_post - t_start).count();

  if (ENABLE_LOGS) LOGI("Profile: Prep=%lld us, Set=%lld us, Run=%lld us, Get=%lld us, Post=%lld us, Total=%lld us", 
       us_prep, us_set, us_run, us_get, us_post, us_total);

  return resultArray;
}
