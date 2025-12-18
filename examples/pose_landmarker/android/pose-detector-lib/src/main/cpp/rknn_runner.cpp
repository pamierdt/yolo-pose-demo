#include <algorithm>
#include <android/log.h>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <jni.h>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "rknn_api.h"

// RGA headers for hardware-accelerated image preprocessing
// Manually define RGA version to avoid circular dependency
#define RGA_API_MAJOR_VERSION 1
#define RGA_API_MINOR_VERSION 10
#define RGA_API_REVISION_VERSION 0
#define RGA_API_BUILD_VERSION 2
#define RGA_CURRENT_API_VERSION                                                \
  ((RGA_API_MAJOR_VERSION & 0xff) << 24 |                                      \
   (RGA_API_MINOR_VERSION & 0xff) << 16 |                                      \
   (RGA_API_REVISION_VERSION & 0xff) << 8 | (RGA_API_BUILD_VERSION & 0xff))
#define RGA_CURRENT_API_HEADER_VERSION RGA_CURRENT_API_VERSION

// Now include RGA headers
#include "im2d_buffer.h"
#include "im2d_single.h"
#include "im2d_type.h"
#include "rga.h"

#include <chrono>

#define LOG_TAG "rknn_jni"
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
static bool ENABLE_LOGS = true;
static bool g_force_quant_output = false; // 输出采用量化值，手动反量化
static bool g_input_mem_cacheable = true; // 输入缓冲是否使用可缓存分配

class Timer {
public:
  Timer(const char *name) : name_(name) {
    start_ = std::chrono::high_resolution_clock::now();
  }
  ~Timer() {
    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start_)
            .count();
    if (ENABLE_LOGS)
      LOGI("%s took %lld us", name_, duration);
  }
  long long get_duration_us() {
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(end - start_)
        .count();
  }

private:
  const char *name_;
  std::chrono::time_point<std::chrono::high_resolution_clock> start_;
};

namespace {

struct RknnHolder {
  rknn_context ctx = 0;
  rknn_tensor_attr input_attr{};
  std::vector<rknn_tensor_attr> output_attrs;
  bool logged_shape = false;

  // RGA hardware acceleration state
  bool rga_initialized = false;
  bool rga_available = true; // Fallback flag if RGA initialization fails

  // Pre-allocated buffers to avoid repeated allocation
  std::vector<uint16_t> fp16_lut;
  std::vector<uint16_t> fp16_input_buffer;
  std::vector<uint8_t> uint8_input_buffer;

  // RKNN zero-copy input buffer
  rknn_tensor_mem *input_mem = nullptr;
  size_t input_tensor_bytes = 0;

  // Reusable quantized output buffer when want_float=0
  std::vector<uint8_t> quant_output_buffer;

  bool zero_copy_warning_shown = false;
};

// Simple float to half conversion for 0-255 integers
uint16_t FloatToHalf(float x) {
  uint32_t f = *((uint32_t *)&x);
  return ((f >> 16) & 0x8000) |
         ((((f & 0x7f800000) - 0x38000000) >> 13) & 0x7c00) |
         ((f >> 13) & 0x03ff);
}

void InitFp16Lut(RknnHolder *holder) {
  holder->fp16_lut.resize(256);
  for (int i = 0; i < 256; ++i) {
    holder->fp16_lut[i] = FloatToHalf(static_cast<float>(i));
  }
}

void ThrowIllegalState(JNIEnv *env, const std::string &msg) {
  jclass exClass = env->FindClass("java/lang/IllegalStateException");
  if (exClass) {
    env->ThrowNew(exClass, msg.c_str());
  }
}

RknnHolder *FromHandle(jlong handle) {
  return reinterpret_cast<RknnHolder *>(handle);
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

// Validate RKNN-managed input buffer size/bounds.
// Note: RKNN runtime 1.5.2 (librknnrt.so) doesn't expose rknn_mem_sync(),
// so this is a no-op beyond sanity checks for compatibility.
bool SyncInputTensorMem(RknnHolder *holder, size_t used_bytes) {
  if (!holder->input_mem)
    return true;
  if (!holder->input_mem->virt_addr || holder->input_tensor_bytes == 0) {
    LOGE("Zero-copy input buffer is not initialized");
    return false;
  }
  if (used_bytes > holder->input_tensor_bytes) {
    LOGE("Zero-copy input overflow: used=%zu, capacity=%zu", used_bytes,
         holder->input_tensor_bytes);
    return false;
  }
  return true;
}

void DequantizeToFloat(const rknn_tensor_attr &attr, const uint8_t *src,
                       std::vector<float> &dst) {
  dst.resize(attr.n_elems);
  float scale = attr.scale;
  int zp = attr.zp;
  for (uint32_t i = 0; i < attr.n_elems; ++i) {
    int q = (attr.type == RKNN_TENSOR_UINT8)
                ? static_cast<int>(src[i])
                : static_cast<int>(static_cast<int8_t>(src[i]));
    dst[i] = (static_cast<int>(q) - zp) * scale;
  }
}

// Ensure zero-copy writes match RKNN stride expectations, otherwise disable
bool IsZeroCopyStrideCompatible(const RknnHolder *holder,
                                size_t dst_bytes_with_stride) {
  if (!holder->input_mem)
    return false;
  if (dst_bytes_with_stride > holder->input_tensor_bytes)
    return false;
  if (holder->input_attr.size_with_stride != 0 &&
      dst_bytes_with_stride != holder->input_attr.size_with_stride)
    return false;
  return true;
}

struct PoseKeypoint {
  float x, y, score;
};

struct Pose {
  float score;
  float x, y, w, h;
  std::vector<PoseKeypoint> keypoints;
};

inline float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

float ComputeIoU(const Pose &a, const Pose &b) {
  float x1 = std::max(a.x - a.w / 2.0f, b.x - b.w / 2.0f);
  float y1 = std::max(a.y - a.h / 2.0f, b.y - b.h / 2.0f);
  float x2 = std::min(a.x + a.w / 2.0f, b.x + b.w / 2.0f);
  float y2 = std::min(a.y + a.h / 2.0f, b.y + b.h / 2.0f);

  if (x2 < x1 || y2 < y1)
    return 0.0f;

  float intersection = (x2 - x1) * (y2 - y1);
  float area_a = a.w * a.h;
  float area_b = b.w * b.h;

  return intersection / (area_a + area_b - intersection);
}

void NMS(std::vector<Pose> &poses, float iou_threshold) {
  std::sort(poses.begin(), poses.end(),
            [](const Pose &a, const Pose &b) { return a.score > b.score; });

  for (size_t i = 0; i < poses.size(); ++i) {
    if (poses[i].score == 0.0f)
      continue;
    for (size_t j = i + 1; j < poses.size(); ++j) {
      if (poses[j].score == 0.0f)
        continue;
      if (ComputeIoU(poses[i], poses[j]) > iou_threshold) {
        poses[j].score = 0.0f;
      }
    }
  }

  poses.erase(std::remove_if(poses.begin(), poses.end(),
                             [](const Pose &p) { return p.score == 0.0f; }),
              poses.end());
}

} // namespace

// Lightweight rope counter: EMA baseline + velocity gate to avoid per-frame
// Kalman math. 轻量级跳绳计数器：使用指数移动平均(EMA)基准线 +
// 速度门控，避免每帧都进行复杂的卡尔曼滤波计算。 JumpRopeCounter logic moved to
// JumpRopeCounter.h/cpp
extern "C" JNIEXPORT void JNICALL
Java_com_yolo_pose_detector_RknnRunner_nativeSetPerfOptions(
    JNIEnv * /*env*/, jobject /*thiz*/, jboolean useQuantOutput,
    jboolean cacheableInput) {
  g_force_quant_output = (useQuantOutput == JNI_TRUE);
  g_input_mem_cacheable = (cacheableInput == JNI_TRUE);
  if (ENABLE_LOGS)
    LOGI("PerfOptions: quant_output=%d cacheable_input=%d",
         g_force_quant_output ? 1 : 0, g_input_mem_cacheable ? 1 : 0);
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_yolo_pose_detector_RknnRunner_nativeInit(JNIEnv *env, jobject /*thiz*/,
                                                  jobject model_buffer,
                                                  jint model_size) {
  if (!model_buffer) {
    ThrowIllegalState(env, "Model buffer is null");
    return 0;
  }

  auto *buffer_ptr =
      static_cast<uint8_t *>(env->GetDirectBufferAddress(model_buffer));
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

  // Prefer all NPU cores; harmless on single-core parts
  ret = rknn_set_core_mask(holder->ctx, RKNN_NPU_CORE_AUTO);
  if (ret != RKNN_SUCC && ENABLE_LOGS) {
    LOGE("rknn_set_core_mask failed: %d", ret);
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
      if (i + 1 < holder->input_attr.n_dims)
        dims_str.append("x");
    }
    if (ENABLE_LOGS)
      LOGI("RKNN input attr: n_dims=%u dims=%s n_elems=%u size=%u type=%d "
           "fmt=%d qnt_type=%d scale=%f zp=%d",
           holder->input_attr.n_dims, dims_str.c_str(),
           holder->input_attr.n_elems, holder->input_attr.size,
           holder->input_attr.type, holder->input_attr.fmt,
           holder->input_attr.qnt_type, holder->input_attr.scale,
           holder->input_attr.zp);
  }

  // Query input/output count
  rknn_input_output_num in_out_num{};
  ret = rknn_query(holder->ctx, RKNN_QUERY_IN_OUT_NUM, &in_out_num,
                   sizeof(in_out_num));
  if (ret != RKNN_SUCC || in_out_num.n_output == 0) {
    LOGE("rknn_query IN_OUT_NUM failed: %d", ret);
    rknn_destroy(holder->ctx);
    ThrowIllegalState(env,
                      "rknn_query IN_OUT_NUM failed: " + std::to_string(ret));
    return 0;
  }

  holder->output_attrs.resize(in_out_num.n_output);
  for (uint32_t i = 0; i < in_out_num.n_output; ++i) {
    std::memset(&holder->output_attrs[i], 0, sizeof(rknn_tensor_attr));
    holder->output_attrs[i].index = i;
    ret = rknn_query(holder->ctx, RKNN_QUERY_OUTPUT_ATTR,
                     &holder->output_attrs[i], sizeof(holder->output_attrs[i]));
    if (ret != RKNN_SUCC) {
      LOGE("rknn_query OUTPUT_ATTR[%u] failed: %d", i, ret);
      rknn_destroy(holder->ctx);
      ThrowIllegalState(env, "rknn_query OUTPUT_ATTR failed: " +
                                 std::to_string(ret));
      return 0;
    }
  }

  if (ENABLE_LOGS)
    LOGI("RKNN init ok. Input dims: %u %u %u %u, output num: %u",
         holder->input_attr.dims[0], holder->input_attr.dims[1],
         holder->input_attr.dims[2], holder->input_attr.dims[3],
         in_out_num.n_output);

  holder->input_tensor_bytes = holder->input_attr.size_with_stride != 0
                                   ? holder->input_attr.size_with_stride
                                   : holder->input_attr.size;
  if (holder->input_tensor_bytes == 0) {
    holder->input_tensor_bytes =
        holder->input_attr.n_elems * ElemSize(holder->input_attr.type);
  }

  // RKNN 1.5.2 only provides rknn_create_mem(); newer APIs like rknn_create_mem2
  // are not available in the downgraded runtime.
  holder->input_mem =
      rknn_create_mem(holder->ctx,
                      static_cast<uint32_t>(holder->input_tensor_bytes));
  if (!holder->input_mem) {
    LOGE("rknn_create_mem failed, zero-copy disabled");
  } else {
    holder->input_attr.pass_through = 0;
    ret = rknn_set_io_mem(holder->ctx, holder->input_mem, &holder->input_attr);
    if (ret != RKNN_SUCC) {
      LOGE("rknn_set_io_mem failed: %d", ret);
      rknn_destroy_mem(holder->ctx, holder->input_mem);
      holder->input_mem = nullptr;
      holder->input_tensor_bytes = 0;
    } else {
      holder->rga_initialized = true;
      if (ENABLE_LOGS)
        LOGI("Zero-copy input ready: phys=%p, bytes=%zu, w_stride=%u (create_mem)",
             reinterpret_cast<void *>(holder->input_mem->phys_addr),
             holder->input_tensor_bytes, holder->input_attr.w_stride);
    }
  }

  return reinterpret_cast<jlong>(holder.release());
}

extern "C" JNIEXPORT jintArray JNICALL
Java_com_yolo_pose_detector_RknnRunner_nativeGetOutputShape(JNIEnv *env,
                                                            jobject /*thiz*/,
                                                            jlong handle) {
  auto *holder = FromHandle(handle);
  if (!holder || holder->output_attrs.empty()) {
    ThrowIllegalState(env, "RKNN handle is null or not initialized");
    return nullptr;
  }

  const rknn_tensor_attr &out_attr = holder->output_attrs[0];
  jintArray shape_array = env->NewIntArray(out_attr.n_dims);
  if (!shape_array) {
    ThrowIllegalState(env, "Failed to allocate shape array");
    return nullptr;
  }
  env->SetIntArrayRegion(shape_array, 0, out_attr.n_dims,
                         reinterpret_cast<const jint *>(out_attr.dims));
  return shape_array;
}

extern "C" JNIEXPORT jfloatArray JNICALL
Java_com_yolo_pose_detector_RknnRunner_nativeRun(JNIEnv *env, jobject /*thiz*/,
                                                 jlong handle,
                                                 jobject input_buffer,
                                                 jint input_size) {
  auto t_start = std::chrono::high_resolution_clock::now();
  auto *holder = FromHandle(handle);
  if (!holder) {
    ThrowIllegalState(env, "RKNN handle is null");
    return nullptr;
  }

  auto *input_ptr =
      static_cast<float *>(env->GetDirectBufferAddress(input_buffer));
  if (!input_ptr) {
    ThrowIllegalState(env, "Input buffer must be a direct ByteBuffer");
    return nullptr;
  }

  const size_t expected_size_bytes =
      holder->input_attr.n_elems * ElemSize(holder->input_attr.type);
  if (expected_size_bytes != static_cast<size_t>(input_size)) {
    LOGE("Input size mismatch: expected %zu (type %d), got %d",
         expected_size_bytes, holder->input_attr.type, input_size);
  }

  bool use_zero_copy = holder->input_mem &&
                       expected_size_bytes <= holder->input_tensor_bytes &&
                       holder->input_mem->virt_addr != nullptr;
  if (use_zero_copy) {
    size_t copy_size = static_cast<size_t>(input_size);
    if (copy_size > holder->input_tensor_bytes) {
      LOGE("Input too large for zero-copy buffer: %zu > %zu", copy_size,
           holder->input_tensor_bytes);
      use_zero_copy = false;
    } else {
      std::memcpy(holder->input_mem->virt_addr, input_ptr, copy_size);
      if (!SyncInputTensorMem(holder, copy_size)) {
        ThrowIllegalState(env,
                          "Zero-copy input buffer validation failed for input");
        return nullptr;
      }
    }
  }

  if (!use_zero_copy) {
    rknn_input inputs[1];
    std::memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].type = holder->input_attr.type;
    inputs[0].fmt = holder->input_attr.fmt;
    inputs[0].size = input_size;
    inputs[0].pass_through = 0;
    inputs[0].buf = input_ptr;

    int ret = rknn_inputs_set(holder->ctx, 1, inputs);
    if (ret != RKNN_SUCC) {
      LOGE("rknn_inputs_set failed: %d", ret);
      ThrowIllegalState(env, "rknn_inputs_set failed: " + std::to_string(ret));
      return nullptr;
    }
  }

  auto t_set = std::chrono::high_resolution_clock::now();

  int ret = rknn_run(holder->ctx, nullptr);
  if (ret != RKNN_SUCC) {
    LOGE("rknn_run failed: %d", ret);
    ThrowIllegalState(env, "rknn_run failed: " + std::to_string(ret));
    return nullptr;
  }
  auto t_run = std::chrono::high_resolution_clock::now();

  const rknn_tensor_attr &out_attr = holder->output_attrs[0];
  const size_t out_elems = out_attr.n_elems;
  std::vector<float> output(out_elems, 0.f);
  bool use_quant_output = g_force_quant_output &&
                          out_attr.qnt_type != RKNN_TENSOR_QNT_NONE &&
                          (out_attr.type == RKNN_TENSOR_INT8 ||
                           out_attr.type == RKNN_TENSOR_UINT8) &&
                          out_attr.scale != 0.0f;
  if (use_quant_output) {
    holder->quant_output_buffer.resize(out_attr.size);
  }

  std::vector<rknn_output> outputs(holder->output_attrs.size());
  for (size_t i = 0; i < outputs.size(); ++i) {
    outputs[i].want_float = use_quant_output ? 0 : 1;
    if (i == 0) {
      outputs[i].is_prealloc = 1;
      outputs[i].buf =
          use_quant_output
              ? static_cast<void *>(holder->quant_output_buffer.data())
              : static_cast<void *>(output.data());
      outputs[i].size =
          use_quant_output ? out_attr.size : out_elems * sizeof(float);
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
    ThrowIllegalState(env, "rknn_outputs_get failed: " + std::to_string(ret));
    return nullptr;
  }
  auto t_get = std::chrono::high_resolution_clock::now();

  // Release non-prealloc outputs if any
  rknn_outputs_release(holder->ctx, static_cast<uint32_t>(outputs.size()),
                       outputs.data());

  if (use_quant_output) {
    DequantizeToFloat(out_attr, holder->quant_output_buffer.data(), output);
  }

  auto t_end = std::chrono::high_resolution_clock::now();
  if (ENABLE_LOGS) {
    long long us_set =
        std::chrono::duration_cast<std::chrono::microseconds>(t_set - t_start)
            .count();
    long long us_run =
        std::chrono::duration_cast<std::chrono::microseconds>(t_run - t_set)
            .count();
    long long us_get =
        std::chrono::duration_cast<std::chrono::microseconds>(t_get - t_run)
            .count();
    long long us_total =
        std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start)
            .count();
    LOGI("nativeRun profile: Set=%lld us Run=%lld us Get=%lld us Total=%lld us",
         us_set, us_run, us_get, us_total);
  }

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
      const rknn_tensor_attr &attr0 = holder->output_attrs[0];
      std::string dims_str;
      for (uint32_t i = 0; i < attr0.n_dims; ++i) {
        dims_str.append(std::to_string(attr0.dims[i]));
        if (i + 1 < attr0.n_dims)
          dims_str.append("x");
      }
      if (ENABLE_LOGS)
        LOGI("RKNN output[0] attr: n_dims=%u dims=%s n_elems=%u size=%u "
             "type=%d qnt_type=%d",
             attr0.n_dims, dims_str.c_str(), attr0.n_elems, attr0.size,
             attr0.type, attr0.qnt_type);
      holder->logged_shape = true;
    }
    if (ENABLE_LOGS)
      LOGI("RKNN output[0] size=%zu sample(first %zu): %s", output.size(),
           sample, buf.c_str());
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
Java_com_yolo_pose_detector_RknnRunner_nativeRunPixels(JNIEnv *env,
                                                       jobject /*thiz*/,
                                                       jlong handle,
                                                       jintArray pixels) {
  auto t_start = std::chrono::high_resolution_clock::now();
  auto *holder = FromHandle(handle);
  if (!holder) {
    ThrowIllegalState(env, "RKNN handle is null");
    return nullptr;
  }

  jsize pixel_count = env->GetArrayLength(pixels);
  jint *pixel_ptr = env->GetIntArrayElements(pixels, nullptr);
  if (!pixel_ptr) {
    ThrowIllegalState(env, "Failed to get pixel array elements");
    return nullptr;
  }

  // Determine target input type and resize buffer
  rknn_tensor_type input_type = holder->input_attr.type;
  size_t num_elements = pixel_count * 3; // RGB

  void *input_buf = nullptr;
  size_t input_size_bytes = 0;
  size_t contiguous_bytes = num_elements * ElemSize(holder->input_attr.type);
  bool zero_copy_possible =
      IsZeroCopyStrideCompatible(holder, contiguous_bytes);

  if (input_type == RKNN_TENSOR_FLOAT16) {
    const uint16_t *lut = holder->fp16_lut.data();
    uint16_t *dst = nullptr;
    if (zero_copy_possible &&
        num_elements * sizeof(uint16_t) <= holder->input_tensor_bytes) {
      dst = static_cast<uint16_t *>(holder->input_mem->virt_addr);
    } else {
      if (holder->fp16_input_buffer.size() != num_elements) {
        holder->fp16_input_buffer.resize(num_elements);
      }
      dst = holder->fp16_input_buffer.data();
    }

    // Convert ARGB int to FP16 RGB using LUT
    for (int i = 0; i < pixel_count; ++i) {
      jint p = pixel_ptr[i];
      dst[i * 3 + 0] = lut[(p >> 16) & 0xFF]; // R
      dst[i * 3 + 1] = lut[(p >> 8) & 0xFF];  // G
      dst[i * 3 + 2] = lut[p & 0xFF];         // B
    }
    input_buf = dst;
    input_size_bytes = num_elements * 2;
  } else if (input_type == RKNN_TENSOR_UINT8) {
    // Direct UINT8 pass-through for quantized models
    // pixels are ARGB_8888 (int), need to extract RGB bytes
    uint8_t *dst = nullptr;
    if (zero_copy_possible && num_elements <= holder->input_tensor_bytes) {
      dst = static_cast<uint8_t *>(holder->input_mem->virt_addr);
    } else {
      if (holder->uint8_input_buffer.size() != num_elements) {
        holder->uint8_input_buffer.resize(num_elements);
      }
      dst = holder->uint8_input_buffer.data();
    }
    for (int i = 0; i < pixel_count; ++i) {
      jint p = pixel_ptr[i];
      dst[i * 3 + 0] = static_cast<uint8_t>((p >> 16) & 0xFF);
      dst[i * 3 + 1] = static_cast<uint8_t>((p >> 8) & 0xFF);
      dst[i * 3 + 2] = static_cast<uint8_t>(p & 0xFF);
    }
    input_buf = dst;
    input_size_bytes = num_elements;
  } else {
    ThrowIllegalState(env, "Unsupported input type for runPixels: " +
                               std::to_string(input_type));
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

  bool using_zero_copy =
      holder->input_mem && input_buf == holder->input_mem->virt_addr;
  int ret = RKNN_SUCC;
  if (using_zero_copy) {
    if (!SyncInputTensorMem(holder, input_size_bytes)) {
      ThrowIllegalState(env,
                        "Zero-copy input buffer validation failed for pixels");
      return nullptr;
    }
  } else {
    ret = rknn_inputs_set(holder->ctx, 1, inputs);
    if (ret != RKNN_SUCC) {
      LOGE("rknn_inputs_set failed: %d", ret);
      ThrowIllegalState(env, "rknn_inputs_set failed: " + std::to_string(ret));
      return nullptr;
    }
  }

  ret = rknn_run(holder->ctx, nullptr);
  if (ret != RKNN_SUCC) {
    LOGE("rknn_run failed: %d", ret);
    ThrowIllegalState(env, "rknn_run failed: " + std::to_string(ret));
    return nullptr;
  }

  auto t_run = std::chrono::high_resolution_clock::now();

  // Get outputs
  const rknn_tensor_attr &out_attr = holder->output_attrs[0];
  const size_t out_elems = out_attr.n_elems;
  std::vector<float> output(out_elems, 0.f);
  bool use_quant_output = g_force_quant_output &&
                          out_attr.qnt_type != RKNN_TENSOR_QNT_NONE &&
                          (out_attr.type == RKNN_TENSOR_INT8 ||
                           out_attr.type == RKNN_TENSOR_UINT8) &&
                          out_attr.scale != 0.0f;
  if (use_quant_output) {
    holder->quant_output_buffer.resize(out_attr.size);
  }

  std::vector<rknn_output> outputs(holder->output_attrs.size());
  for (size_t i = 0; i < outputs.size(); ++i) {
    outputs[i].want_float = use_quant_output ? 0 : 1;
    if (i == 0) {
      outputs[i].is_prealloc = 1;
      outputs[i].buf =
          use_quant_output
              ? static_cast<void *>(holder->quant_output_buffer.data())
              : static_cast<void *>(output.data());
      outputs[i].size =
          use_quant_output ? out_attr.size : out_elems * sizeof(float);
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
    ThrowIllegalState(env, "rknn_outputs_get failed: " + std::to_string(ret));
    return nullptr;
  }
  auto t_get = std::chrono::high_resolution_clock::now();

  // Release non-prealloc outputs if any
  rknn_outputs_release(holder->ctx, static_cast<uint32_t>(outputs.size()),
                       outputs.data());

  if (use_quant_output) {
    DequantizeToFloat(out_attr, holder->quant_output_buffer.data(), output);
  }

  auto t_end = std::chrono::high_resolution_clock::now();
  if (ENABLE_LOGS) {
    long long us_run =
        std::chrono::duration_cast<std::chrono::microseconds>(t_run - t_start)
            .count();
    long long us_get =
        std::chrono::duration_cast<std::chrono::microseconds>(t_get - t_run)
            .count();
    long long us_total =
        std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start)
            .count();
    LOGI("nativeRunPixels profile: Run=%lld us Get=%lld us Total=%lld us",
         us_run, us_get, us_total);
  }

  if (use_quant_output) {
    DequantizeToFloat(out_attr, holder->quant_output_buffer.data(), output);
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

extern "C" JNIEXPORT void JNICALL
Java_com_yolo_pose_detector_RknnRunner_nativeRelease(JNIEnv * /*env*/,
                                                     jobject /*thiz*/,
                                                     jlong handle) {
  auto *holder = FromHandle(handle);
  if (!holder)
    return;
  if (holder->input_mem) {
    rknn_destroy_mem(holder->ctx, holder->input_mem);
    holder->input_mem = nullptr;
  }
  if (holder->ctx != 0) {
    rknn_destroy(holder->ctx);
  }
  delete holder;
}

#include <android/bitmap.h>

// ========================= RGA Preprocessing Function
// =========================
/**
 * PreprocessWithRGA - 使用硬件RGA进行图像预处理
 * Hardware-accelerated image preprocessing using RGA (zero-copy to NPU)
 *
 * @param holder - RKNN context holder containing RGA state
 * @param src_pixels - Source bitmap pixel data (RGBA_8888 format)
 * @param info - Android bitmap information
 * @param output_buf - Output buffer pointer (will be set by this function)
 * @param output_size - Output buffer size in bytes (will be set by this
 * function)
 * @return 0 on success, -1 on failure (falls back to CPU processing)
 */
int PreprocessWithRGA(RknnHolder *holder, void *src_pixels,
                      AndroidBitmapInfo *info, void **output_buf,
                      size_t *output_size) {
  Timer t("RGA Preprocessing");

  rknn_tensor_type input_type = holder->input_attr.type;
  int target_width = holder->input_attr.dims[2]; // Typically 320 or 640
  int target_height = holder->input_attr.dims[1];
  int dst_w_stride =
      holder->input_attr.w_stride ? holder->input_attr.w_stride : target_width;
  int dst_h_stride =
      holder->input_attr.h_stride ? holder->input_attr.h_stride : target_height;

  // 1. Create source buffer descriptor (Android Bitmap: RGBA8888)
  rga_buffer_t src_buf = wrapbuffer_virtualaddr(
      src_pixels, static_cast<int>(info->width), static_cast<int>(info->height),
      RK_FORMAT_RGBA_8888, static_cast<int>(info->stride / 4),
      static_cast<int>(info->height));

  // 2. Prepare destination buffer (RGB format)
  size_t num_elements = static_cast<size_t>(target_width) * target_height * 3;
  size_t dst_bytes_with_stride =
      static_cast<size_t>(dst_w_stride) * dst_h_stride * 3;
  bool has_zero_copy =
      holder->input_mem && holder->input_mem->virt_addr &&
      IsZeroCopyStrideCompatible(holder, dst_bytes_with_stride);
  bool has_phys_addr = holder->input_mem && holder->input_mem->phys_addr != 0;
  if (!has_zero_copy && ENABLE_LOGS && !holder->zero_copy_warning_shown) {
    LOGI("Zero-copy disabled (input_mem=%p, virt=%p, phys=%p, stride=%u)",
         static_cast<void *>(holder->input_mem),
         holder->input_mem ? holder->input_mem->virt_addr : nullptr,
         holder->input_mem
             ? reinterpret_cast<void *>(holder->input_mem->phys_addr)
             : nullptr,
         holder->input_attr.size_with_stride);
    holder->zero_copy_warning_shown = true;
  }

  if (input_type == RKNN_TENSOR_UINT8) {
    // NPU needs UINT8 input - directly output RGB888 into RKNN-managed memory
    rga_buffer_t dst_buf{};
    uint8_t *dst_ptr = nullptr;

    if (has_zero_copy) {
      if (has_phys_addr) {
        dst_buf = wrapbuffer_physicaladdr(
            reinterpret_cast<void *>(holder->input_mem->phys_addr),
            target_width, target_height, RK_FORMAT_RGB_888, dst_w_stride,
            dst_h_stride);
      } else {
        dst_buf = wrapbuffer_virtualaddr(
            holder->input_mem->virt_addr, target_width, target_height,
            RK_FORMAT_RGB_888, dst_w_stride, dst_h_stride);
      }
      dst_ptr = static_cast<uint8_t *>(holder->input_mem->virt_addr);
    } else {
      if (holder->uint8_input_buffer.size() != num_elements) {
        holder->uint8_input_buffer.resize(num_elements);
      }
      dst_buf = wrapbuffer_virtualaddr(holder->uint8_input_buffer.data(),
                                       target_width, target_height,
                                       RK_FORMAT_RGB_888);
      dst_ptr = holder->uint8_input_buffer.data();
      dst_bytes_with_stride = num_elements;
    }

    // 3. Execute RGA operation: RGBA->RGB + Resize (hardware accelerated)
    IM_STATUS ret = imresize(src_buf, dst_buf);
    if (ret != IM_STATUS_SUCCESS) {
      LOGE("RGA imresize failed: %d", ret);
      return -1;
    }

    *output_buf = dst_ptr;
    *output_size = dst_bytes_with_stride;

  } else if (input_type == RKNN_TENSOR_FLOAT16) {
    // First output UINT8 via RGA, then CPU convert to FP16
    // (RGA doesn't directly support FP16 output)
    if (holder->uint8_input_buffer.size() != num_elements) {
      holder->uint8_input_buffer.resize(num_elements);
    }

    rga_buffer_t dst_buf =
        wrapbuffer_virtualaddr(holder->uint8_input_buffer.data(), target_width,
                               target_height, RK_FORMAT_RGB_888);

    IM_STATUS ret = imresize(src_buf, dst_buf);
    if (ret != IM_STATUS_SUCCESS) {
      LOGE("RGA imresize failed: %d", ret);
      return -1;
    }

    // 4. Fast CPU conversion UINT8 -> FP16 (data is already resized, much
    // smaller)
    const uint16_t *lut = holder->fp16_lut.data();
    const uint8_t *src = holder->uint8_input_buffer.data();
    size_t fp16_bytes_with_stride =
        static_cast<size_t>(dst_w_stride) * dst_h_stride * 3 * sizeof(uint16_t);
    bool fp16_zero_copy =
        holder->input_mem &&
        IsZeroCopyStrideCompatible(holder, fp16_bytes_with_stride);

    if (fp16_zero_copy) {
      uint16_t *dst = static_cast<uint16_t *>(holder->input_mem->virt_addr);
      size_t dst_row_elems = static_cast<size_t>(dst_w_stride) * 3;
      size_t valid_row_elems = static_cast<size_t>(target_width) * 3;
      for (int y = 0; y < target_height && y < dst_h_stride; ++y) {
        uint16_t *dst_row = dst + y * dst_row_elems;
        const uint8_t *src_row = src + static_cast<size_t>(y) * valid_row_elems;
        for (size_t x = 0; x < valid_row_elems; ++x) {
          dst_row[x] = lut[src_row[x]];
        }
        if (dst_row_elems > valid_row_elems) {
          std::memset(dst_row + valid_row_elems, 0,
                      (dst_row_elems - valid_row_elems) * sizeof(uint16_t));
        }
      }
      if (dst_h_stride > target_height) {
        std::memset(dst + static_cast<size_t>(target_height) * dst_row_elems, 0,
                    (dst_h_stride - target_height) * dst_row_elems *
                        sizeof(uint16_t));
      }
      *output_buf = dst;
      *output_size = fp16_bytes_with_stride;
    } else {
      if (holder->fp16_input_buffer.size() != num_elements) {
        holder->fp16_input_buffer.resize(num_elements);
      }
      uint16_t *dst = holder->fp16_input_buffer.data();
      for (size_t i = 0; i < num_elements; ++i) {
        dst[i] = lut[src[i]];
      }
      *output_buf = dst;
      *output_size = num_elements * 2;
    }

  } else {
    LOGE("Unsupported input type for RGA preprocessing: %d", input_type);
    return -1;
  }

  bool zero_copy_used =
      holder->input_mem && *output_buf == holder->input_mem->virt_addr;
  if (ENABLE_LOGS)
    LOGI("RGA preprocessing successful: %dx%d -> %dx%d (%s)", info->width,
         info->height, target_width, target_height,
         zero_copy_used ? "zero-copy" : "cpu-buf");
  return 0;
}

// ...

inline float inverse_sigmoid(float y) {
  if (y <= 0.0001f)
    return -10.0f;
  if (y >= 0.9999f)
    return 10.0f;
  return -std::log(1.0f / y - 1.0f);
}

extern "C" JNIEXPORT jfloatArray JNICALL
Java_com_yolo_pose_detector_RknnRunner_nativeRunBitmapWithNms(
    JNIEnv *env, jobject /*thiz*/, jlong handle, jobject bitmap,
    jfloat detectThresh, jfloat nmsThresh) {
  auto *holder = FromHandle(handle);
  if (!holder) {
    ThrowIllegalState(env, "RKNN handle is null");
    return nullptr;
  }
  if (ENABLE_LOGS)
    LOGI("开始处理视频帧");

  auto t_start = std::chrono::high_resolution_clock::now();

  AndroidBitmapInfo info;
  if (AndroidBitmap_getInfo(env, bitmap, &info) < 0) {
    ThrowIllegalState(env, "AndroidBitmap_getInfo failed");
    return nullptr;
  }
  if (ENABLE_LOGS)
    LOGI("Bitmap info: %u x %u stride=%u format=%d", info.width, info.height,
         info.stride, info.format);

  if (info.format != ANDROID_BITMAP_FORMAT_RGBA_8888) {
    ThrowIllegalState(env, "Bitmap format must be ARGB_8888");
    return nullptr;
  }

  void *pixels_ptr = nullptr;
  if (AndroidBitmap_lockPixels(env, bitmap, &pixels_ptr) < 0) {
    ThrowIllegalState(env, "AndroidBitmap_lockPixels failed");
    return nullptr;
  }

  auto t_lock = std::chrono::high_resolution_clock::now();
  auto t_preprocess_start = t_lock;

  rknn_tensor_type input_type = holder->input_attr.type;
  size_t num_pixels = info.width * info.height;
  size_t num_elements = num_pixels * 3;

  void *input_buf = nullptr;
  size_t input_size_bytes = 0;

  // ========== Try RGA hardware acceleration first ==========
  bool rga_success = false;
  if (holder->rga_available) {
    int rga_ret = PreprocessWithRGA(holder, pixels_ptr, &info, &input_buf,
                                    &input_size_bytes);
    if (rga_ret == 0) {
      rga_success = true;
      if (ENABLE_LOGS)
        LOGI("Using RGA hardware preprocessing");
    } else {
      LOGE("RGA preprocessing failed, falling back to CPU");
      holder->rga_available = false; // Disable RGA for subsequent frames
    }
  }

  // ========== Fallback to CPU preprocessing if RGA failed/unavailable
  // ==========
  if (!rga_success) {
    if (ENABLE_LOGS && !holder->rga_available) {
      LOGI("Using CPU preprocessing (RGA unavailable)");
    }

    int dst_w_stride = holder->input_attr.w_stride
                           ? holder->input_attr.w_stride
                           : static_cast<int>(info.width);
    int dst_h_stride = holder->input_attr.h_stride
                           ? holder->input_attr.h_stride
                           : static_cast<int>(info.height);
    int copy_width = std::min(dst_w_stride, static_cast<int>(info.width));
    int copy_height = std::min(dst_h_stride, static_cast<int>(info.height));

    if (input_type == RKNN_TENSOR_FLOAT16) {
      const uint16_t *lut = holder->fp16_lut.data();
      size_t stride_bytes = static_cast<size_t>(dst_w_stride) * dst_h_stride *
                            3 * sizeof(uint16_t);
      bool zero_copy_possible =
          holder->input_mem && IsZeroCopyStrideCompatible(holder, stride_bytes);

      uint16_t *dst = nullptr;
      size_t target_bytes = num_elements * sizeof(uint16_t);
      if (zero_copy_possible) {
        dst = static_cast<uint16_t *>(holder->input_mem->virt_addr);
      } else {
        if (holder->fp16_input_buffer.size() != num_elements) {
          holder->fp16_input_buffer.resize(num_elements);
        }
        dst = holder->fp16_input_buffer.data();
      }

      uint8_t *src_rows = static_cast<uint8_t *>(pixels_ptr);
      if (zero_copy_possible) {
        size_t dst_row_elems = static_cast<size_t>(dst_w_stride) * 3;
        size_t valid_row_elems = static_cast<size_t>(copy_width) * 3;
        for (int y = 0; y < copy_height; ++y) {
          uint16_t *dst_row = dst + static_cast<size_t>(y) * dst_row_elems;
          uint8_t *src_pixel = src_rows + static_cast<size_t>(y) * info.stride;
          for (int x = 0; x < copy_width; ++x) {
            size_t base = static_cast<size_t>(x) * 3;
            dst_row[base + 0] = lut[src_pixel[2]];
            dst_row[base + 1] = lut[src_pixel[1]];
            dst_row[base + 2] = lut[src_pixel[0]];
            src_pixel += 4;
          }
          if (dst_row_elems > valid_row_elems) {
            std::memset(dst_row + valid_row_elems, 0,
                        (dst_row_elems - valid_row_elems) * sizeof(uint16_t));
          }
        }
        if (dst_h_stride > copy_height) {
          std::memset(dst + static_cast<size_t>(copy_height) * dst_row_elems, 0,
                      (dst_h_stride - copy_height) * dst_row_elems *
                          sizeof(uint16_t));
        }
        input_size_bytes = stride_bytes;
      } else {
        size_t dst_idx = 0;
        for (int y = 0; y < info.height; ++y) {
          uint8_t *src_pixel = src_rows + static_cast<size_t>(y) * info.stride;
          for (int x = 0; x < info.width; ++x) {
            uint8_t b = src_pixel[0];
            uint8_t g = src_pixel[1];
            uint8_t r = src_pixel[2];

            dst[dst_idx++] = lut[r];
            dst[dst_idx++] = lut[g];
            dst[dst_idx++] = lut[b];

            src_pixel += 4;
          }
        }
        input_size_bytes = target_bytes;
      }
      input_buf = dst;
    } else if (input_type == RKNN_TENSOR_UINT8) {
      size_t stride_bytes =
          static_cast<size_t>(dst_w_stride) * dst_h_stride * 3;
      bool zero_copy_possible =
          holder->input_mem && IsZeroCopyStrideCompatible(holder, stride_bytes);

      uint8_t *dst = nullptr;
      if (zero_copy_possible) {
        dst = static_cast<uint8_t *>(holder->input_mem->virt_addr);
      } else {
        if (holder->uint8_input_buffer.size() != num_elements) {
          holder->uint8_input_buffer.resize(num_elements);
        }
        dst = holder->uint8_input_buffer.data();
      }

      uint8_t *src_rows = static_cast<uint8_t *>(pixels_ptr);
      if (zero_copy_possible) {
        size_t dst_row_bytes = static_cast<size_t>(dst_w_stride) * 3;
        size_t valid_row_bytes = static_cast<size_t>(copy_width) * 3;
        for (int y = 0; y < copy_height; ++y) {
          uint8_t *dst_row = dst + static_cast<size_t>(y) * dst_row_bytes;
          uint8_t *src_pixel = src_rows + static_cast<size_t>(y) * info.stride;
          for (int x = 0; x < copy_width; ++x) {
            size_t base = static_cast<size_t>(x) * 3;
            dst_row[base + 0] = src_pixel[2];
            dst_row[base + 1] = src_pixel[1];
            dst_row[base + 2] = src_pixel[0];
            src_pixel += 4;
          }
          if (dst_row_bytes > valid_row_bytes) {
            std::memset(dst_row + valid_row_bytes, 0,
                        dst_row_bytes - valid_row_bytes);
          }
        }
        if (dst_h_stride > copy_height) {
          std::memset(dst + static_cast<size_t>(copy_height) * dst_row_bytes, 0,
                      (dst_h_stride - copy_height) * dst_row_bytes);
        }
        input_size_bytes = stride_bytes;
      } else {
        size_t dst_idx = 0;
        for (int y = 0; y < info.height; ++y) {
          uint8_t *src_pixel = src_rows + static_cast<size_t>(y) * info.stride;
          for (int x = 0; x < info.width; ++x) {
            uint8_t b = src_pixel[0];
            uint8_t g = src_pixel[1];
            uint8_t r = src_pixel[2];

            dst[dst_idx++] = r;
            dst[dst_idx++] = g;
            dst[dst_idx++] = b;

            src_pixel += 4;
          }
        }
        input_size_bytes = num_elements;
      }
      input_buf = dst;
    } else {
      AndroidBitmap_unlockPixels(env, bitmap);
      ThrowIllegalState(env, "Unsupported input type for runPixels: " +
                                 std::to_string(input_type));
      return nullptr;
    }
  } // End of if (!rga_success) for input buffer preparation

  AndroidBitmap_unlockPixels(env, bitmap);

  auto t_prep = std::chrono::high_resolution_clock::now();
  long long us_preprocess =
      std::chrono::duration_cast<std::chrono::microseconds>(t_prep -
                                                            t_preprocess_start)
          .count();
  if (ENABLE_LOGS)
    LOGI("预处理耗时: %lld us (%s)", us_preprocess,
         rga_success ? "RGA" : "CPU");

  bool using_zero_copy =
      holder->input_mem && input_buf == holder->input_mem->virt_addr;

  if (ENABLE_LOGS)
    LOGI("Prep done: input type=%d elements=%zu bytes=%zu (%s)", input_type,
         num_elements, input_size_bytes,
         using_zero_copy ? "zero-copy" : "copy");

  int ret = RKNN_SUCC;
  if (using_zero_copy) {
    if (!SyncInputTensorMem(holder, input_size_bytes)) {
      ThrowIllegalState(
          env, "Zero-copy input buffer validation failed for bitmap input");
      return nullptr;
    }
  } else {
    rknn_input inputs[1];
    std::memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].type = input_type;
    inputs[0].fmt = holder->input_attr.fmt;
    inputs[0].size = static_cast<uint32_t>(input_size_bytes);
    inputs[0].buf = input_buf;

    ret = rknn_inputs_set(holder->ctx, 1, inputs);
    if (ret != RKNN_SUCC) {
      ThrowIllegalState(env, "rknn_inputs_set failed: " + std::to_string(ret));
      return nullptr;
    }
  }

  auto t_set = std::chrono::high_resolution_clock::now();

  ret = rknn_run(holder->ctx, nullptr);
  if (ret != RKNN_SUCC) {
    ThrowIllegalState(env, "rknn_run failed: " + std::to_string(ret));
    return nullptr;
  }

  auto t_run = std::chrono::high_resolution_clock::now();
  long long us_infer =
      std::chrono::duration_cast<std::chrono::microseconds>(t_run - t_set)
          .count();
  if (ENABLE_LOGS)
    LOGI("模型推理耗时: %lld us", us_infer);

  const rknn_tensor_attr &out_attr = holder->output_attrs[0];
  const size_t out_elems = out_attr.n_elems;
  std::vector<float> output(out_elems);
  bool use_quant_output = g_force_quant_output &&
                          out_attr.qnt_type != RKNN_TENSOR_QNT_NONE &&
                          (out_attr.type == RKNN_TENSOR_INT8 ||
                           out_attr.type == RKNN_TENSOR_UINT8) &&
                          out_attr.scale != 0.0f;
  if (use_quant_output) {
    holder->quant_output_buffer.resize(out_attr.size);
  }
  rknn_output rknn_outputs[1];
  std::memset(rknn_outputs, 0, sizeof(rknn_outputs));
  rknn_outputs[0].want_float = use_quant_output ? 0 : 1;
  rknn_outputs[0].is_prealloc = 1;
  rknn_outputs[0].buf =
      use_quant_output ? static_cast<void *>(holder->quant_output_buffer.data())
                       : static_cast<void *>(output.data());
  rknn_outputs[0].size =
      use_quant_output ? out_attr.size : out_elems * sizeof(float);

  ret = rknn_outputs_get(holder->ctx, 1, rknn_outputs, nullptr);
  if (ret != RKNN_SUCC) {
    ThrowIllegalState(env, "rknn_outputs_get failed: " + std::to_string(ret));
    return nullptr;
  }
  rknn_outputs_release(holder->ctx, 1, rknn_outputs);

  if (use_quant_output) {
    DequantizeToFloat(out_attr, holder->quant_output_buffer.data(), output);
  }

  auto t_get = std::chrono::high_resolution_clock::now();

  // --- End of inference logic ---

  // --- Start of Post-Processing ---
  // outputShape: [1, 56, 8400] or [1, 8400, 56]
  int dims[4] = {0};
  for (int i = 0; i < out_attr.n_dims; ++i)
    dims[i] = out_attr.dims[i];

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
  if (ENABLE_LOGS)
    LOGI("Output: n_dims=%u dims=%d %d %d %d channelFirst=%d candidates=%d "
         "values=%d",
         out_attr.n_dims, dims[0], dims[1], dims[2], dims[3],
         channelFirst ? 1 : 0, candidateCount, valuesPerCandidate);

  int numKeypoints = (valuesPerCandidate - 5) / 3;
  std::vector<Pose> candidates;
  candidates.reserve(100);

  // Assuming NHWC input layout for width/height retrieval if needed
  // Using dims[2] as width (commonly 1, H, W, C)
  float inputSize = static_cast<float>(holder->input_attr.dims[2]);
  if (inputSize == 0)
    inputSize = 640.0f;

  float rawDetectThresh = inverse_sigmoid(detectThresh);

  std::vector<std::pair<float, int>> passed;
  passed.reserve(candidateCount);
  for (int i = 0; i < candidateCount; ++i) {
    float rawScore = channelFirst ? output[4 * candidateCount + i]
                                  : output[i * valuesPerCandidate + 4];
    if (rawScore >= rawDetectThresh)
      passed.emplace_back(rawScore, i);
  }
  int topk = static_cast<int>(passed.size());
  if (topk > 300) {
    topk = 300;
    std::nth_element(
        passed.begin(), passed.begin() + topk, passed.end(),
        [](const std::pair<float, int> &a, const std::pair<float, int> &b) {
          return a.first > b.first;
        });
    passed.resize(topk);
  }

  if (ENABLE_LOGS)
    LOGI("Decode: candidates=%d, passed=%zu, topk=%d", candidateCount,
         passed.size(), topk);

  for (int idx = 0; idx < topk; ++idx) {
    int i = passed[idx].second;
    float rawScore = passed[idx].first;
    float score = sigmoid(rawScore);

    Pose pose;
    pose.score = score;

    float cx = channelFirst ? output[0 * candidateCount + i]
                            : output[i * valuesPerCandidate + 0];
    float cy = channelFirst ? output[1 * candidateCount + i]
                            : output[i * valuesPerCandidate + 1];
    float w = channelFirst ? output[2 * candidateCount + i]
                           : output[i * valuesPerCandidate + 2];
    float h = channelFirst ? output[3 * candidateCount + i]
                           : output[i * valuesPerCandidate + 3];

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
      float kx = channelFirst ? output[(5 + k * 3) * candidateCount + i]
                              : output[i * valuesPerCandidate + (5 + k * 3)];
      float ky = channelFirst
                     ? output[(5 + k * 3 + 1) * candidateCount + i]
                     : output[i * valuesPerCandidate + (5 + k * 3 + 1)];
      float ks = channelFirst
                     ? output[(5 + k * 3 + 2) * candidateCount + i]
                     : output[i * valuesPerCandidate + (5 + k * 3 + 2)];

      if (kx > 2.0f) {
        kx /= inputSize;
        ky /= inputSize;
      }

      pose.keypoints.push_back({kx, ky, sigmoid(ks)});
    }
    candidates.push_back(pose);
  }

  if (ENABLE_LOGS)
    LOGI("Before NMS: %zu candidates", candidates.size());
  NMS(candidates, nmsThresh);
  if (ENABLE_LOGS)
    LOGI("After NMS: %zu", candidates.size());

  // Serialize results
  // Format: [num_poses, score, x, y, w, h, kp1_x, kp1_y, kp1_s, ...,
  // score2...]
  int poseSize = 5 + numKeypoints * 3;
  std::vector<float> resultData;
  resultData.reserve(1 + candidates.size() * poseSize);

  resultData.push_back(static_cast<float>(candidates.size()));

  for (const auto &p : candidates) {
    resultData.push_back(p.score);
    resultData.push_back(p.x);
    resultData.push_back(p.y);
    resultData.push_back(p.w);
    resultData.push_back(p.h);
    for (const auto &kp : p.keypoints) {
      resultData.push_back(kp.x);
      resultData.push_back(kp.y);
      resultData.push_back(kp.score);
    }
  }

  jfloatArray resultArray = env->NewFloatArray(resultData.size());
  env->SetFloatArrayRegion(resultArray, 0, resultData.size(),
                           resultData.data());

  auto t_post = std::chrono::high_resolution_clock::now();

  long long us_prep =
      std::chrono::duration_cast<std::chrono::microseconds>(t_prep - t_start)
          .count();
  long long us_set =
      std::chrono::duration_cast<std::chrono::microseconds>(t_set - t_prep)
          .count();
  long long us_run =
      std::chrono::duration_cast<std::chrono::microseconds>(t_run - t_set)
          .count();
  long long us_get =
      std::chrono::duration_cast<std::chrono::microseconds>(t_get - t_run)
          .count();
  long long us_post =
      std::chrono::duration_cast<std::chrono::microseconds>(t_post - t_get)
          .count();
  long long us_total =
      std::chrono::duration_cast<std::chrono::microseconds>(t_post - t_start)
          .count();

  if (ENABLE_LOGS)
    LOGI("Profile: Prep=%lld us, Set=%lld us, Run=%lld us, Get=%lld us, "
         "Post=%lld us, Total=%lld us",
         us_prep, us_set, us_run, us_get, us_post, us_total);

  if (ENABLE_LOGS)
    LOGI("视频帧处理完成");

  return resultArray;
}
