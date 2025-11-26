#include <jni.h>
#include <android/log.h>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "rknn_api.h"

#define LOG_TAG "rknn_jni"
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)

namespace {

struct RknnHolder {
  rknn_context ctx = 0;
  rknn_tensor_attr input_attr{};
  std::vector<rknn_tensor_attr> output_attrs;
  bool logged_shape = false;
};

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

}  // namespace

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
    LOGI("RKNN input attr: n_dims=%u dims=%s n_elems=%u size=%u type=%d fmt=%d qnt_type=%d scale=%f zp=%d",
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

  LOGI("RKNN init ok. Input dims: %u %u %u %u, output num: %u",
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

  // Debug: log a small sample of output values for the first tensor
  if (!output.empty()) {
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
      LOGI("RKNN output[0] attr: n_dims=%u dims=%s n_elems=%u size=%u type=%d qnt_type=%d",
           attr0.n_dims, dims_str.c_str(), attr0.n_elems, attr0.size,
           attr0.type, attr0.qnt_type);
      holder->logged_shape = true;
    }
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
