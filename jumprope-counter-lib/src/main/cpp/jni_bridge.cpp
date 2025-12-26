#include "JumpRopeCounter.h"
#include <android/log.h>
#include <jni.h>

#define LOG_TAG "JumpRopeCounter_JNI"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)

static bool ENABLE_LOGS = true;

void ThrowIllegalState(JNIEnv *env, const char *msg) {
  jclass exClass = env->FindClass("java/lang/IllegalStateException");
  if (exClass) {
    env->ThrowNew(exClass, msg);
  }
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_yolo_jumprope_JumpRopeCounter_nativeCreate(JNIEnv *env, jobject,
                                                    jfloat minIntervalMs) {
  if (ENABLE_LOGS)
    LOGI("nativeCreate: minInt=%.2f", minIntervalMs);
  auto *obj = new JumpRopeCounter(minIntervalMs);
  return reinterpret_cast<jlong>(obj);
}

extern "C" JNIEXPORT void JNICALL
Java_com_yolo_jumprope_JumpRopeCounter_nativeReset(JNIEnv *env, jobject,
                                                   jlong handle) {
  auto *obj = reinterpret_cast<JumpRopeCounter *>(handle);
  if (!obj)
    return;
  if (ENABLE_LOGS)
    LOGI("nativeReset called for handle %lld", (long long)handle);
  obj->reset();
}

extern "C" JNIEXPORT jint JNICALL
Java_com_yolo_jumprope_JumpRopeCounter_nativeUpdate(JNIEnv *env, jobject,
                                                    jlong handle,
                                                    jfloat shoulderY,
                                                    jfloat hipY, jfloat ankleY,
                                                    jdouble timestampMs) {
  auto *obj = reinterpret_cast<JumpRopeCounter *>(handle);
  if (!obj) {
    ThrowIllegalState(env, "JumpRopeCounter handle is null");
    return 0;
  }
  return static_cast<jint>(obj->update(shoulderY, hipY, ankleY, timestampMs));
}

extern "C" JNIEXPORT jint JNICALL
Java_com_yolo_jumprope_JumpRopeCounter_nativeGetCount(JNIEnv *env, jobject,
                                                      jlong handle) {
  auto *obj = reinterpret_cast<JumpRopeCounter *>(handle);
  if (!obj) {
    ThrowIllegalState(env, "JumpRopeCounter handle is null");
    return 0;
  }
  int count = obj->getCount();
  return static_cast<jint>(count);
}

extern "C" JNIEXPORT jfloat JNICALL
Java_com_yolo_jumprope_JumpRopeCounter_nativeGetGroundY(JNIEnv *env, jobject,
                                                        jlong handle) {
  auto *obj = reinterpret_cast<JumpRopeCounter *>(handle);
  if (!obj) {
    ThrowIllegalState(env, "JumpRopeCounter handle is null");
    return 0.f;
  }
  return static_cast<jfloat>(obj->getGroundY());
}

extern "C" JNIEXPORT jint JNICALL
Java_com_yolo_jumprope_JumpRopeCounter_nativeGetState(JNIEnv *env, jobject,
                                                      jlong handle) {
  auto *obj = reinterpret_cast<JumpRopeCounter *>(handle);
  if (!obj) {
    ThrowIllegalState(env, "JumpRopeCounter handle is null");
    return 0;
  }
  return static_cast<jint>(obj->getState());
}

extern "C" JNIEXPORT void JNICALL
Java_com_yolo_jumprope_JumpRopeCounter_nativeSetThresholds(JNIEnv *env, jobject,
                                                           jlong handle,
                                                           jfloat upRatio,
                                                           jfloat downRatio) {
  auto *obj = reinterpret_cast<JumpRopeCounter *>(handle);
  if (!obj) {
    ThrowIllegalState(env, "JumpRopeCounter handle is null");
    return;
  }
  obj->setThresholds(upRatio, downRatio);
}

extern "C" JNIEXPORT void JNICALL
Java_com_yolo_jumprope_JumpRopeCounter_nativeRelease(JNIEnv *env, jobject,
                                                     jlong handle) {
  auto *obj = reinterpret_cast<JumpRopeCounter *>(handle);
  if (!obj)
    return;
  delete obj;
}
