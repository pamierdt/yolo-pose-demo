package com.yolo.jumprope

import android.util.Log

/**
 * JumpRopeCounter - 跳绳计数器 JNI 包装类
 * 
 * 这是一个 Kotlin 包装类，用于调用 C++ 实现的跳绳计数算法
 * This is a Kotlin wrapper class for calling the C++ jump rope counting algorithm
 * 
 * 使用方法 / Usage:
 * ```kotlin
 * val counter = JumpRopeCounter(minIntervalMs = 300f)
 * val count = counter.update(shoulderY, hipY, ankleY, timestampMs)
 * counter.close()  // 或使用 use {} 自动释放
 * ```
 * 
 * @param minIntervalMs 两次跳跃之间的最小间隔（毫秒），默认300ms
 *                      Minimum interval between jumps (milliseconds), default 300ms
 */
class JumpRopeCounter(
    minIntervalMs: Float = 300f
) : AutoCloseable {
    
    companion object {
        private const val TAG = "JumpRopeCounter"
        
        init {
            try {
                System.loadLibrary("jumprope_counter")
                Log.d(TAG, "Native library loaded successfully")
            } catch (e: UnsatisfiedLinkError) {
                Log.e(TAG, "Failed to load native library", e)
                throw e
            }
        }
    }

    // Native 对象句柄 / Native object handle
    // 0 表示未初始化或已释放 / 0 means uninitialized or released
    private var handle: Long = 0

    init {
        handle = nativeCreate(minIntervalMs)
        if (handle == 0L) {
            Log.e(TAG, "Failed to create native JumpRopeCounter")
            throw IllegalStateException("Failed to create JumpRopeCounter")
        }
        Log.d(TAG, "JumpRopeCounter created: handle=$handle, minInterval=${minIntervalMs}ms")
    }

    /**
     * 更新跳绳状态并返回当前计数
     * Update jump rope state and return current count
     * 
     * @param shoulderY 肩部Y坐标（归一化，0-1）/ Shoulder Y (normalized, 0-1)
     * @param hipY 髋部Y坐标（归一化，0-1）/ Hip Y (normalized, 0-1)
     * @param ankleY 踝部Y坐标（归一化，0-1）/ Ankle Y (normalized, 0-1)
     * @param timestampMs 时间戳（毫秒）/ Timestamp (milliseconds)
     * @return 当前跳跃计数 / Current jump count
     * @throws IllegalStateException 如果计数器已关闭 / If counter is closed
     */
    fun update(shoulderY: Float, hipY: Float, ankleY: Float, timestampMs: Double): Int {
        check(handle != 0L) { "JumpRopeCounter has been closed" }
        
        // 参数验证 / Parameter validation
        if (shoulderY !in 0f..1f || hipY !in 0f..1f || ankleY !in 0f..1f) {
            Log.w(TAG, "Warning: Coordinates out of range [0,1]: shoulder=$shoulderY, hip=$hipY, ankle=$ankleY")
        }
        
        return nativeUpdate(handle, shoulderY, hipY, ankleY, timestampMs)
    }

    /**
     * 获取当前跳跃计数 / Get current jump count
     */
    fun getCount(): Int {
        check(handle != 0L) { "JumpRopeCounter has been closed" }
        return nativeGetCount(handle)
    }

    /**
     * 获取地面基准Y坐标 / Get ground baseline Y coordinate
     * @return 髋部地面基准值 / Hip ground baseline value
     */
    fun getGroundY(): Float {
        check(handle != 0L) { "JumpRopeCounter has been closed" }
        return nativeGetGroundY(handle)
    }

    /**
     * 获取当前状态 / Get current state
     * @return 0=地面, 1=上升, 2=下降 / 0=GROUND, 1=ASCENDING, 2=DESCENDING
     */
    fun getState(): Int {
        check(handle != 0L) { "JumpRopeCounter has been closed" }
        return nativeGetState(handle)
    }

    /**
     * 重置计数器 / Reset counter
     * 清除所有状态和计数 / Clear all states and counts
     */
    fun reset() {
        check(handle != 0L) { "JumpRopeCounter has been closed" }
        Log.d(TAG, "Resetting counter")
        nativeReset(handle)
    }

    /**
     * 设置跳跃检测阈值
     * Set jump detection thresholds
     * @param upRatio 起跳阈值比例 (默认0.60) / Jump up threshold ratio (default 0.60)
     * @param downRatio 落地阈值比例 (默认0.35) / Landing down threshold ratio (default 0.35)
     */
    fun setThresholds(upRatio: Float, downRatio: Float) {
        check(handle != 0L) { "JumpRopeCounter has been closed" }
        nativeSetThresholds(handle, upRatio, downRatio)
    }

    /**
     * 释放 Native 资源 / Release native resources
     * 实现 AutoCloseable 接口，支持 use {} 语法
     * Implements AutoCloseable interface, supports use {} syntax
     */
    override fun close() {
        if (handle != 0L) {
            Log.d(TAG, "Releasing native JumpRopeCounter: handle=$handle")
            nativeRelease(handle)
            handle = 0
        }
    }

    /**
     * 析构函数，确保资源被释放 / Destructor, ensure resources are released
     */
    protected fun finalize() {
        if (handle != 0L) {
            Log.w(TAG, "JumpRopeCounter finalized without explicit close()")
        }
        close()
    }

    // ========== JNI Native 方法声明 / JNI Native Method Declarations ==========
    
    /**
     * 创建 Native 计数器对象 / Create native counter object
     * @return Native 对象句柄，0表示失败 / Native object handle, 0 means failure
     */
    private external fun nativeCreate(minIntervalMs: Float): Long
    
    /**
     * 更新状态（Native 调用）/ Update state (native call)
     */
    private external fun nativeUpdate(
        handle: Long, 
        shoulderY: Float, 
        hipY: Float, 
        ankleY: Float, 
        timestampMs: Double
    ): Int
    
    /**
     * 获取计数（Native 调用）/ Get count (native call)
     */
    private external fun nativeGetCount(handle: Long): Int
    
    /**
     * 获取地面基准（Native 调用）/ Get ground baseline (native call)
     */
    private external fun nativeGetGroundY(handle: Long): Float
    
    /**
     * 获取状态（Native 调用）/ Get state (native call)
     */
    private external fun nativeGetState(handle: Long): Int

    /**
     * 设置阈值（Native 调用）/ Set thresholds (native call)
     */
    private external fun nativeSetThresholds(
        handle: Long,
        upRatio: Float,
        downRatio: Float
    )
    
    /**
     * 重置计数器（Native 调用）/ Reset counter (native call)
     */
    private external fun nativeReset(handle: Long)
    
    /**
     * 释放 Native 资源（Native 调用）/ Release native resources (native call)
     */
    private external fun nativeRelease(handle: Long)
}
