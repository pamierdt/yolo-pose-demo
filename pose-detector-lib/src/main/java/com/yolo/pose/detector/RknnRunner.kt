package com.yolo.pose.detector

import android.graphics.Bitmap
import android.util.Log
import java.nio.ByteBuffer

/**
 * RknnRunner - RKNN运行时包装类 / RKNN Runtime Wrapper
 * 
 * 用于加载和运行YOLO姿态估计RKNN模型的JNI包装器
 * Thin JNI wrapper around RKNN runtime for loading and running YOLO pose RKNN models.
 * 
 * 使用方法 / Usage:
 * ```kotlin
 * val modelBuffer = loadModelFromAssets("yolov8n-pose.rknn")
 * val runner = RknnRunner(modelBuffer)
 * val result = runner.runBitmapWithNms(bitmap, detectThreshold = 0.5f, nmsThreshold = 0.5f)
 * runner.close()
 * ```
 */
class RknnRunner(modelBuffer: ByteBuffer) : AutoCloseable {

    companion object {
        private const val TAG = "RknnRunner"
        
        init {
            try {
                // Load RGA library (optional, for hardware acceleration)
                try {
                    System.loadLibrary("rga")
                    Log.d(TAG, "RGA library loaded successfully")
                } catch (e: UnsatisfiedLinkError) {
                    Log.w(TAG, "RGA library not available, using software fallback")
                }
                
                // Load RKNN runtime
                System.loadLibrary("rknnrt")
                Log.d(TAG, "RKNN runtime loaded successfully")
                
                // Load pose detector library
                System.loadLibrary("pose_detector")
                Log.d(TAG, "Pose detector library loaded successfully")
            } catch (e: UnsatisfiedLinkError) {
                Log.e(TAG, "Failed to load native libraries", e)
                throw e
            }
        }
    }

    private var handle: Long = 0
    val outputShape: IntArray

    init {
        require(modelBuffer.isDirect) { "Model buffer must be a direct ByteBuffer" }
        val modelSize = modelBuffer.remaining()
        handle = nativeInit(modelBuffer, modelSize)
        if (handle == 0L) {
            throw IllegalStateException("Failed to init RKNN runtime, see logcat for details")
        }
        outputShape = nativeGetOutputShape(handle)
        Log.d(TAG, "RknnRunner initialized successfully, output shape: ${outputShape.contentToString()}")
    }

    /**
     * 运行推理（使用预处理的输入缓冲区）
     * Run inference with preprocessed input buffer
     */
    fun run(inputBuffer: ByteBuffer): FloatArray {
        check(handle != 0L) { "RKNN runtime not initialized" }
        require(inputBuffer.isDirect) { "Input buffer must be direct" }
        val inputSize = inputBuffer.remaining()
        return nativeRun(handle, inputBuffer, inputSize)
    }

    /**
     * 高效运行推理（传递原始ARGB像素）
     * Efficiently runs inference by passing raw ARGB pixels.
     * 转换为模型输入格式（如FP16 RGB）在Native/C++中完成以节省时间
     * Conversion to model input format (e.g. FP16 RGB) happens in Native/C++ to save time.
     */
    fun runPixels(pixels: IntArray): FloatArray {
        check(handle != 0L) { "RKNN runtime not initialized" }
        return nativeRunPixels(handle, pixels)
    }

    /**
     * 高效运行推理（使用Bitmap直接访问，可能零拷贝）
     * Efficiently runs inference using direct Bitmap access (zero-copy if possible).
     * 
     * @param bitmap 输入图像 / Input image
     * @param detectThreshold 检测阈值 (0.0-1.0) / Detection threshold
     * @param nmsThreshold NMS阈值 (0.0-1.0) / NMS threshold
     * @return 扁平化的姿态结果数组 / Flattened pose results array
     */
    fun runBitmapWithNms(bitmap: Bitmap, detectThreshold: Float, nmsThreshold: Float): FloatArray {
        check(handle != 0L) { "RKNN runtime not initialized" }
        return try {
            nativeRunBitmapWithNms(handle, bitmap, detectThreshold, nmsThreshold)
        } catch (t: Throwable) {
            // 兜底：避免 native 不兼容（例如 INT8 输入）导致直接闪退
            Log.e(TAG, "nativeRunBitmapWithNms failed: ${t.message}", t)
            FloatArray(0)
        }
    }

    /**
     * 设置性能选项 / Set performance options
     * @param useQuantOutput 使用量化输出（手动反量化）/ Use quantized output (manual dequantization)
     * @param cacheableInput 输入缓冲是否可缓存 / Whether input buffer is cacheable
     */
    fun setPerfOptions(useQuantOutput: Boolean, cacheableInput: Boolean) {
        nativeSetPerfOptions(useQuantOutput, cacheableInput)
    }

    override fun close() {
        if (handle != 0L) {
            Log.d(TAG, "Releasing RKNN runtime: handle=$handle")
            nativeRelease(handle)
            handle = 0
        }
    }

    protected fun finalize() {
        if (handle != 0L) {
            Log.w(TAG, "RknnRunner finalized without explicit close()")
        }
        close()
    }

    // ========== JNI Native 方法声明 / JNI Native Method Declarations ==========
    
    private external fun nativeInit(modelBuffer: ByteBuffer, modelSize: Int): Long
    private external fun nativeGetOutputShape(handle: Long): IntArray
    private external fun nativeRun(handle: Long, inputBuffer: ByteBuffer, inputSize: Int): FloatArray
    private external fun nativeRunPixels(handle: Long, pixels: IntArray): FloatArray
    private external fun nativeRunBitmapWithNms(handle: Long, bitmap: Bitmap, detectThresh: Float, nmsThresh: Float): FloatArray
    private external fun nativeSetPerfOptions(useQuantOutput: Boolean, cacheableInput: Boolean)
    private external fun nativeRelease(handle: Long)
}
