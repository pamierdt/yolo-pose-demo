package com.google.mediapipe.examples.poselandmarker

import android.graphics.Bitmap
import java.nio.ByteBuffer

/**
 * Thin JNI wrapper around RKNN runtime for loading and running YOLO pose RKNN models.
 */
class RknnRunner(modelBuffer: ByteBuffer) : AutoCloseable {

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
    }

    fun run(inputBuffer: ByteBuffer): FloatArray {
        check(handle != 0L) { "RKNN runtime not initialized" }
        require(inputBuffer.isDirect) { "Input buffer must be direct" }
        val inputSize = inputBuffer.remaining()
        return nativeRun(handle, inputBuffer, inputSize)
    }

    /**
     * Efficiently runs inference by passing raw ARGB pixels.
     * Conversion to model input format (e.g. FP16 RGB) happens in Native/C++ to save time.
     */
    fun runPixels(pixels: IntArray): FloatArray {
        check(handle != 0L) { "RKNN runtime not initialized" }
        return nativeRunPixels(handle, pixels)
    }

    /**
     * Efficiently runs inference using direct Bitmap access (zero-copy if possible).
     */
    fun runBitmapWithNms(bitmap: Bitmap, detectThreshold: Float, nmsThreshold: Float): FloatArray {
        check(handle != 0L) { "RKNN runtime not initialized" }
        return nativeRunBitmapWithNms(handle, bitmap, detectThreshold, nmsThreshold)
    }

    override fun close() {
        if (handle != 0L) {
            nativeRelease(handle)
            handle = 0
        }
    }

    protected fun finalize() {
        close()
    }

    private external fun nativeInit(modelBuffer: ByteBuffer, modelSize: Int): Long
    private external fun nativeGetOutputShape(handle: Long): IntArray
    private external fun nativeRun(handle: Long, inputBuffer: ByteBuffer, inputSize: Int): FloatArray
    private external fun nativeRunPixels(handle: Long, pixels: IntArray): FloatArray
    private external fun nativeRunBitmapWithNms(handle: Long, bitmap: Bitmap, detectThresh: Float, nmsThresh: Float): FloatArray
    private external fun nativeRelease(handle: Long)

    companion object {
        init {
            NativeLibLoader.ensureLoaded()
        }
    }
}
