package com.google.mediapipe.examples.poselandmarker

class JumpRopeCounter(
    qPos: Float = 1e-3f,
    qVel: Float = 1e-3f,
    rMeas: Float = 1e-2f,
    alpha: Float = 0.2f,
    threshold: Float = 0.1f,
    minIntervalMs: Float = 300f
) : AutoCloseable {

    private var handle: Long = 0

    init {
        handle = nativeCreate(qPos, qVel, rMeas, alpha, threshold, minIntervalMs)
        if (handle == 0L) throw IllegalStateException("Failed to create JumpRopeCounter")
    }

    fun update(measurement: Float, dtMs: Float): Int {
        check(handle != 0L)
        return nativeUpdate(handle, measurement, dtMs)
    }

    fun getCount(): Int {
        check(handle != 0L)
        return nativeGetCount(handle)
    }

    fun getFiltered(): Float {
        check(handle != 0L)
        return nativeGetFiltered(handle)
    }

    fun reset() {
        check(handle != 0L)
        nativeReset(handle)
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

    private external fun nativeCreate(qPos: Float, qVel: Float, rMeas: Float, alpha: Float, threshold: Float, minIntervalMs: Float): Long
    private external fun nativeUpdate(handle: Long, measurement: Float, dtMs: Float): Int
    private external fun nativeGetCount(handle: Long): Int
    private external fun nativeGetFiltered(handle: Long): Float
    private external fun nativeReset(handle: Long)
    private external fun nativeRelease(handle: Long)

    companion object {
        init {
            System.loadLibrary("rknn_jni")
        }
    }
}
