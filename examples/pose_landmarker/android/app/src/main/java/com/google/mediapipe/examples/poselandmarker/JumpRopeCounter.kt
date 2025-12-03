package com.google.mediapipe.examples.poselandmarker

class JumpRopeCounter(
    minIntervalMs: Float = 300f
) : AutoCloseable {

    private var handle: Long = 0

    init {
        handle = nativeCreate(minIntervalMs)
        if (handle == 0L) throw IllegalStateException("Failed to create JumpRopeCounter")
    }

    fun update(shoulderY: Float, hipY: Float, ankleY: Float, timestampMs: Double): Int {
        check(handle != 0L)
        return nativeUpdate(handle, shoulderY, hipY, ankleY, timestampMs)
    }

    fun getCount(): Int {
        check(handle != 0L)
        return nativeGetCount(handle)
    }

    fun getGroundY(): Float {
        check(handle != 0L)
        return nativeGetGroundY(handle)
    }

    fun getState(): Int {
        check(handle != 0L)
        return nativeGetState(handle)
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

    private external fun nativeCreate(minIntervalMs: Float): Long
    private external fun nativeUpdate(handle: Long, shoulderY: Float, hipY: Float, ankleY: Float, timestampMs: Double): Int
    private external fun nativeGetCount(handle: Long): Int
    private external fun nativeGetGroundY(handle: Long): Float
    private external fun nativeGetState(handle: Long): Int
    private external fun nativeReset(handle: Long)
    private external fun nativeRelease(handle: Long)

    companion object {
        init {
            NativeLibLoader.ensureLoaded()
        }
    }
}
