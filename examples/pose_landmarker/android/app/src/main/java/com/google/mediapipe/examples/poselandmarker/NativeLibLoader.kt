package com.yolo.pose.demo

import android.util.Log

object NativeLibLoader {
    private const val TAG = "NativeLibLoader"
    @Volatile private var loaded = false

    fun ensureLoaded() {
        if (loaded) return
        synchronized(this) {
            if (loaded) return

            // Optional Rockchip RGA lib; load when it is packaged.
            try {
                System.loadLibrary("rga_dt")
            } catch (e: UnsatisfiedLinkError) {
                Log.w(TAG, "librga_dt.so not bundled; skipping RGA preload (${e.message})")
            }

            System.loadLibrary("rknnrt_dt")
            System.loadLibrary("rknn_jni")
            loaded = true
        }
    }
}
