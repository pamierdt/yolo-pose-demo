package com.google.mediapipe.examples.poselandmarker

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
                System.loadLibrary("rga")
            } catch (e: UnsatisfiedLinkError) {
                Log.w(TAG, "librga.so not bundled; skipping RGA preload (${e.message})")
            }

            System.loadLibrary("rknnrt")
            System.loadLibrary("rknn_jni")
            loaded = true
        }
    }
}
