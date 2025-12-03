package com.google.mediapipe.examples.poselandmarker

import android.R.attr.value
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.Rect
import android.graphics.RectF
import android.media.MediaMetadataRetriever
import android.net.Uri
import android.os.SystemClock
import android.util.Log
import androidx.annotation.VisibleForTesting
import androidx.camera.core.ImageProxy
import com.google.mediapipe.tasks.vision.core.RunningMode
import java.io.File
import java.io.FileInputStream
import java.util.Locale
import org.tensorflow.lite.Delegate
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import java.io.FileNotFoundException
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min

/**
 * Helper for running YOLO pose models (TFLite by default, RKNN asset name)
 * in the same places the original MediaPipe PoseLandmarker helper was used.
 */
class PoseLandmarkerHelper(
    var minPoseDetectionConfidence: Float = DEFAULT_POSE_DETECTION_CONFIDENCE,
    var minPoseTrackingConfidence: Float = DEFAULT_KEYPOINT_CONFIDENCE,
    var minPosePresenceConfidence: Float = DEFAULT_NMS_THRESHOLD,
    var currentModel: Int = MODEL_YOLO11_TFLITE,
    var currentDelegate: Int = DELEGATE_GPU,
    var runningMode: RunningMode = RunningMode.LIVE_STREAM,
    val context: Context,
    val poseLandmarkerHelperListener: LandmarkerListener? = null
) {

    private var interpreter: Interpreter? = null
    private var gpuDelegate: Delegate? = null
    private var rknnRunner: RknnRunner? = null
    private var loggedPreprocessInfo = false
    private var inputBitmap: Bitmap? = null
    private var pixelBuffer: IntArray? = null
    private var debugLogging: Boolean = true
    // Map tracking ID to JumpRopeCounter
    private var ropeCounters: MutableMap<Int, JumpRopeCounter> = mutableMapOf()
    // Simple tracker state: List of (ID, lastBoundingBox, missingFrames)
    private data class Track(val id: Int, var rect: RectF, var missingFrames: Int)
    private var activeTracks: MutableList<Track> = mutableListOf()
    private var nextTrackId = 1

    private var lastCounterTs: Long = 0L
    private var lastRopeCount: Int = 0

    init {
        setupPoseLandmarker()
    }

    fun clearPoseLandmarker() {
        interpreter?.close()
        interpreter = null
        closeDelegates()
        rknnRunner?.close()
        rknnRunner = null
    }

    fun isClose(): Boolean = interpreter == null && rknnRunner == null

    fun setupPoseLandmarker() {
        clearPoseLandmarker()

        val modelName = when (currentModel) {
            MODEL_YOLO8_TFLITE -> MODEL_FILE_YOLO8_TFLITE
            MODEL_YOLO8_RKNN -> MODEL_FILE_YOLO8_RKNN
            MODEL_YOLO11_TFLITE -> MODEL_FILE_YOLO11_TFLITE
            MODEL_YOLO11_RKNN -> MODEL_FILE_YOLO11_RKNN
            else -> MODEL_FILE_YOLO11_TFLITE
        }

        val isRknn = modelName.lowercase(Locale.US).endsWith(".rknn")
        val model = try {
            loadModelFile(modelName)
        } catch (fnf: FileNotFoundException) {
            poseLandmarkerHelperListener?.onError(
                "未找到 YOLO pose 模型文件 $modelName，请将模型放到 assets 或外部存储。"
            )
            Log.e(TAG, "Model file missing", fnf)
            return
        } catch (io: IOException) {
            poseLandmarkerHelperListener?.onError(
                "读取 YOLO pose 模型失败，详情见日志。"
            )
            Log.e(TAG, "Failed reading model", io)
            return
        } catch (e: Exception) {
            poseLandmarkerHelperListener?.onError(
                "YOLO pose 初始化失败：${e.message}"
            )
            Log.e(TAG, "Model load failed", e)
            return
        }

        if (isRknn) {
            try {
                rknnRunner = RknnRunner(model)
                Log.i(TAG, "Initialized RKNN runtime for $modelName")
                return
            } catch (e: Exception) {
                poseLandmarkerHelperListener?.onError(
                    "初始化 RKNN 模型失败：${e.message}"
                )
                Log.e(TAG, "RKNN init failed", e)
            }
        }

        val options = buildInterpreterOptions()
        try {
            interpreter = Interpreter(model, options)

        } catch (e: IllegalArgumentException) {
            poseLandmarkerHelperListener?.onError(
                "模型格式不受支持，请检查 $modelName。"
            )
            Log.e(TAG, "Interpreter init failed", e)
        } catch (e: Exception) {
            poseLandmarkerHelperListener?.onError(
                "YOLO pose 初始化失败：${e.message}"
            )
            Log.e(TAG, "Interpreter init failed", e)
        }
    }

    private fun loadModelFile(modelName: String): ByteBuffer {
        val candidates = buildList {
            add(File(context.filesDir, modelName))
            context.getExternalFilesDir(null)?.let { add(File(it, modelName)) }
            add(File("/data/local/tmp/$modelName"))
            add(File("/sdcard/Download/$modelName"))
        }

        for (file in candidates) {
            if (file.exists()) {
                Log.i(TAG, "Loading model from filesystem: ${file.absolutePath}")
                FileInputStream(file).use { input ->
                    val bytes = input.readBytes()
                    return ByteBuffer.allocateDirect(bytes.size)
                        .order(ByteOrder.nativeOrder())
                        .apply {
                            put(bytes)
                            rewind()
                        }
                }
            }
        }

        Log.i(TAG, "Model not found on filesystem, falling back to assets: $modelName")
        context.assets.open(modelName).use { input ->
            val bytes = input.readBytes()
            return ByteBuffer.allocateDirect(bytes.size).order(ByteOrder.nativeOrder()).apply {
                put(bytes)
                rewind()
            }
        }
    }

    private var cachedImageBitmap: Bitmap? = null

    fun detectLiveStream(imageProxy: ImageProxy, isFrontCamera: Boolean) {
        if (runningMode != RunningMode.LIVE_STREAM) {
            throw IllegalArgumentException(
                "detectLiveStream 只能在 RunningMode.LIVE_STREAM 模式下调用"
            )
        }

        val frameTime = SystemClock.uptimeMillis()
        
        // Optimize: reuse bitmap and avoid cropping copy
        val (bitmapBuffer, validWidth, validHeight) = imageProxyToBitmapResult(imageProxy)
        
        val rotation = imageProxy.imageInfo.rotationDegrees
        if (debugLogging) Log.d(TAG, "Live frame: src=${validWidth}x${validHeight} rot=${rotation} front=${isFrontCamera}")
        imageProxy.close()

        runPoseEstimation(bitmapBuffer, validWidth, validHeight)?.let { result ->
            val transformed = if (result.results.isNotEmpty()) {
                transformForDisplay(result.results.first(), rotation, isFrontCamera)
            } else null
            
            // Use valid dimensions for display calculation
            val displayWidth = if (rotation == 90 || rotation == 270) validHeight else validWidth
            val displayHeight = if (rotation == 90 || rotation == 270) validWidth else validHeight
            
            if (debugLogging) Log.d(TAG, "Live result: poses=${(transformed?.poses ?: result.results.firstOrNull()?.poses ?: emptyList()).size} disp=${displayWidth}x${displayHeight}")

            val now = SystemClock.uptimeMillis()
            val dt = if (lastCounterTs == 0L) 0f else (now - lastCounterTs).toFloat()
            lastCounterTs = now
            val poses = transformed?.poses ?: result.results.firstOrNull()?.poses ?: emptyList()

            logPoseData(poses, "LiveStream")

            val algoStartTime = SystemClock.uptimeMillis()
            val updatedPoses = updateCounterFromPoses(poses, now.toDouble())
            val algoTime = SystemClock.uptimeMillis() - algoStartTime

            val finalPoseResult = if (transformed != null) {
                transformed.copy(poses = updatedPoses)
            } else {
                result.results.firstOrNull()?.copy(poses = updatedPoses)
            }

            poseLandmarkerHelperListener?.onResults(
                result.copy(
                    results = if (finalPoseResult != null) listOf(finalPoseResult) else emptyList(),
                    inputImageHeight = displayHeight,
                    inputImageWidth = displayWidth,
                    algorithmTime = algoTime
                )
            )
        } ?: poseLandmarkerHelperListener?.onError("YOLO pose 推理失败")
    }

    private fun logPoseData(poses: List<Pose>, prefix: String) {
        poses.forEachIndexed { index, pose ->
            Log.i(TAG, "$prefix Pose #$index (id=${pose.id}): score=${pose.score}, box=${pose.boundingBox}")
            pose.keypoints.forEachIndexed { kpIndex, kp ->
                Log.i(TAG, "  Keypoint #$kpIndex: x=${kp.x}, y=${kp.y}, score=${kp.score}")
            }
        }
    }



    // Helper to create a new counter
    private fun createCounter(): JumpRopeCounter {
        // New constructor only takes minIntervalMs (default 300)
        return JumpRopeCounter(minIntervalMs = 300f)
    }

    private fun updateCounterFromPoses(poses: List<Pose>, timestampMs: Double): List<Pose> {
        if (poses.isEmpty()) {
            activeTracks.forEach { it.missingFrames++ }
            activeTracks.removeAll { it.missingFrames > 10 }
            return emptyList()
        }

        val iouMatrix = mutableListOf<Triple<Int, Int, Float>>()
        activeTracks.forEachIndexed { trackIdx, track ->
            poses.forEachIndexed { poseIdx, pose ->
                val iouVal = iou(track.rect, pose.boundingBox)
                if (iouVal > 0.3f) {
                    iouMatrix.add(Triple(trackIdx, poseIdx, iouVal))
                }
            }
        }

        iouMatrix.sortByDescending { it.third }

        val matchedTrackIndices = mutableSetOf<Int>()
        val assignedPoses = mutableSetOf<Int>()
        val poseToTrackMap = mutableMapOf<Int, Int>()

        for ((trackIdx, poseIdx, _) in iouMatrix) {
            if (trackIdx in matchedTrackIndices || poseIdx in assignedPoses) continue
            val track = activeTracks[trackIdx]
            track.rect = poses[poseIdx].boundingBox
            track.missingFrames = 0
            matchedTrackIndices.add(trackIdx)
            assignedPoses.add(poseIdx)
            poseToTrackMap[poseIdx] = track.id
        }

        for (i in activeTracks.indices) {
            if (i !in matchedTrackIndices) {
                activeTracks[i].missingFrames++
            }
        }

        val lostTrackIds = activeTracks.filter { it.missingFrames > 5 }.map { it.id }
        activeTracks.removeAll { it.missingFrames > 5 }
        lostTrackIds.forEach { 
            ropeCounters.remove(it)
            if (debugLogging) Log.d(TAG, "Removed track $it due to missing frames")
        }

        poses.forEachIndexed { poseIdx, pose ->
            if (poseIdx !in assignedPoses) {
                val newId = nextTrackId++
                activeTracks.add(Track(newId, pose.boundingBox, 0))
                ropeCounters[newId] = createCounter()
                poseToTrackMap[poseIdx] = newId
                if (debugLogging) Log.d(TAG, "Created new track $newId for pose $poseIdx")
            }
        }

        val updatedPoses = poses.mapIndexed { index, pose ->
            val trackId = poseToTrackMap[index] ?: 0
            val counter = ropeCounters[trackId]
            var currentCount = counter?.getCount() ?: 0

            if (counter != null) {
                val leftHip = pose.keypoints.getOrNull(11)
                val rightHip = pose.keypoints.getOrNull(12)
                val leftShoulder = pose.keypoints.getOrNull(5)
                val rightShoulder = pose.keypoints.getOrNull(6)
                val leftAnkle = pose.keypoints.getOrNull(15)
                val rightAnkle = pose.keypoints.getOrNull(16)
                
                val minScore = minPoseTrackingConfidence

                // Robust Keypoint Extraction: prefer Avg, fallback to single
                var hipY = 0f
                var shoulderY = 0f
                var ankleY = 0f
                var validHip = false
                var validShoulder = false
                var validAnkle = false

                // Hip
                if (leftHip != null && rightHip != null && leftHip.score >= minScore && rightHip.score >= minScore) {
                    hipY = (leftHip.y + rightHip.y) / 2f
                    validHip = true
                } else if (leftHip != null && leftHip.score >= minScore) {
                    hipY = leftHip.y
                    validHip = true
                } else if (rightHip != null && rightHip.score >= minScore) {
                    hipY = rightHip.y
                    validHip = true
                }

                // Shoulder
                if (leftShoulder != null && rightShoulder != null && leftShoulder.score >= minScore && rightShoulder.score >= minScore) {
                    shoulderY = (leftShoulder.y + rightShoulder.y) / 2f
                    validShoulder = true
                } else if (leftShoulder != null && leftShoulder.score >= minScore) {
                    shoulderY = leftShoulder.y
                    validShoulder = true
                } else if (rightShoulder != null && rightShoulder.score >= minScore) {
                    shoulderY = rightShoulder.y
                    validShoulder = true
                }
                
                // Ankle
                if (leftAnkle != null && rightAnkle != null && leftAnkle.score >= minScore && rightAnkle.score >= minScore) {
                    ankleY = (leftAnkle.y + rightAnkle.y) / 2f
                    validAnkle = true
                } else if (leftAnkle != null && leftAnkle.score >= minScore) {
                    ankleY = leftAnkle.y
                    validAnkle = true
                } else if (rightAnkle != null && rightAnkle.score >= minScore) {
                    ankleY = rightAnkle.y
                    validAnkle = true
                }

                if (validHip && validShoulder && validAnkle) {
                    currentCount = counter.update(shoulderY, hipY, ankleY, timestampMs)
                    if (debugLogging) {
                        val state = counter.getState()
                        val groundY = counter.getGroundY()
                        Log.d(TAG, "Track $trackId [Valid]: " +
                                "S=%.3f, H=%.3f, A=%.3f | ".format(shoulderY, hipY, ankleY) +
                                "State=$state, Count=$currentCount, Ground=%.3f".format(groundY))
                    }
                } else {
                     if (debugLogging) {
                         Log.d(TAG, "Track $trackId [Skip]: Missing KPs " +
                                 "(H:$validHip S:$validShoulder A:$validAnkle) | " +
                                 "Scores: H(${leftHip?.score},${rightHip?.score}) " +
                                 "S(${leftShoulder?.score},${rightShoulder?.score}) " +
                                 "A(${leftAnkle?.score},${rightAnkle?.score})")
                     }
                }
            }

            pose.copy(id = trackId, jumpCount = currentCount)
        }

        this.processedPoses = updatedPoses
        return updatedPoses
    }
    
    // Temporary storage for the processed poses with IDs
    private var processedPoses: List<Pose> = emptyList()


    private fun transformForDisplay(src: PoseResult, rotation: Int, mirror: Boolean): PoseResult {
        fun rot(x: Float, y: Float): Pair<Float, Float> {
            val r = ((rotation % 360) + 360) % 360
            return when (r) {
                90 -> Pair(y, 1f - x)
                180 -> Pair(1f - x, 1f - y)
                270 -> Pair(1f - y, x)
                else -> Pair(x, y)
            }
        }
        fun apply(x: Float, y: Float): Pair<Float, Float> {
            val p = rot(x, y)
            val nx = if (mirror) 1f - p.first else p.first
            return Pair(nx, p.second)
        }
        val poses = src.poses.map { p ->
            val lt = apply(p.boundingBox.left, p.boundingBox.top)
            val rb = apply(p.boundingBox.right, p.boundingBox.bottom)
            val lb = apply(p.boundingBox.left, p.boundingBox.bottom)
            val rt = apply(p.boundingBox.right, p.boundingBox.top)
            val xs = listOf(lt.first, rb.first, lb.first, rt.first)
            val ys = listOf(lt.second, rb.second, lb.second, rt.second)
            val box = RectF(
                xs.minOrNull()!!.coerceIn(0f, 1f),
                ys.minOrNull()!!.coerceIn(0f, 1f),
                xs.maxOrNull()!!.coerceIn(0f, 1f),
                ys.maxOrNull()!!.coerceIn(0f, 1f)
            )
            val kps = p.keypoints.map { kp ->
                val t = apply(kp.x, kp.y)
                // Remove coerceIn to preserve off-screen coordinates for accurate jump detection
                PoseKeypoint(t.first, t.second, kp.score)
            }
            Pose(p.score, box, kps, p.id)
        }
        return PoseResult(withPoseIds(poses))
    }

    private fun imageProxyToBitmapResult(imageProxy: ImageProxy): Triple<Bitmap, Int, Int> {
        val plane = imageProxy.planes[0]
        val buffer = plane.buffer
        buffer.rewind()

        val pixelStride = plane.pixelStride
        val rowStride = plane.rowStride
        val rowPadding = (rowStride - pixelStride * imageProxy.width).coerceAtLeast(0)
        val bitmapWidth = imageProxy.width + rowPadding / pixelStride

        if (debugLogging) Log.d(TAG, "ImageProxy: src=${imageProxy.width}x${imageProxy.height} pixelStride=${pixelStride} rowStride=${rowStride} rowPadding=${rowPadding} paddedW=${bitmapWidth}")

        if (cachedImageBitmap == null || cachedImageBitmap?.width != bitmapWidth || cachedImageBitmap?.height != imageProxy.height) {
            cachedImageBitmap = Bitmap.createBitmap(bitmapWidth, imageProxy.height, Bitmap.Config.ARGB_8888)
        }
        val paddedBitmap = cachedImageBitmap!!
        paddedBitmap.copyPixelsFromBuffer(buffer)

        // Return the padded bitmap and the valid width/height
        return Triple(paddedBitmap, imageProxy.width, imageProxy.height)
    }

    fun detectVideoFile(videoUri: Uri, inferenceIntervalMs: Long): ResultBundle? {
        if (runningMode != RunningMode.VIDEO) {
            throw IllegalArgumentException(
                "detectVideoFile 只能在 RunningMode.VIDEO 模式下调用"
            )
        }

        val retriever = MediaMetadataRetriever()
        retriever.setDataSource(context, videoUri)
        val videoLengthMs =
            retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)?.toLong()
        val firstFrame = retriever.getFrameAtTime(0)
        val width = firstFrame?.width
        val height = firstFrame?.height
        if (videoLengthMs == null || width == null || height == null) {
            retriever.release()
            return null
        }

        val resultList = mutableListOf<PoseResult>()
        val numberOfFrameToRead = videoLengthMs.div(inferenceIntervalMs)
        val startTime = SystemClock.uptimeMillis()

        for (i in 0..numberOfFrameToRead) {
            val timestampMs = i * inferenceIntervalMs
            retriever
                .getFrameAtTime(
                    timestampMs * 1000,
                    MediaMetadataRetriever.OPTION_CLOSEST
                )
                ?.let { frame ->
                    val argb8888Frame =
                        if (frame.config == Bitmap.Config.ARGB_8888) frame
                        else frame.copy(Bitmap.Config.ARGB_8888, false)

                    runPoseEstimation(argb8888Frame)?.results?.firstOrNull()?.let {
                        logPoseData(it.poses, "Video Frame $i")
                        resultList.add(it)
                    }
                }
        }

        retriever.release()

        val inferenceTimePerFrame =
            (SystemClock.uptimeMillis() - startTime).div(max(1, numberOfFrameToRead))

        return ResultBundle(resultList, inferenceTimePerFrame, height, width)
    }

    fun detectImage(image: Bitmap): ResultBundle? {
        if (runningMode != RunningMode.IMAGE) {
            throw IllegalArgumentException(
                "detectImage 只能在 RunningMode.IMAGE 模式下调用"
            )
        }
        val result = runPoseEstimation(image)
        result?.results?.firstOrNull()?.poses?.let { logPoseData(it, "Image") }
        return result
    }

    @VisibleForTesting
    fun runPoseEstimation(bitmap: Bitmap, validWidth: Int = bitmap.width, validHeight: Int = bitmap.height): ResultBundle? {
        val startTime = SystemClock.uptimeMillis()
        
        // If RKNN runner is available, use the optimized path with C++ NMS
        if (rknnRunner != null) {
            val runner = rknnRunner!!
            val t0 = SystemClock.uptimeMillis()
            val (inputBitmap, letterbox) = preprocessToBitmap(bitmap, validWidth, validHeight)
            
            // Use new native method with built-in NMS and direct Bitmap access
            val t1 = SystemClock.uptimeMillis()
            val serializedOutput = runner.runBitmapWithNms(
                inputBitmap, 
                minPoseDetectionConfidence, 
                minPosePresenceConfidence
            )
            val t2 = SystemClock.uptimeMillis()
            val inferenceTime = SystemClock.uptimeMillis() - startTime
            val poseResult = parseSerializedOutput(serializedOutput, letterbox, validWidth, validHeight)
            val t3 = SystemClock.uptimeMillis()
            if (debugLogging) Log.d(TAG, "Kotlin Profile: Letterbox=${t1 - t0} ms, Native=${t2 - t1} ms, Parse=${t3 - t2} ms, Total=${t3 - t0} ms")
            if (debugLogging) Log.d(TAG, "RKNN Pose decoded: poses=${poseResult.poses.size}")
            return ResultBundle(listOf(poseResult), inferenceTime, validHeight, validWidth)
        }

        // TFLite fallback path
        val localInterpreter = interpreter ?: return null
        val (inputBuffer, letterbox) = preprocess(bitmap, validWidth, validHeight)
        val outputShapeLocal = localInterpreter.getOutputTensor(0).shape()
        val outputSize = outputShapeLocal.fold(1) { acc, i -> acc * i }
        val outputBuffer = ByteBuffer.allocateDirect(4 * outputSize).order(ByteOrder.nativeOrder())
        localInterpreter.run(inputBuffer, outputBuffer)
        val floatArrayLocal = FloatArray(outputSize)
        outputBuffer.rewind()
        outputBuffer.asFloatBuffer().get(floatArrayLocal)
        
        val inferenceTime = SystemClock.uptimeMillis() - startTime
        val poseResult = decodeOutputs(floatArrayLocal, outputShapeLocal, letterbox, validWidth, validHeight)
        return ResultBundle(listOf(poseResult), inferenceTime, validHeight, validWidth)
    }
    
    private fun parseSerializedOutput(
        data: FloatArray,
        letterbox: LetterboxParams,
        originalWidth: Int,
        originalHeight: Int
    ): PoseResult {
        if (data.isEmpty()) return PoseResult(emptyList())

        val count = data[0].toInt()
        val poses = mutableListOf<Pose>()
        var offset = 1

        if (count == 0) return PoseResult(emptyList())

        // Calculate values per pose: (total size - 1 count) / count
        val valuesPerPose = (data.size - 1) / count
        // C++ returns: score, x, y, w, h (5 floats) + kps (3 floats per kp)
        val numKeypoints = (valuesPerPose - 5) / 3

        for (i in 0 until count) {
            val score = data[offset++]
            val cxNorm = data[offset++]
            val cyNorm = data[offset++]
            val wNorm = data[offset++]
            val hNorm = data[offset++]

            // C++ returns normalized coords relative to MODEL_INPUT_SIZE.
            // Map back to original image using letterbox params.
            val realCx = (cxNorm * MODEL_INPUT_SIZE - letterbox.padX) / letterbox.scale
            val realCy = (cyNorm * MODEL_INPUT_SIZE - letterbox.padY) / letterbox.scale
            val realW = (wNorm * MODEL_INPUT_SIZE) / letterbox.scale
            val realH = (hNorm * MODEL_INPUT_SIZE) / letterbox.scale

            val left = (realCx - realW / 2f) / originalWidth
            val top = (realCy - realH / 2f) / originalHeight
            val right = (realCx + realW / 2f) / originalWidth
            val bottom = (realCy + realH / 2f) / originalHeight

            val boundingBox = RectF(
                max(0f, left),
                max(0f, top),
                min(1f, right),
                min(1f, bottom)
            )

            val keypoints = mutableListOf<PoseKeypoint>()
            for (k in 0 until numKeypoints) {
                val kxNorm = data[offset++]
                val kyNorm = data[offset++]
                val ks = data[offset++]

                val realKx = (kxNorm * MODEL_INPUT_SIZE - letterbox.padX) / letterbox.scale
                val realKy = (kyNorm * MODEL_INPUT_SIZE - letterbox.padY) / letterbox.scale

                keypoints.add(PoseKeypoint(
                    (realKx / originalWidth).coerceIn(0f, 1f),
                    (realKy / originalHeight).coerceIn(0f, 1f),
                    ks
                ))
            }
            poses.add(Pose(score, boundingBox, keypoints))
        }
        return PoseResult(withPoseIds(poses))
    }

    private fun preprocessToBitmap(bitmap: Bitmap, validWidth: Int, validHeight: Int): Pair<Bitmap, LetterboxParams> {
        if (inputBitmap == null || inputBitmap?.width != MODEL_INPUT_SIZE || inputBitmap?.height != MODEL_INPUT_SIZE) {
            inputBitmap = Bitmap.createBitmap(MODEL_INPUT_SIZE, MODEL_INPUT_SIZE, Bitmap.Config.ARGB_8888)
        }
        val currentInputBitmap = inputBitmap!!
        
        // Use eraseColor to clear background efficiently instead of drawing a rect
        currentInputBitmap.eraseColor(Color.DKGRAY)
        
        val canvas = Canvas(currentInputBitmap)
        
        val scale = min(
            MODEL_INPUT_SIZE.toFloat() / validWidth.toFloat(),
            MODEL_INPUT_SIZE.toFloat() / validHeight.toFloat()
        )
        val newWidth = (validWidth * scale).toInt()
        val newHeight = (validHeight * scale).toInt()
        val dx = (MODEL_INPUT_SIZE - newWidth) / 2f
        val dy = (MODEL_INPUT_SIZE - newHeight) / 2f

        val srcRect = Rect(0, 0, validWidth, validHeight)
        val destRect = RectF(dx, dy, dx + newWidth, dy + newHeight)
        canvas.drawBitmap(bitmap, srcRect, destRect, null)

        if (!loggedPreprocessInfo && debugLogging) {
            Log.d(
                TAG,
                "Preprocess: src=${validWidth}x${validHeight}, inputSize=${MODEL_INPUT_SIZE}, " +
                        "scale=${"%.4f".format(scale)}, paddedDest=${newWidth}x${newHeight}, pad=(${dx.toInt()},${dy.toInt()})"
            )
            loggedPreprocessInfo = true
        }
        return currentInputBitmap to LetterboxParams(scale, dx, dy)
    }

    private fun preprocessToPixels(bitmap: Bitmap, validWidth: Int, validHeight: Int): Pair<IntArray, LetterboxParams> {
        val (currentInputBitmap, letterbox) = preprocessToBitmap(bitmap, validWidth, validHeight)

        if (pixelBuffer == null || pixelBuffer?.size != MODEL_INPUT_SIZE * MODEL_INPUT_SIZE) {
            pixelBuffer = IntArray(MODEL_INPUT_SIZE * MODEL_INPUT_SIZE)
        }
        val pixels = pixelBuffer!!
        
        currentInputBitmap.getPixels(
            pixels, 0, MODEL_INPUT_SIZE, 0, 0, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE
        )
        return pixels to letterbox
    }

    private fun preprocess(bitmap: Bitmap, validWidth: Int = bitmap.width, validHeight: Int = bitmap.height): Pair<ByteBuffer, LetterboxParams> {
        val (pixels, letterbox) = preprocessToPixels(bitmap, validWidth, validHeight)
        val byteBuffer =
            ByteBuffer.allocateDirect(4 * MODEL_INPUT_SIZE * MODEL_INPUT_SIZE * 3)
                .order(ByteOrder.nativeOrder())
        for (pixel in pixels) {
            byteBuffer.putFloat(((pixel shr 16) and 0xFF) / 255f)
            byteBuffer.putFloat(((pixel shr 8) and 0xFF) / 255f)
            byteBuffer.putFloat((pixel and 0xFF) / 255f)
        }
        byteBuffer.rewind()
        return byteBuffer to letterbox
    }

    @VisibleForTesting
    fun decodeOutputs(
        output: FloatArray,
        outputShape: IntArray,
        letterbox: LetterboxParams,
        originalWidth: Int,
        originalHeight: Int
    ): PoseResult {
        // YOLO pose export usually gives [1,56,8400] or [1,8400,56]; some RKNN exports flatten to [8400,56].
        val shapeSize = outputShape.size
        val channelFirst = shapeSize >= 3 && outputShape[1] < outputShape[2]
        var candidateCount = 0
        var valuesPerCandidate = 0
        if (shapeSize == 2) {
            // Handle flattened [N, C] or [C, N]; use larger dim as candidate count.
            val d0 = outputShape[0]
            val d1 = outputShape[1]
            if (d0 > d1) {
                candidateCount = d0
                valuesPerCandidate = d1
            } else {
                candidateCount = d1
                valuesPerCandidate = d0
            }
        } else {
            valuesPerCandidate =
                if (channelFirst) outputShape.getOrNull(1) ?: 0 else outputShape.getOrNull(2) ?: outputShape.getOrNull(
                    1
                ) ?: 0
            candidateCount =
                if (channelFirst) outputShape.getOrNull(2) ?: 0 else outputShape.getOrNull(1) ?: 0
        }

        if (candidateCount == 0 || valuesPerCandidate == 0) {
            Log.w(
                TAG,
                "Unsupported output shape: ${outputShape.joinToString("x")} (candidateCount=$candidateCount, valuesPerCandidate=$valuesPerCandidate)"
            )
            return PoseResult(emptyList())
        }

        fun value(candidate: Int, idx: Int): Float {
            return if (channelFirst) {
                output[idx * candidateCount + candidate]
            } else {
                output[candidate * valuesPerCandidate + idx]
            }
        }

        if (debugLogging) {
            Log.d(
                TAG,
                "Decode RKNN/TFLite output: shape=${outputShape.joinToString("x")}, " +
                        "channelFirst=$channelFirst, candidateCount=$candidateCount, valuesPerCandidate=$valuesPerCandidate"
            )
        }

        // Inspect score channel before thresholding (channel index 4)
        if (debugLogging && candidateCount > 0 && valuesPerCandidate > 4) {
            var minScore = Float.POSITIVE_INFINITY
            var maxScore = Float.NEGATIVE_INFINITY
            val sampleCount = min(20, candidateCount)
            val scoreSamples = (0 until sampleCount).joinToString(", ") { idx ->
                val raw = value(idx, 4)
                minScore = min(minScore, raw)
                maxScore = max(maxScore, raw)
                "%.3f(raw=%.3f)".format(sigmoid(raw), raw)
            }
            Log.d(
                TAG,
                "Score channel sample (sigmoid(raw)=...): first $sampleCount -> [$scoreSamples], " +
                        "rawMin=$minScore rawMax=$maxScore"
            )
        }
        val numKeypoints = ((valuesPerCandidate - 5) / 3).coerceAtLeast(0)

        var logged = false
        val detectionThreshold = minPoseDetectionConfidence

        val candidates = mutableListOf<Pose>()
        var accepted = 0
        for (i in 0 until candidateCount) {
            val score = sigmoid(value(i, 4))
            if (score < detectionThreshold) continue
            accepted++

            // Log raw data for the first detected pose candidate
            if (debugLogging && !logged) {
                val rawData = (0 until valuesPerCandidate).map { value(i, it) }.joinToString(", ")
                Log.d(TAG, "--- Begin Raw Candidate Data ---")
                Log.d(TAG, "Candidate index: $i, Score: ${value(i, 4)} -> ${"%.4f".format(score)}")
                Log.d(TAG, "Raw values ($valuesPerCandidate total): [$rawData]")
                Log.d(TAG, "--- End Raw Candidate Data ---")
                logged = true
            }

            var centerX = value(i, 0)
            var centerY = value(i, 1)
            var width = value(i, 2)
            var height = value(i, 3)

            // Some exports return normalized coordinates [0,1], others in pixel space.
            if (width > 2f) { // Likely in pixel space
                 centerX = (centerX - letterbox.padX) / letterbox.scale
                 centerY = (centerY - letterbox.padY) / letterbox.scale
                 width /= letterbox.scale
                 height /= letterbox.scale
            } else { // Likely in normalized space
                 centerX = (centerX * MODEL_INPUT_SIZE - letterbox.padX) / letterbox.scale
                 centerY = (centerY * MODEL_INPUT_SIZE - letterbox.padY) / letterbox.scale
                 width = width * MODEL_INPUT_SIZE / letterbox.scale
                 height = height * MODEL_INPUT_SIZE / letterbox.scale
            }

            val left = (centerX - width / 2f) / originalWidth.toFloat()
            val top = (centerY - height / 2f) / originalHeight.toFloat()
            val right = (centerX + width / 2f) / originalWidth.toFloat()
            val bottom = (centerY + height / 2f) / originalHeight.toFloat()

            val boundingBox = RectF(
                max(0f, left),
                max(0f, top),
                min(1f, right),
                min(1f, bottom)
            )

            val keypoints = mutableListOf<PoseKeypoint>()
            for (k in 0 until numKeypoints) {
                var kpX = value(i, 5 + k * 3)
                var kpY = value(i, 5 + k * 3 + 1)
                
                if (kpX > 2f) { // Likely in pixel space
                    kpX = (kpX - letterbox.padX) / letterbox.scale
                    kpY = (kpY - letterbox.padY) / letterbox.scale
                } else { // Likely in normalized space
                    kpX = (kpX * MODEL_INPUT_SIZE - letterbox.padX) / letterbox.scale
                    kpY = (kpY * MODEL_INPUT_SIZE - letterbox.padY) / letterbox.scale
                }

                val finalX = kpX / originalWidth.toFloat()
                val finalY = kpY / originalHeight.toFloat()
                val kpScore = sigmoid(value(i, 5 + k * 3 + 2))
                
                keypoints.add(
                    PoseKeypoint(
                        finalX.coerceIn(0f, 1f),
                        finalY.coerceIn(0f, 1f),
                        kpScore
                    )
                )
            }

            candidates.add(Pose(score, boundingBox, keypoints))
        }

        val filtered = applyNms(candidates, minPosePresenceConfidence)
        if (debugLogging) {
            Log.d(
                TAG,
                "Decode summary: accepted=$accepted, afterNms=${filtered.size}, thresh=$detectionThreshold"
            )
        }

        if (accepted == 0 && candidateCount > 0 && valuesPerCandidate > 0) {
            val firstCandidate = (0 until valuesPerCandidate).joinToString(", ") { idx ->
                "%.3f".format(value(0, idx))
            }
            if (debugLogging) Log.d(TAG, "No candidates passed threshold; candidate[0] raw: [$firstCandidate]")
        }
        return PoseResult(withPoseIds(filtered))
    }

    private fun applyNms(poses: List<Pose>, iouThreshold: Float): List<Pose> {
        val sorted = poses.sortedByDescending { it.score }
        val selected = mutableListOf<Pose>()

        for (pose in sorted) {
            var shouldSelect = true
            for (picked in selected) {
                if (iou(pose.boundingBox, picked.boundingBox) > iouThreshold) {
                    shouldSelect = false
                    break
                }
            }
            if (shouldSelect) selected.add(pose)
        }
        return selected
    }

    private fun iou(a: RectF, b: RectF): Float {
        val areaA = max(0f, a.right - a.left) * max(0f, a.bottom - a.top)
        val areaB = max(0f, b.right - b.left) * max(0f, b.bottom - b.top)
        val interLeft = max(a.left, b.left)
        val interTop = max(a.top, b.top)
        val interRight = min(a.right, b.right)
        val interBottom = min(a.bottom, b.bottom)
        val interArea = max(0f, interRight - interLeft) * max(0f, interBottom - interTop)
        val union = areaA + areaB - interArea
        return if (union <= 0f) 0f else interArea / union
    }

    private fun withPoseIds(poses: List<Pose>): List<Pose> {
        return poses.mapIndexed { index, pose ->
            if (pose.id > 0) pose else pose.copy(id = index + 1)
        }
    }

    private fun sigmoid(x: Float): Float = (1f / (1f + exp(-x))).coerceIn(0f, 1f)

    private fun buildInterpreterOptions(): Interpreter.Options {
        val options = Interpreter.Options()

        fun tryGpu(): Boolean {
            return try {
                val delegateOptions = GpuDelegate.Options().apply {
                    inferencePreference =
                        GpuDelegate.Options.INFERENCE_PREFERENCE_SUSTAINED_SPEED
                }
                val delegate = GpuDelegate(delegateOptions) // Create GpuDelegate directly
                gpuDelegate = delegate
                options.addDelegate(delegate)
                if (debugLogging) Log.i(TAG, "使用 GPU delegate 进行推理")
                true
                true
            } catch (e: Exception) {
                if (debugLogging) Log.w(TAG, "GPU delegate 不可用，尝试回退", e)
                false
            }
        }

        fun tryNnapi(): Boolean {
            return try {
                options.setUseNNAPI(true)
                if (debugLogging) Log.i(TAG, "使用 NNAPI (NPU) delegate 进行推理")
                true
            } catch (e: Exception) {
                if (debugLogging) Log.w(TAG, "NNAPI delegate 不可用，回退到 CPU", e)
                false
            }
        }

        when (currentDelegate) {
            DELEGATE_GPU -> {
                if (!tryGpu()) {
                    tryNnapi()
                }
            }

            DELEGATE_NNAPI -> {
                if (!tryNnapi()) {
                    tryGpu()
                }
            }

            else -> {
                // 即便选择了 CPU，也尽量利用 GPU/NPU 提升速度
                if (!tryGpu()) {
                    tryNnapi()
                }
            }
        }
        return options
    }

    private fun closeDelegates() {
        try {
            (gpuDelegate as? AutoCloseable)?.close()
        } catch (e: Exception) {
            Log.w(TAG, "关闭 GPU delegate 失败", e)
        }
        gpuDelegate = null
    }

    companion object {
        const val TAG = "PoseLandmarkerHelper"

        const val DELEGATE_GPU = 0
        const val DELEGATE_NNAPI = 1
        const val DELEGATE_CPU = 2
        const val DEFAULT_POSE_DETECTION_CONFIDENCE = 0.6F
        const val DEFAULT_KEYPOINT_CONFIDENCE = 0.3F
        const val DEFAULT_NMS_THRESHOLD = 0.5F
        const val MODEL_INPUT_SIZE = 320

        const val MODEL_YOLO8_TFLITE = 0
        const val MODEL_YOLO8_RKNN = 1
        const val MODEL_YOLO11_TFLITE = 2
        const val MODEL_YOLO11_RKNN = 3

        const val MODEL_FILE_YOLO8_TFLITE = "yolov8n-pose_float32.tflite"
        const val MODEL_FILE_YOLO8_RKNN = "yolov8n-pose-rk3568.rknn"
        const val MODEL_FILE_YOLO11_TFLITE = "yolo11n-pose_float32.tflite"
        const val MODEL_FILE_YOLO11_RKNN = "yolo11n-pose-rk3568.rknn"
    }

    data class ResultBundle(
        val results: List<PoseResult>,
        val inferenceTime: Long,
        val inputImageHeight: Int,
        val inputImageWidth: Int,
        val algorithmTime: Long = 0L
    )

    interface LandmarkerListener {
        fun onError(error: String, errorCode: Int = 0)
        fun onResults(resultBundle: ResultBundle)
    }
}

data class PoseResult(
    val poses: List<Pose>
)

data class Pose(
    val score: Float,
    val boundingBox: RectF,
    val keypoints: List<PoseKeypoint>,
    val id: Int = 0,
    val jumpCount: Int = 0
)

data class PoseKeypoint(
    val x: Float,
    val y: Float,
    val score: Float
)

data class LetterboxParams(
    val scale: Float,
    val padX: Float,
    val padY: Float
)
