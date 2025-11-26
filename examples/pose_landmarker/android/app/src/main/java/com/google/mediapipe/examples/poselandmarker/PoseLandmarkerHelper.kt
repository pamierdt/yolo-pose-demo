package com.google.mediapipe.examples.poselandmarker

import android.R.attr.value
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.RectF
import android.media.MediaMetadataRetriever
import android.net.Uri
import android.os.SystemClock
import android.util.Log
import android.util.Half
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

    fun detectLiveStream(imageProxy: ImageProxy, isFrontCamera: Boolean) {
        if (runningMode != RunningMode.LIVE_STREAM) {
            throw IllegalArgumentException(
                "detectLiveStream 只能在 RunningMode.LIVE_STREAM 模式下调用"
            )
        }

        val frameTime = SystemClock.uptimeMillis()
        val bitmapBuffer = imageProxyToBitmap(imageProxy)
        imageProxy.close()

        val matrix = Matrix().apply {
            postRotate(imageProxy.imageInfo.rotationDegrees.toFloat())
            if (isFrontCamera) {
                // Mirror around center to match the PreviewView behavior.
                postScale(-1f, 1f, bitmapBuffer.width / 2f, bitmapBuffer.height / 2f)
            }
        }

        val rotatedBitmap = Bitmap.createBitmap(
            bitmapBuffer, 0, 0, bitmapBuffer.width, bitmapBuffer.height, matrix, true
        )

        runPoseEstimation(rotatedBitmap)?.let { result ->
            poseLandmarkerHelperListener?.onResults(
                result.copy(
                    inferenceTime = SystemClock.uptimeMillis() - frameTime,
                    inputImageHeight = rotatedBitmap.height,
                    inputImageWidth = rotatedBitmap.width
                )
            )
        } ?: poseLandmarkerHelperListener?.onError("YOLO pose 推理失败")
    }

    private fun imageProxyToBitmap(imageProxy: ImageProxy): Bitmap {
        val plane = imageProxy.planes[0]
        val buffer = plane.buffer
        buffer.rewind()

        val pixelStride = plane.pixelStride
        val rowStride = plane.rowStride
        val rowPadding = (rowStride - pixelStride * imageProxy.width).coerceAtLeast(0)
        val bitmapWidth = imageProxy.width + rowPadding / pixelStride

        val paddedBitmap =
            Bitmap.createBitmap(bitmapWidth, imageProxy.height, Bitmap.Config.ARGB_8888)
        paddedBitmap.copyPixelsFromBuffer(buffer)

        // Crop away the padded columns if rowStride was wider than width * pixelStride.
        return Bitmap.createBitmap(paddedBitmap, 0, 0, imageProxy.width, imageProxy.height)
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
        return runPoseEstimation(image)
    }

    @VisibleForTesting
    fun runPoseEstimation(bitmap: Bitmap): ResultBundle? {
        val useRknn = rknnRunner != null
        val (inputBuffer, letterbox) = preprocess(bitmap, useRknn)

        val startTime = SystemClock.uptimeMillis()
        val (floatArray, outputShape) = if (rknnRunner != null) {
            val runner = rknnRunner ?: return null
            val output = runner.run(inputBuffer)
            output to runner.outputShape
        } else {
            val localInterpreter = interpreter ?: return null
            val outputShapeLocal = localInterpreter.getOutputTensor(0).shape()
            val outputSize = outputShapeLocal.fold(1) { acc, i -> acc * i }
            val outputBuffer =
                ByteBuffer.allocateDirect(4 * outputSize).order(ByteOrder.nativeOrder())
            localInterpreter.run(inputBuffer, outputBuffer)
            val floatArrayLocal = FloatArray(outputSize)
            outputBuffer.rewind()
            outputBuffer.asFloatBuffer().get(floatArrayLocal)
            floatArrayLocal to outputShapeLocal
        }
        val inferenceTime = SystemClock.uptimeMillis() - startTime

        // Log the first 100 raw float values from the model's output
        val rawOutputSample = floatArray.take(100).joinToString(", ")
        Log.d(
            TAG,
            "Raw model output: shape=${outputShape.joinToString("x")} size=${floatArray.size}, " +
                    "first 100 floats=[$rawOutputSample]"
        )

        val poseResult =
            decodeOutputs(floatArray, outputShape, letterbox, bitmap.width, bitmap.height)
        
        val topScore = poseResult.poses.maxOfOrNull { it.score } ?: 0f
        Log.d(
            TAG,
            "YOLO pose decoded: poses=${poseResult.poses.size}, topScore=$topScore"
        )

        return ResultBundle(listOf(poseResult), inferenceTime, bitmap.height, bitmap.width)
    }

    private fun preprocess(bitmap: Bitmap, forRknn: Boolean): Pair<ByteBuffer, LetterboxParams> {
        val inputBitmap =
            Bitmap.createBitmap(MODEL_INPUT_SIZE, MODEL_INPUT_SIZE, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(inputBitmap)
        val paint = Paint().apply { color = Color.DKGRAY }

        canvas.drawRect(
            0f, 0f, MODEL_INPUT_SIZE.toFloat(), MODEL_INPUT_SIZE.toFloat(), paint
        )

        val scale = min(
            MODEL_INPUT_SIZE.toFloat() / bitmap.width.toFloat(),
            MODEL_INPUT_SIZE.toFloat() / bitmap.height.toFloat()
        )
        val newWidth = (bitmap.width * scale).toInt()
        val newHeight = (bitmap.height * scale).toInt()
        val dx = (MODEL_INPUT_SIZE - newWidth) / 2f
        val dy = (MODEL_INPUT_SIZE - newHeight) / 2f

        val destRect = RectF(dx, dy, dx + newWidth, dy + newHeight)
        canvas.drawBitmap(bitmap, null, destRect, null)

        if (!loggedPreprocessInfo) {
            Log.d(
                TAG,
                "Preprocess: src=${bitmap.width}x${bitmap.height}, inputSize=${MODEL_INPUT_SIZE}, " +
                        "scale=${"%.4f".format(scale)}, paddedDest=${newWidth}x${newHeight}, pad=(${dx.toInt()},${dy.toInt()})"
            )
            loggedPreprocessInfo = true
        }

        val pixels = IntArray(MODEL_INPUT_SIZE * MODEL_INPUT_SIZE)
        inputBitmap.getPixels(
            pixels, 0, MODEL_INPUT_SIZE, 0, 0, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE
        )

        return if (forRknn) {
            // RKNN 模型输入被检测为 FP16 (type 1)，因此需要 2 字节/通道
            // 之前的日志显示归一化到 0-1 后模型输出全为 0，这暗示模型可能期望 0-255 的数据范围
            // 因此这里我们尝试传入 0-255 的 FP16 数值
            val byteBuffer =
                ByteBuffer.allocateDirect(2 * MODEL_INPUT_SIZE * MODEL_INPUT_SIZE * 3)
                    .order(ByteOrder.nativeOrder())
            for (pixel in pixels) {
                // 不除以 255f，直接传入像素值
                byteBuffer.putShort(Half.toHalf(((pixel shr 16) and 0xFF).toFloat())) // R
                byteBuffer.putShort(Half.toHalf(((pixel shr 8) and 0xFF).toFloat()))  // G
                byteBuffer.putShort(Half.toHalf((pixel and 0xFF).toFloat()))          // B
            }
            byteBuffer.rewind()
            byteBuffer to LetterboxParams(scale, dx, dy)
        } else {
            val byteBuffer =
                ByteBuffer.allocateDirect(4 * MODEL_INPUT_SIZE * MODEL_INPUT_SIZE * 3)
                    .order(ByteOrder.nativeOrder())
            for (pixel in pixels) {
                byteBuffer.putFloat(((pixel shr 16) and 0xFF) / 255f)
                byteBuffer.putFloat(((pixel shr 8) and 0xFF) / 255f)
                byteBuffer.putFloat((pixel and 0xFF) / 255f)
            }
            byteBuffer.rewind()
            byteBuffer to LetterboxParams(scale, dx, dy)
        }
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

        Log.d(
            TAG,
            "Decode RKNN/TFLite output: shape=${outputShape.joinToString("x")}, " +
                    "channelFirst=$channelFirst, candidateCount=$candidateCount, valuesPerCandidate=$valuesPerCandidate"
        )

        // Inspect score channel before thresholding (channel index 4)
        if (candidateCount > 0 && valuesPerCandidate > 4) {
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
            if (!logged) {
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
        Log.d(
            TAG,
            "Decode summary: accepted=$accepted, afterNms=${filtered.size}, thresh=$detectionThreshold"
        )

        if (accepted == 0 && candidateCount > 0 && valuesPerCandidate > 0) {
            val firstCandidate = (0 until valuesPerCandidate).joinToString(", ") { idx ->
                "%.3f".format(value(0, idx))
            }
            Log.d(TAG, "No candidates passed threshold; candidate[0] raw: [$firstCandidate]")
        }
        return PoseResult(filtered)
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
                Log.i(TAG, "使用 GPU delegate 进行推理")
                true
                true
            } catch (e: Exception) {
                Log.w(TAG, "GPU delegate 不可用，尝试回退", e)
                false
            }
        }

        fun tryNnapi(): Boolean {
            return try {
                options.setUseNNAPI(true)
                Log.i(TAG, "使用 NNAPI (NPU) delegate 进行推理")
                true
            } catch (e: Exception) {
                Log.w(TAG, "NNAPI delegate 不可用，回退到 CPU", e)
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
        const val DEFAULT_KEYPOINT_CONFIDENCE = 0.5F
        const val DEFAULT_NMS_THRESHOLD = 0.5F
        const val MODEL_INPUT_SIZE = 640

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
    val keypoints: List<PoseKeypoint>
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
