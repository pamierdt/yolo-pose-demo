/*
 * Copyright 2023 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the a specific language governing permissions and
 * limitations under the License.
 */
package com.google.mediapipe.examples.poselandmarker

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.AttributeSet
import android.util.Log
import android.view.View
import androidx.core.content.ContextCompat
import com.google.mediapipe.tasks.vision.core.RunningMode
import kotlin.math.max
import kotlin.math.min

class OverlayView(context: Context?, attrs: AttributeSet?) :
    View(context, attrs) {

    private var results: PoseResult? = null
    private var pointPaint = Paint()
    private var linePaint = Paint()
    private var boxPaint = Paint()
    private var textPaint = Paint()

    private var scaleFactor: Float = 1f
    private var imageWidth: Int = 1
    private var imageHeight: Int = 1
    private var imageTranslateX: Float = 0f
    private var imageTranslateY: Float = 0f
    private var minKeypointScore: Float = PoseLandmarkerHelper.DEFAULT_KEYPOINT_CONFIDENCE
    private var debugLogsRemaining = 3
    private var lastLogSignature: String? = null


    init {
        initPaints()
    }

    fun clear() {
        results = null
        pointPaint.reset()
        linePaint.reset()
        boxPaint.reset()
        invalidate()
        initPaints()
    }

    private fun initPaints() {
        linePaint.color =
            ContextCompat.getColor(context!!, R.color.mp_color_primary)
        linePaint.strokeWidth = LANDMARK_STROKE_WIDTH
        linePaint.style = Paint.Style.STROKE

        pointPaint.color = Color.YELLOW
        pointPaint.strokeWidth = LANDMARK_STROKE_WIDTH
        pointPaint.style = Paint.Style.FILL

        boxPaint.color = Color.argb(120, 3, 169, 244)
        boxPaint.strokeWidth = LANDMARK_STROKE_WIDTH / 2
        boxPaint.style = Paint.Style.STROKE
        textPaint.color = Color.RED
        textPaint.textSize = 48f
        textPaint.isAntiAlias = true
        textPaint.textAlign = Paint.Align.CENTER
    }

    override fun draw(canvas: Canvas) {
        super.draw(canvas)
        results?.poses?.forEachIndexed { index, pose ->
            pose.keypoints.forEach { keypoint ->
                if (keypoint.score >= minKeypointScore) {
                    canvas.drawPoint(
                        keypoint.x * imageWidth * scaleFactor + imageTranslateX,
                        keypoint.y * imageHeight * scaleFactor + imageTranslateY,
                        pointPaint
                    )
                }
            }

            KEYPOINT_CONNECTIONS.forEach { (start, end) ->
                val startKp = pose.keypoints.getOrNull(start)
                val endKp = pose.keypoints.getOrNull(end)
                if (startKp != null && endKp != null &&
                    startKp.score >= minKeypointScore &&
                    endKp.score >= minKeypointScore
                ) {
                    canvas.drawLine(
                        startKp.x * imageWidth * scaleFactor + imageTranslateX,
                        startKp.y * imageHeight * scaleFactor + imageTranslateY,
                        endKp.x * imageWidth * scaleFactor + imageTranslateX,
                        endKp.y * imageHeight * scaleFactor + imageTranslateY,
                        linePaint
                    )
                }
            }

            val boxLeft = pose.boundingBox.left * imageWidth * scaleFactor + imageTranslateX
            val boxTop = pose.boundingBox.top * imageHeight * scaleFactor + imageTranslateY
            val boxRight = pose.boundingBox.right * imageWidth * scaleFactor + imageTranslateX
            val boxBottom = pose.boundingBox.bottom * imageHeight * scaleFactor + imageTranslateY

            canvas.drawRect(boxLeft, boxTop, boxRight, boxBottom, boxPaint)

            val poseId = pose.id.takeIf { it > 0 } ?: index + 1
            val labelX = boxLeft + LANDMARK_STROKE_WIDTH
            val labelY = max(boxTop + textPaint.textSize, textPaint.textSize)
            val previousAlign = textPaint.textAlign
            textPaint.textAlign = Paint.Align.LEFT
            canvas.drawText("ID $poseId", labelX, labelY, textPaint)
            textPaint.textAlign = previousAlign

            // Draw counter for each pose
            if (pose.jumpCount > 0) {
                val cx = (boxLeft + boxRight) / 2f
                val cy = (boxTop + boxBottom) / 2f
                
                val fm = textPaint.fontMetrics
                val textY = cy - (fm.ascent + fm.descent) / 2f
                
                canvas.drawText("Count: ${pose.jumpCount}", cx, textY, textPaint)
            }
        }
    }

    fun setResults(
        poseLandmarkerResults: PoseResult,
        imageHeight: Int,
        imageWidth: Int,
        runningMode: RunningMode = RunningMode.IMAGE,
        keypointScoreThreshold: Float = PoseLandmarkerHelper.DEFAULT_KEYPOINT_CONFIDENCE
    ) {
        results = poseLandmarkerResults

        this.imageHeight = imageHeight
        this.imageWidth = imageWidth
        this.minKeypointScore = keypointScoreThreshold

        scaleFactor = when (runningMode) {
            RunningMode.IMAGE,
            RunningMode.VIDEO -> {
                min(width * 1f / imageWidth, height * 1f / imageHeight)
            }
            RunningMode.LIVE_STREAM -> {
                // PreviewView is in FILL_START mode. So we need to scale up the
                // landmarks to match with the size that the captured images will be
                // displayed.
                max(width * 1f / imageWidth, height * 1f / imageHeight)
            }
        }

        val scaledWidth = imageWidth * scaleFactor
        val scaledHeight = imageHeight * scaleFactor

        if (runningMode == RunningMode.LIVE_STREAM) {
            // PreviewView fillStart anchors the image at top/left, so we keep translations at 0
            imageTranslateX = 0f
            imageTranslateY = 0f
        } else {
            imageTranslateX = (width - scaledWidth) / 2f
            imageTranslateY = (height - scaledHeight) / 2f
        }

        if (debugLogsRemaining > 0) {
            val signature = "$runningMode-$width-$height-$imageWidth-$imageHeight-$scaleFactor-$imageTranslateX-$imageTranslateY"
            if (signature != lastLogSignature) {
                lastLogSignature = signature
                debugLogsRemaining--
            }
        }

        invalidate()
    }



    companion object {
        private const val TAG = "OverlayView"
        private const val LANDMARK_STROKE_WIDTH = 12F
        private val KEYPOINT_CONNECTIONS = listOf(
            0 to 1, 0 to 2, 1 to 3, 2 to 4,
            0 to 5, 0 to 6, 5 to 7, 7 to 9, 6 to 8, 8 to 10,
            5 to 6, 5 to 11, 6 to 12, 11 to 12, 11 to 13,
            13 to 15, 12 to 14, 14 to 16
        )
    }
}
