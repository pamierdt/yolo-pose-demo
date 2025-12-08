package com.yolo.pose.demo

import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import android.media.MediaMetadataRetriever
import android.net.Uri
import android.util.Log
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt
import kotlin.math.pow

/**
 * JumpRopeVideoProcessor - 跳绳视频处理器
 * 
 * 功能 / Features:
 * - 多人追踪 (Multi-person tracking)
 * - IoU 匹配算法 (IoU matching algorithm)
 * - 跳绳计数集成 (Jump rope counting integration)
 * - 视频帧处理 (Video frame processing)
 * 
 * 算法流程 / Algorithm Flow:
 * 1. 提取视频帧 (Extract video frames)
 * 2. 姿态估计 (Pose estimation)
 * 3. 多人追踪 (Multi-person tracking with IoU)
 * 4. 跳绳计数 (Jump rope counting per person)
 * 
 * @param context Android 上下文 / Android context
 * @param poseHelper 姿态估计辅助类 / Pose estimation helper
 */
class JumpRopeVideoProcessor(
    private val context: Context,
    private val poseHelper: PoseLandmarkerHelper
) {
    companion object {
        private const val TAG = "JumpRopeVideoProcessor"
        private const val IO_TIMEOUT_MS = 3000L
        
        // 追踪参数 / Tracking parameters
        private const val IOU_THRESHOLD = 0.3f        // IoU 匹配阈值 / IoU matching threshold
        private const val MAX_MISSING_FRAMES = 5      // 最大丢失帧数 / Max missing frames before track removal
        private const val TRACK_INIT_MISSING = 10     // 初始化时的丢失帧阈值 / Initial missing frame threshold
    }

    // 每个追踪对象的跳绳计数器 / Jump rope counter for each tracked person
    // Key: trackId, Value: JumpRopeCounter instance
    private val ropeCounters: MutableMap<Int, JumpRopeCounter> = mutableMapOf()
    
    // 活跃的追踪对象列表 / Active tracking objects list
    private val activeTracks: MutableList<Track> = mutableListOf()
    
    // 下一个追踪ID / Next track ID
    private var nextTrackId = 1

    /**
     * 追踪对象数据类 / Track data class
     * @param id 追踪ID（唯一标识）/ Track ID (unique identifier)
     * @param rect 边界框 / Bounding box
     * @param missingFrames 连续丢失帧数 / Consecutive missing frames
     * @param lastHipY 上一次的髋部Y坐标（用于距离匹配）/ Last hip Y coordinate
     * @param lastHipX 上一次的髋部X坐标（用于距离匹配）/ Last hip X coordinate
     */
    private data class Track(
        val id: Int, 
        var rect: RectF, 
        var missingFrames: Int,
        var lastHipX: Float = 0f,
        var lastHipY: Float = 0f
    )


    /**
     * 处理视频并返回每帧的姿态结果
     * Process video and return pose results for each frame
     * 
     * @param videoUri 视频URI / Video URI
     * @param inferenceIntervalMs 推理间隔 (毫秒) / Inference interval (ms)
     * @param progressCallback 进度回调 (当前帧, 总帧数, 当前总跳绳数) / Progress callback
     * @return 结果包 / ResultBundle
     */
    fun processVideo(
        videoUri: Uri, 
        inferenceIntervalMs: Long = 300,
        progressCallback: ((Int, Int, Int) -> Unit)? = null
    ): PoseLandmarkerHelper.ResultBundle? {
        val startTime = System.currentTimeMillis()
        
        // ========== 重置状态 / Reset State ==========
        // 释放旧的计数器并清空所有追踪数据
        // Release old counters and clear all tracking data
        ropeCounters.values.forEach { it.close() }
        ropeCounters.clear()
        activeTracks.clear()
        nextTrackId = 1
        Log.i(TAG, "State reset complete")

        // ========== 初始化视频读取器 / Initialize Video Retriever ==========
        val retriever = MediaMetadataRetriever()
        try {
            retriever.setDataSource(context, videoUri)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to set data source: ${e.message}", e)
            return null
        }
        
        val videoLengthMs = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)?.toLong()
        val firstFrame = retriever.getFrameAtTime(0)
        val width = firstFrame?.width
        val height = firstFrame?.height
        
        if (videoLengthMs == null || width == null || height == null) {
            Log.e(TAG, "Failed to get video metadata")
            retriever.release()
            return null
        }

        // ========== 提取视频元数据 / Extract Video Metadata ==========
        // val durationStr = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)
        // val videoLengthMs = durationStr?.toLongOrNull() ?: 0L
        
        // 获取帧率，默认30fps / Get frame rate, default 30fps
        val frameRateStr = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_CAPTURE_FRAMERATE)
        val frameRate = frameRateStr?.toFloatOrNull() ?: 30f
        val intervalMs = (1000f / frameRate).toLong()
        
        val numberOfFrames = videoLengthMs.div(inferenceIntervalMs).toInt()
        val results = mutableListOf<PoseResult>()
        
        Log.i(TAG, "========== Video Processing Started ==========")
        Log.i(TAG, "Duration: ${videoLengthMs}ms, FPS: $frameRate, Frames: $numberOfFrames, Interval: ${intervalMs}ms")

        // ========== 逐帧处理 / Frame-by-Frame Processing ==========
        var totalPoseTime = 0L
        var totalTrackingTime = 0L
        
        for (i in 0..numberOfFrames) {
            val timestampMs = i * inferenceIntervalMs
            val frame = retriever.getFrameAtTime(timestampMs * 1000, MediaMetadataRetriever.OPTION_CLOSEST)
            
            if (frame == null) {
                Log.w(TAG, "Frame $i: Failed to retrieve at ${timestampMs}ms")
                results.add(PoseResult(emptyList()))
                continue
            }
            
            // 确保 ARGB_8888 格式 / Ensure ARGB_8888 format
            val argbFrame = if (frame.config == Bitmap.Config.ARGB_8888) {
                frame
            } else {
                frame.copy(Bitmap.Config.ARGB_8888, false).also { frame.recycle() }
            }

            // ========== 姿态估计 / Pose Estimation ==========
            val poseStartTime = System.currentTimeMillis()
            val resultBundle = poseHelper.runPoseEstimation(argbFrame, argbFrame.width, argbFrame.height)
            val poseTime = System.currentTimeMillis() - poseStartTime
            totalPoseTime += poseTime
            
            if (resultBundle != null && resultBundle.results.isNotEmpty()) {
                val poses = resultBundle.results.first().poses
                
                // ========== 多人追踪与计数 / Multi-Person Tracking & Counting ==========
                val trackingStartTime = System.currentTimeMillis()
                val updatedPoses = updateCounterFromPoses(poses, timestampMs.toDouble())
                val trackingTime = System.currentTimeMillis() - trackingStartTime
                totalTrackingTime += trackingTime
                
                results.add(PoseResult(updatedPoses))
                
                if (i % 30 == 0) {  // 每30帧记录一次 / Log every 30 frames
                    Log.d(TAG, "Frame $i: ${poses.size} poses, pose=${poseTime}ms, tracking=${trackingTime}ms")
                }
            } else {
                results.add(PoseResult(emptyList()))
            }
            
            argbFrame.recycle()
            
            // Calculate total jump count across all active counters
            val totalJumpCount = ropeCounters.values.sumOf { it.getCount() }
            progressCallback?.invoke(i, numberOfFrames, totalJumpCount)
        }
        
        retriever.release()
        
        // ========== 性能统计 / Performance Statistics ==========
        val totalTime = System.currentTimeMillis() - startTime
        val avgPoseTime = if (numberOfFrames > 0) totalPoseTime / numberOfFrames else 0
        val avgTrackingTime = if (numberOfFrames > 0) totalTrackingTime / numberOfFrames else 0
        
        Log.i(TAG, "========== Video Processing Complete ==========")
        Log.i(TAG, "Total: ${totalTime}ms, Frames: ${results.size}")
        Log.i(TAG, "Avg Pose: ${avgPoseTime}ms, Avg Tracking: ${avgTrackingTime}ms")
        Log.i(TAG, "Active Tracks: ${activeTracks.size}, Total Counters: ${ropeCounters.size}")
        
        val inferenceTimePerFrame = if (numberOfFrames > 0) totalTime / numberOfFrames else 0
        
        return PoseLandmarkerHelper.ResultBundle(
            results,
            inferenceTimePerFrame,
            height,
            width,
            avgPoseTime
        )
    }

    /**
     * 从姿态列表更新计数器（多人追踪核心逻辑）
     * Update counters from pose list (core multi-person tracking logic)
     * 
     * 算法步骤 / Algorithm Steps:
     * 1. 处理空帧情况 / Handle empty frames
     * 2. IoU 匹配现有追踪 / IoU matching with existing tracks
     * 3. 更新丢失追踪 / Update missing tracks
     * 4. 创建新追踪 / Create new tracks
     * 5. 更新跳绳计数 / Update jump rope counts
     * 
     * @param poses 当前帧检测到的姿态列表 / Detected poses in current frame
     * @param timestampMs 当前时间戳 / Current timestamp
     * @return 更新后的姿态列表（包含trackId和jumpCount）/ Updated poses with trackId and jumpCount
     */
    private fun updateCounterFromPoses(poses: List<Pose>, timestampMs: Double): List<Pose> {
        // ========== 1. 处理空帧 / Handle Empty Frame ==========
        if (poses.isEmpty()) {
            // 所有追踪对象的丢失帧数+1 / Increment missing frames for all tracks
            activeTracks.forEach { it.missingFrames++ }
            // 移除丢失时间过长的追踪 / Remove tracks missing for too long
            val removedCount = activeTracks.count { it.missingFrames > TRACK_INIT_MISSING }
            activeTracks.removeAll { it.missingFrames > TRACK_INIT_MISSING }
            if (removedCount > 0) {
                Log.d(TAG, "Empty frame: Removed $removedCount lost tracks")
            }
            return emptyList()
        }

        // ========== 2. 双重匹配逻辑 (IoU + 关键点距离) / Dual Matching Logic (IoU + Keypoint Distance) ==========
        // 构建匹配矩阵：计算所有追踪对象与所有检测框的匹配分数
        // Build matching matrix: Calculate match score between all tracks and all detections
        val matchMatrix = mutableListOf<Triple<Int, Int, Float>>()
        
        activeTracks.forEachIndexed { trackIdx, track ->
            poses.forEachIndexed { poseIdx, pose ->
                val iouVal = iou(track.rect, pose.boundingBox)
                
                // 计算关键点距离分数 (基于髋部中心)
                // Calculate keypoint distance score (based on hip center)
                val distScore = calculateDistanceScore(track, pose)
                
                // 综合评分: 60% IoU + 40% 距离
                // Combined score: 60% IoU + 40% Distance
                val matchScore = 0.6f * iouVal + 0.4f * distScore
                
                // 降低匹配阈值以适应综合评分 (0.3 -> 0.25)
                if (matchScore > 0.25f) {
                    matchMatrix.add(Triple(trackIdx, poseIdx, matchScore))
                }
            }
        }

        // 按分数降序排序
        // Sort by score descending
        matchMatrix.sortByDescending { it.third }
        
        if (matchMatrix.isNotEmpty()) {
            Log.d(TAG, "Dual matching: ${matchMatrix.size} potential matches, best score=${matchMatrix.first().third}")
        }

        // 贪心匹配
        // Greedy matching
        val matchedTrackIndices = mutableSetOf<Int>()
        val assignedPoses = mutableSetOf<Int>()
        val poseToTrackMap = mutableMapOf<Int, Int>()

        for ((trackIdx, poseIdx, score) in matchMatrix) {
            if (trackIdx in matchedTrackIndices || poseIdx in assignedPoses) continue
            
            val track = activeTracks[trackIdx]
            track.rect = poses[poseIdx].boundingBox
            // 更新追踪对象的关键点中心（用于下一帧距离计算）
            updateTrackKeypoints(track, poses[poseIdx])
            
            track.missingFrames = 0
            matchedTrackIndices.add(trackIdx)
            assignedPoses.add(poseIdx)
            poseToTrackMap[poseIdx] = track.id
            
            Log.v(TAG, "Matched: track${track.id} <-> pose$poseIdx (Score=$score)")
        }

        // ========== 3. 更新丢失的追踪 / Update Missing Tracks ==========
        for (i in activeTracks.indices) {
            if (i !in matchedTrackIndices) {
                activeTracks[i].missingFrames++
            }
        }

        // 移除丢失时间过长的追踪 / Remove lost tracks
        val lostTrackIds = activeTracks.filter { it.missingFrames > MAX_MISSING_FRAMES }.map { it.id }
        activeTracks.removeAll { it.missingFrames > MAX_MISSING_FRAMES }
        
        // 释放对应的计数器 / Release corresponding counters
        lostTrackIds.forEach { trackId ->
            ropeCounters[trackId]?.close()
            ropeCounters.remove(trackId)
            Log.d(TAG, "Track $trackId lost (missing frames > $MAX_MISSING_FRAMES)")
        }

        // ========== 4. 创建新追踪 / Create New Tracks ==========
        poses.forEachIndexed { poseIdx, pose ->
            if (poseIdx !in assignedPoses) {
                val newId = nextTrackId++
                val newTrack = Track(newId, pose.boundingBox, 0)
                updateTrackKeypoints(newTrack, pose) // 初始化关键点
                activeTracks.add(newTrack)
                ropeCounters[newId] = JumpRopeCounter()
                poseToTrackMap[poseIdx] = newId
                Log.i(TAG, "New track created: ID=$newId, bbox=${pose.boundingBox}")
            }
        }

        // ========== 5. 更新跳绳计数 / Update Jump Rope Counters ==========
        val updatedPoses = poses.mapIndexed { index, pose ->
            val trackId = poseToTrackMap[index] ?: 0
            val counter = ropeCounters[trackId]
            var currentCount = counter?.getCount() ?: 0

            if (counter != null) {
                // 提取关键点 / Extract keypoints
                val leftHip = pose.keypoints.getOrNull(11)
                val rightHip = pose.keypoints.getOrNull(12)
                val leftShoulder = pose.keypoints.getOrNull(5)
                val rightShoulder = pose.keypoints.getOrNull(6)
                val leftAnkle = pose.keypoints.getOrNull(15)
                val rightAnkle = pose.keypoints.getOrNull(16)
                
                val minScore = poseHelper.minPoseTrackingConfidence

                // ========== 鲁棒的关键点提取 / Robust Keypoint Extraction ==========
                // 优先使用双侧平均值，单侧缺失时使用另一侧
                // Prefer bilateral average, fallback to single side if one is missing
                var hipY = 0f
                var shoulderY = 0f
                var ankleY = 0f
                var validHip = false
                var validShoulder = false
                var validAnkle = false

                // 髋部 / Hip
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

                // 肩部 / Shoulder
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
                
                // 踝部 / Ankle
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

                // 只有所有关键点都有效时才更新计数器
                // Only update counter when all keypoints are valid
                if (validHip && validShoulder && validAnkle) {
                    val prevCount = currentCount
                    currentCount = counter.update(shoulderY, hipY, ankleY, timestampMs)
                    
                    // 记录计数变化 / Log count changes
                    if (currentCount > prevCount) {
                        Log.i(TAG, "🎯 Track $trackId: Jump counted! Total=$currentCount")
                    }
                } else {
                    // 记录关键点缺失情况 / Log missing keypoints
                    val missing = mutableListOf<String>()
                    if (!validHip) missing.add("hip")
                    if (!validShoulder) missing.add("shoulder")
                    if (!validAnkle) missing.add("ankle")
                    Log.v(TAG, "Track $trackId: Missing keypoints: ${missing.joinToString(", ")}")
                }
            }

            pose.copy(id = trackId, jumpCount = currentCount)
        }

        return updatedPoses
    }

    /**
     * 计算关键点距离分数 (0-1)
     * Calculate keypoint distance score (0-1)
     */
    private fun calculateDistanceScore(track: Track, pose: Pose): Float {
        // 获取当前姿态的髋部中心
        val leftHip = pose.keypoints.getOrNull(11)
        val rightHip = pose.keypoints.getOrNull(12)
        
        var currentHipX = 0f
        var currentHipY = 0f
        var validPoints = 0
        
        if (leftHip != null && leftHip.score > 0.2f) {
            currentHipX += leftHip.x
            currentHipY += leftHip.y
            validPoints++
        }
        if (rightHip != null && rightHip.score > 0.2f) {
            currentHipX += rightHip.x
            currentHipY += rightHip.y
            validPoints++
        }
        
        if (validPoints == 0) return 0f // 无法计算距离
        
        currentHipX /= validPoints
        currentHipY /= validPoints
        
        // 如果是新追踪对象（没有历史数据），返回默认分
        if (track.lastHipX == 0f && track.lastHipY == 0f) return 0.5f
        
        // 计算欧氏距离
        val dx = currentHipX - track.lastHipX
        val dy = currentHipY - track.lastHipY
        val distance = sqrt(dx*dx + dy*dy)
        
        // 归一化距离分数：距离越小分数越高
        // 假设最大移动距离为 0.2 (屏幕宽度的20%)
        // Score = max(0, 1 - distance / 0.2)
        return max(0f, 1f - distance / 0.2f)
    }

    /**
     * 更新追踪对象的关键点信息
     */
    private fun updateTrackKeypoints(track: Track, pose: Pose) {
        val leftHip = pose.keypoints.getOrNull(11)
        val rightHip = pose.keypoints.getOrNull(12)
        
        var currentHipX = 0f
        var currentHipY = 0f
        var validPoints = 0
        
        if (leftHip != null && leftHip.score > 0.2f) {
            currentHipX += leftHip.x
            currentHipY += leftHip.y
            validPoints++
        }
        if (rightHip != null && rightHip.score > 0.2f) {
            currentHipX += rightHip.x
            currentHipY += rightHip.y
            validPoints++
        }
        
        if (validPoints > 0) {
            track.lastHipX = currentHipX / validPoints
            track.lastHipY = currentHipY / validPoints
        }
    }

    /**
     * 计算两个矩形的 IoU (Intersection over Union)
     * Calculate IoU (Intersection over Union) of two rectangles
     * 
     * IoU = 交集面积 / 并集面积
     * IoU = Intersection Area / Union Area
     * 
     * @param a 矩形A / Rectangle A
     * @param b 矩形B / Rectangle B
     * @return IoU 值 (0-1) / IoU value (0-1)
     */
    private fun iou(a: RectF, b: RectF): Float {
        // 计算面积 / Calculate areas
        val areaA = max(0f, a.right - a.left) * max(0f, a.bottom - a.top)
        val areaB = max(0f, b.right - b.left) * max(0f, b.bottom - b.top)
        
        // 计算交集 / Calculate intersection
        val interLeft = max(a.left, b.left)
        val interTop = max(a.top, b.top)
        val interRight = min(a.right, b.right)
        val interBottom = min(a.bottom, b.bottom)
        val interArea = max(0f, interRight - interLeft) * max(0f, interBottom - interTop)
        
        // 计算并集 / Calculate union
        val union = areaA + areaB - interArea
        
        return if (union <= 0f) 0f else interArea / union
    }
}
