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
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.yolo.pose.demo.fragment

import android.annotation.SuppressLint
import android.content.Context
import android.content.res.Configuration
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraManager
import android.hardware.camera2.CaptureRequest
import android.os.Bundle
import android.util.Log
import android.util.Range
import android.util.Size
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.AdapterView
import android.widget.Toast
import androidx.appcompat.app.AlertDialog
import androidx.camera.core.Preview
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Camera
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.camera2.interop.Camera2CameraInfo
import androidx.camera.camera2.interop.Camera2Interop
import androidx.annotation.OptIn
import androidx.camera.camera2.interop.ExperimentalCamera2Interop
import androidx.core.content.ContextCompat
import androidx.fragment.app.Fragment
import androidx.fragment.app.activityViewModels
import androidx.navigation.Navigation
import com.yolo.pose.demo.PoseLandmarkerHelper
import com.yolo.pose.demo.MainViewModel
import com.yolo.pose.demo.R
import com.yolo.pose.demo.databinding.FragmentCameraBinding
import com.google.mediapipe.tasks.vision.core.RunningMode
import java.util.Locale
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import kotlin.jvm.Volatile

class CameraFragment : Fragment(), PoseLandmarkerHelper.LandmarkerListener {

    companion object {
        private const val TAG = "YOLO Pose"
    }

    private var _fragmentCameraBinding: FragmentCameraBinding? = null

    private val fragmentCameraBinding
        get() = _fragmentCameraBinding!!

    private lateinit var poseLandmarkerHelper: PoseLandmarkerHelper
    private val viewModel: MainViewModel by activityViewModels()
    private var preview: Preview? = null
    private var imageAnalyzer: ImageAnalysis? = null
    private var camera: Camera? = null
    private var cameraProvider: ProcessCameraProvider? = null
    private var cameraFacing = CameraSelector.LENS_FACING_BACK
    private var selectedCameraId: String? = null  // Track selected camera by ID
    private val targetResolution = Size(960, 640)
    @Volatile private var lastAnalysisResolution: Size? = null
    @Volatile private var lastAnalysisRotation: Int = 0
    private var debugLogsRemaining = 5

    // Camera info data class
    data class CameraInfo(
        val id: String,
        val lensFacing: Int,
        val displayName: String
    )

    private val availableCameras = mutableListOf<CameraInfo>()

    /** Blocking ML operations are performed using this executor */
    private lateinit var backgroundExecutor: ExecutorService

    override fun onResume() {
        super.onResume()
        // Make sure that all permissions are still present, since the
        // user could have removed them while the app was in paused state.
        if (!PermissionsFragment.hasPermissions(requireContext())) {
            Navigation.findNavController(
                requireActivity(), R.id.fragment_container
            ).navigate(R.id.action_camera_to_permissions)
        }

        // Start the PoseLandmarkerHelper again when users come back
        // to the foreground.
        backgroundExecutor.execute {
            if(this::poseLandmarkerHelper.isInitialized) {
                if (poseLandmarkerHelper.isClose()) {
                    poseLandmarkerHelper.setupPoseLandmarker()
                }
            }
        }
    }

    override fun onPause() {
        super.onPause()
        if(this::poseLandmarkerHelper.isInitialized) {
            viewModel.setMinPoseDetectionConfidence(poseLandmarkerHelper.minPoseDetectionConfidence)
            viewModel.setMinPoseTrackingConfidence(poseLandmarkerHelper.minPoseTrackingConfidence)
            viewModel.setMinPosePresenceConfidence(poseLandmarkerHelper.minPosePresenceConfidence)
            viewModel.setDelegate(poseLandmarkerHelper.currentDelegate)
            viewModel.setUseQuantOutput(poseLandmarkerHelper.useQuantOutput)
            viewModel.setCacheableInput(poseLandmarkerHelper.cacheableInput)

            // Close the PoseLandmarkerHelper and release resources
            backgroundExecutor.execute { poseLandmarkerHelper.clearPoseLandmarker() }
        }
    }

    override fun onDestroyView() {
        _fragmentCameraBinding = null
        super.onDestroyView()

        // Shut down our background executor
        backgroundExecutor.shutdown()
        backgroundExecutor.awaitTermination(
            Long.MAX_VALUE, TimeUnit.NANOSECONDS
        )
    }

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        _fragmentCameraBinding =
            FragmentCameraBinding.inflate(inflater, container, false)

        return fragmentCameraBinding.root
    }

    @SuppressLint("MissingPermission")
    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        // Initialize our background executor
        backgroundExecutor = Executors.newSingleThreadExecutor()

        // Wait for the views to be properly laid out
        fragmentCameraBinding.viewFinder.post {
            // Set up the camera and its use cases
            setUpCamera()
        }

        // Create the PoseLandmarkerHelper that will handle the inference
        backgroundExecutor.execute {
            poseLandmarkerHelper = PoseLandmarkerHelper(
                context = requireContext(),
                runningMode = RunningMode.LIVE_STREAM,
                minPoseDetectionConfidence = viewModel.currentMinPoseDetectionConfidence,
                minPoseTrackingConfidence = viewModel.currentMinPoseTrackingConfidence,
                minPosePresenceConfidence = viewModel.currentMinPosePresenceConfidence,
                currentDelegate = viewModel.currentDelegate,
                currentModel = viewModel.currentModel,
                poseLandmarkerHelperListener = this
            )
            poseLandmarkerHelper.setPerfOptions(
                viewModel.currentUseQuantOutput,
                viewModel.currentCacheableInput
            )
            poseLandmarkerHelper.setJumpThresholds(
                viewModel.currentJumpUpThreshold,
                viewModel.currentJumpDownThreshold
            )
        }

        // Attach listeners to UI control widgets
        initBottomSheetControls()

        // Setup camera switch button
        setupCameraSwitchButton()
    }

    private fun initBottomSheetControls() {
        // init bottom sheet settings

        fragmentCameraBinding.bottomSheetLayout.detectionThresholdValue.text =
            String.format(
                Locale.US, "%.2f", viewModel.currentMinPoseDetectionConfidence
            )
        fragmentCameraBinding.bottomSheetLayout.trackingThresholdValue.text =
            String.format(
                Locale.US, "%.2f", viewModel.currentMinPoseTrackingConfidence
            )
        fragmentCameraBinding.bottomSheetLayout.presenceThresholdValue.text =
            String.format(
                Locale.US, "%.2f", viewModel.currentMinPosePresenceConfidence
            )
        fragmentCameraBinding.bottomSheetLayout.quantOutputSwitch.isChecked =
            viewModel.currentUseQuantOutput
        fragmentCameraBinding.bottomSheetLayout.cacheableInputSwitch.isChecked =
            viewModel.currentCacheableInput

        // When clicked, lower pose detection score threshold floor
        fragmentCameraBinding.bottomSheetLayout.detectionThresholdMinus.setOnClickListener {
            if (poseLandmarkerHelper.minPoseDetectionConfidence >= 0.2) {
                poseLandmarkerHelper.minPoseDetectionConfidence -= 0.1f
                updateControlsUi()
            }
        }

        // When clicked, raise pose detection score threshold floor
        fragmentCameraBinding.bottomSheetLayout.detectionThresholdPlus.setOnClickListener {
            if (poseLandmarkerHelper.minPoseDetectionConfidence <= 0.8) {
                poseLandmarkerHelper.minPoseDetectionConfidence += 0.1f
                updateControlsUi()
            }
        }

        // When clicked, lower pose tracking score threshold floor
        fragmentCameraBinding.bottomSheetLayout.trackingThresholdMinus.setOnClickListener {
            if (poseLandmarkerHelper.minPoseTrackingConfidence >= 0.2) {
                poseLandmarkerHelper.minPoseTrackingConfidence -= 0.1f
                updateControlsUi()
            }
        }

        // When clicked, raise pose tracking score threshold floor
        fragmentCameraBinding.bottomSheetLayout.trackingThresholdPlus.setOnClickListener {
            if (poseLandmarkerHelper.minPoseTrackingConfidence <= 0.8) {
                poseLandmarkerHelper.minPoseTrackingConfidence += 0.1f
                updateControlsUi()
            }
        }

        // When clicked, lower pose presence score threshold floor
        fragmentCameraBinding.bottomSheetLayout.presenceThresholdMinus.setOnClickListener {
            if (poseLandmarkerHelper.minPosePresenceConfidence >= 0.2) {
                poseLandmarkerHelper.minPosePresenceConfidence -= 0.1f
                updateControlsUi()
            }
        }

        // When clicked, raise pose presence score threshold floor
        fragmentCameraBinding.bottomSheetLayout.presenceThresholdPlus.setOnClickListener {
            if (poseLandmarkerHelper.minPosePresenceConfidence <= 0.8) {
                poseLandmarkerHelper.minPosePresenceConfidence += 0.1f
                updateControlsUi()
            }
        }

        fragmentCameraBinding.bottomSheetLayout.quantOutputSwitch.setOnCheckedChangeListener { _, isChecked ->
            viewModel.setUseQuantOutput(isChecked)
            if (this::poseLandmarkerHelper.isInitialized) {
                backgroundExecutor.execute {
                    poseLandmarkerHelper.setPerfOptions(
                        isChecked,
                        viewModel.currentCacheableInput
                    )
                }
            }
        }

        fragmentCameraBinding.bottomSheetLayout.cacheableInputSwitch.setOnCheckedChangeListener { _, isChecked ->
            viewModel.setCacheableInput(isChecked)
            if (this::poseLandmarkerHelper.isInitialized) {
                backgroundExecutor.execute {
                    poseLandmarkerHelper.setPerfOptions(
                        viewModel.currentUseQuantOutput,
                        isChecked
                    )
                }
            }
        }

        // When clicked, change the underlying hardware used for inference.
        // Current options are CPU and GPU
        fragmentCameraBinding.bottomSheetLayout.spinnerDelegate.setSelection(
            viewModel.currentDelegate, false
        )
        fragmentCameraBinding.bottomSheetLayout.spinnerDelegate.onItemSelectedListener =
            object : AdapterView.OnItemSelectedListener {
                override fun onItemSelected(
                    p0: AdapterView<*>?, p1: View?, p2: Int, p3: Long
                ) {
                    try {
                        poseLandmarkerHelper.currentDelegate = p2
                        updateControlsUi()
                    } catch(e: UninitializedPropertyAccessException) {
                        Log.e(TAG, "PoseLandmarkerHelper has not been initialized yet.")
                    }
                }

                override fun onNothingSelected(p0: AdapterView<*>?) {
                    /* no op */
                }
            }

        // When clicked, change the underlying model used for object detection
        fragmentCameraBinding.bottomSheetLayout.spinnerModel.setSelection(
            viewModel.currentModel,
            false
        )
        fragmentCameraBinding.bottomSheetLayout.spinnerModel.onItemSelectedListener =
            object : AdapterView.OnItemSelectedListener {
                override fun onItemSelected(
                    p0: AdapterView<*>?,
                    p1: View?,
                    p2: Int,
                    p3: Long
                ) {
                    viewModel.setModel(p2)
                    poseLandmarkerHelper.currentModel = p2
                    updateControlsUi()
                }

                override fun onNothingSelected(p0: AdapterView<*>?) {
                    /* no op */
                }
            }

        // Initialize Jump Threshold UI
        fragmentCameraBinding.bottomSheetLayout.jumpUpThresholdValue.text =
            String.format(Locale.US, "%.2f", viewModel.currentJumpUpThreshold)
        fragmentCameraBinding.bottomSheetLayout.jumpDownThresholdValue.text =
            String.format(Locale.US, "%.2f", viewModel.currentJumpDownThreshold)

        // Jump Up Threshold Control
        fragmentCameraBinding.bottomSheetLayout.jumpUpThresholdMinus.setOnClickListener {
            if (viewModel.currentJumpUpThreshold >= 0.1f) {
                val newVal = viewModel.currentJumpUpThreshold - 0.05f
                viewModel.setJumpUpThreshold(newVal)
                fragmentCameraBinding.bottomSheetLayout.jumpUpThresholdValue.text =
                    String.format(Locale.US, "%.2f", newVal)
                if (this::poseLandmarkerHelper.isInitialized) {
                    poseLandmarkerHelper.setJumpThresholds(newVal, viewModel.currentJumpDownThreshold)
                }
            }
        }
        fragmentCameraBinding.bottomSheetLayout.jumpUpThresholdPlus.setOnClickListener {
            if (viewModel.currentJumpUpThreshold <= 1.0f) {
                val newVal = viewModel.currentJumpUpThreshold + 0.05f
                viewModel.setJumpUpThreshold(newVal)
                fragmentCameraBinding.bottomSheetLayout.jumpUpThresholdValue.text =
                    String.format(Locale.US, "%.2f", newVal)
                if (this::poseLandmarkerHelper.isInitialized) {
                    poseLandmarkerHelper.setJumpThresholds(newVal, viewModel.currentJumpDownThreshold)
                }
            }
        }

        // Jump Down Threshold Control
        fragmentCameraBinding.bottomSheetLayout.jumpDownThresholdMinus.setOnClickListener {
            if (viewModel.currentJumpDownThreshold >= 0.1f) {
                val newVal = viewModel.currentJumpDownThreshold - 0.05f
                viewModel.setJumpDownThreshold(newVal)
                fragmentCameraBinding.bottomSheetLayout.jumpDownThresholdValue.text =
                    String.format(Locale.US, "%.2f", newVal)
                if (this::poseLandmarkerHelper.isInitialized) {
                    poseLandmarkerHelper.setJumpThresholds(viewModel.currentJumpUpThreshold, newVal)
                }
            }
        }
        fragmentCameraBinding.bottomSheetLayout.jumpDownThresholdPlus.setOnClickListener {
            if (viewModel.currentJumpDownThreshold <= 1.0f) {
                val newVal = viewModel.currentJumpDownThreshold + 0.05f
                viewModel.setJumpDownThreshold(newVal)
                fragmentCameraBinding.bottomSheetLayout.jumpDownThresholdValue.text =
                    String.format(Locale.US, "%.2f", newVal)
                if (this::poseLandmarkerHelper.isInitialized) {
                    poseLandmarkerHelper.setJumpThresholds(viewModel.currentJumpUpThreshold, newVal)
                }
            }
        }
    }

    // Update the values displayed in the bottom sheet. Reset Poselandmarker
    // helper.
    private fun updateControlsUi() {
        if(this::poseLandmarkerHelper.isInitialized) {
            fragmentCameraBinding.bottomSheetLayout.detectionThresholdValue.text =
                String.format(
                    Locale.US,
                    "%.2f",
                    poseLandmarkerHelper.minPoseDetectionConfidence
                )
            fragmentCameraBinding.bottomSheetLayout.trackingThresholdValue.text =
                String.format(
                    Locale.US,
                    "%.2f",
                    poseLandmarkerHelper.minPoseTrackingConfidence
                )
            fragmentCameraBinding.bottomSheetLayout.presenceThresholdValue.text =
                String.format(
                    Locale.US,
                    "%.2f",
                    poseLandmarkerHelper.minPosePresenceConfidence
                )

            // Needs to be cleared instead of reinitialized because the GPU
            // delegate needs to be initialized on the thread using it when applicable
            backgroundExecutor.execute {
                poseLandmarkerHelper.clearPoseLandmarker()
                poseLandmarkerHelper.setupPoseLandmarker()
            }
            fragmentCameraBinding.overlay.clear()
        }
    }

    // Setup camera switch button
    private fun setupCameraSwitchButton() {
        fragmentCameraBinding.cameraSwitchButton.setOnClickListener {
            Log.i(TAG, "Camera switch button clicked")
            showCameraSelectionDialog()
        }
    }

    // Enumerate all available cameras using Camera2 API directly
    // This can detect more cameras than CameraX alone
    @OptIn(ExperimentalCamera2Interop::class)
    private fun enumerateCameras(provider: ProcessCameraProvider) {
        availableCameras.clear()
        
        // 首先使用 Camera2 API 直接枚举所有摄像头
        val cameraManager = requireContext().getSystemService(Context.CAMERA_SERVICE) as CameraManager
        val camera2CameraIds = try {
            cameraManager.cameraIdList.toSet()
        } catch (e: Exception) {
            Log.e(TAG, "Failed to get camera ID list from Camera2 API", e)
            emptySet()
        }
        
        // CameraX 可见的摄像头ID
        val cameraXCameraIds = mutableSetOf<String>()
        
        Log.i(TAG, "Camera2 API reports ${camera2CameraIds.size} cameras: ${camera2CameraIds.joinToString()}")
        Log.i(TAG, "CameraX reports ${provider.availableCameraInfos.size} cameras")

        // 先添加 CameraX 能识别的摄像头
        provider.availableCameraInfos.forEachIndexed { index, cameraInfo ->
            try {
                val camera2Info = Camera2CameraInfo.from(cameraInfo)
                val cameraId = camera2Info.cameraId
                cameraXCameraIds.add(cameraId)
                
                val lensFacing = camera2Info.getCameraCharacteristic(CameraCharacteristics.LENS_FACING)
                val facing = lensFacing ?: CameraCharacteristics.LENS_FACING_EXTERNAL
                val displayName = when (facing) {
                    CameraCharacteristics.LENS_FACING_BACK -> "后置摄像头 ($cameraId)"
                    CameraCharacteristics.LENS_FACING_FRONT -> "前置摄像头 ($cameraId)"
                    CameraCharacteristics.LENS_FACING_EXTERNAL -> "外接摄像头 ($cameraId)"
                    else -> "摄像头 $cameraId"
                }

                val hardwareLevel = camera2Info.getCameraCharacteristic(
                    CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL
                )
                val hardwareLevelStr = getHardwareLevelString(hardwareLevel)

                availableCameras.add(CameraInfo(cameraId, facing, displayName))
                Log.i(TAG, "[CameraX] Found: $displayName (ID: $cameraId, Facing: $facing, HW: $hardwareLevelStr)")
            } catch (e: Exception) {
                Log.w(TAG, "Error enumerating CameraX camera at index $index", e)
            }
        }

        // 检查 Camera2 API 中有但 CameraX 没有的摄像头
        val missingCameraIds = camera2CameraIds - cameraXCameraIds
        if (missingCameraIds.isNotEmpty()) {
            Log.w(TAG, "Found ${missingCameraIds.size} cameras not visible to CameraX: ${missingCameraIds.joinToString()}")
            
            for (cameraId in missingCameraIds) {
                try {
                    val characteristics = cameraManager.getCameraCharacteristics(cameraId)
                    val lensFacing = characteristics.get(CameraCharacteristics.LENS_FACING)
                        ?: CameraCharacteristics.LENS_FACING_EXTERNAL
                    val hardwareLevel = characteristics.get(CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL)
                    val hardwareLevelStr = getHardwareLevelString(hardwareLevel)
                    
                    val displayName = when (lensFacing) {
                        CameraCharacteristics.LENS_FACING_BACK -> "后置摄像头* ($cameraId)"
                        CameraCharacteristics.LENS_FACING_FRONT -> "前置摄像头* ($cameraId)"
                        CameraCharacteristics.LENS_FACING_EXTERNAL -> "外接摄像头* ($cameraId)"
                        else -> "摄像头* $cameraId"
                    }
                    
                    // 标记为 Camera2-only 摄像头（CameraX 不支持）
                    availableCameras.add(CameraInfo(cameraId, lensFacing, "$displayName [仅Camera2]"))
                    Log.w(TAG, "[Camera2 Only] Found: $displayName (ID: $cameraId, Facing: $lensFacing, HW: $hardwareLevelStr)")
                    Log.w(TAG, "  -> This camera may not work with CameraX. Consider using Camera2 API directly.")
                } catch (e: Exception) {
                    Log.e(TAG, "Failed to get characteristics for camera $cameraId", e)
                }
            }
        }
        
        Log.i(TAG, "Total cameras enumerated: ${availableCameras.size} (CameraX: ${cameraXCameraIds.size}, Camera2-only: ${missingCameraIds.size})")
    }
    
    private fun getHardwareLevelString(hardwareLevel: Int?): String {
        return when (hardwareLevel) {
            CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL_LEGACY -> "LEGACY"
            CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL_LIMITED -> "LIMITED"
            CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL_FULL -> "FULL"
            CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL_3 -> "LEVEL_3"
            CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL_EXTERNAL -> "EXTERNAL"
            else -> "UNKNOWN($hardwareLevel)"
        }
    }

    // Show camera selection dialog
    private fun showCameraSelectionDialog() {
        Log.i(TAG, "showCameraSelectionDialog called, available cameras: ${availableCameras.size}")

        if (availableCameras.isEmpty()) {
            Toast.makeText(requireContext(), "没有可用的摄像头", Toast.LENGTH_SHORT).show()
            return
        }

        val cameraNames = availableCameras.map { it.displayName }.toTypedArray()
        val currentIndex = availableCameras.indexOfFirst {
            it.id == selectedCameraId
        }.takeIf { it >= 0 } ?: 0

        Log.i(TAG, "Showing dialog with ${cameraNames.size} cameras, current index: $currentIndex, current ID: $selectedCameraId")

        AlertDialog.Builder(requireContext())
            .setTitle(R.string.select_camera)
            .setSingleChoiceItems(cameraNames, currentIndex) { dialog, which ->
                val selectedCamera = availableCameras[which]
                Log.i(TAG, "Camera selected: ${selectedCamera.displayName}, ID: ${selectedCamera.id}, facing: ${selectedCamera.lensFacing}")

                if (selectedCamera.id != selectedCameraId) {
                    selectedCameraId = selectedCamera.id
                    cameraFacing = selectedCamera.lensFacing
                    Log.i(TAG, "Switching to camera ID: ${selectedCamera.id}, calling bindCameraUseCases()")
                    bindCameraUseCases()
                } else {
                    Log.i(TAG, "Same camera selected, no switch needed")
                }
                dialog.dismiss()
      }
            .setNegativeButton("取消", null)
            .show()
    }

    // Initialize CameraX, and prepare to bind the camera use cases
    private fun setUpCamera() {
        val cameraProviderFuture =
            ProcessCameraProvider.getInstance(requireContext())
        cameraProviderFuture.addListener(
            {
                // CameraProvider
                cameraProvider = cameraProviderFuture.get()

                // Enumerate all available cameras
                cameraProvider?.let { enumerateCameras(it) }

                // Build and bind the camera use cases
                bindCameraUseCases()
            }, ContextCompat.getMainExecutor(requireContext())
        )
    }

    // Declare and bind preview, capture and analysis use cases
    @OptIn(ExperimentalCamera2Interop::class)
    private fun bindCameraUseCases() {

        // CameraProvider
        val cameraProvider = cameraProvider
            ?: throw IllegalStateException("Camera initialization failed.")

        val cameraSelector = selectCamera(cameraProvider)

        // Preview at 720p to match the analyzer resolution
        preview = Preview.Builder().setMaxResolution(targetResolution)
            .setTargetRotation(fragmentCameraBinding.viewFinder.display.rotation)
            .build()

        // ImageAnalysis. Using RGBA 8888 to match how our models work
        val imageAnalysisBuilder = ImageAnalysis.Builder().setMaxResolution(targetResolution)
            .setTargetRotation(fragmentCameraBinding.viewFinder.display.rotation)
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)

        // Set target FPS to 15
        Camera2Interop.Extender(imageAnalysisBuilder).setCaptureRequestOption(
            CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE,
            Range(15, 15)
        )

        imageAnalyzer = imageAnalysisBuilder.build().also {
            it.setAnalyzer(backgroundExecutor) { image ->
                detectPose(image)
            }
        }

        // Must unbind the use-cases before rebinding them
        cameraProvider.unbindAll()

        try {
            // A variable number of use-cases can be passed here -
            // camera provides access to CameraControl & CameraInfo
            camera = cameraProvider.bindToLifecycle(
                this, cameraSelector, preview, imageAnalyzer
            )

            // Attach the viewfinder's surface provider to preview use case
            preview?.setSurfaceProvider(fragmentCameraBinding.viewFinder.surfaceProvider)
        } catch (exc: Exception) {
            Log.e(TAG, "Use case binding failed", exc)
        }
    }

    @OptIn(ExperimentalCamera2Interop::class)
    private fun selectCamera(provider: ProcessCameraProvider): CameraSelector {
        // If a specific camera ID is selected, try to use it
        if (selectedCameraId != null) {
            try {
                val cameraInfo = provider.availableCameraInfos.find { info ->
                    Camera2CameraInfo.from(info).cameraId == selectedCameraId
                }

                if (cameraInfo != null) {
                    val camera2Info = Camera2CameraInfo.from(cameraInfo)
                    val lensFacing = camera2Info.getCameraCharacteristic(CameraCharacteristics.LENS_FACING)
                    
                    // 对于所有摄像头类型，使用 cameraId 过滤器确保选择正确的摄像头
                    val filteredSelector = CameraSelector.Builder()
                        .addCameraFilter { cameraInfoList ->
                            cameraInfoList.filter { info ->
                                Camera2CameraInfo.from(info).cameraId == selectedCameraId
                            }
                        }
                        .build()

                    // 验证选择器是否有效
                    val filteredCameras = provider.availableCameraInfos.filter { info ->
                        Camera2CameraInfo.from(info).cameraId == selectedCameraId
                    }
                    
                    if (filteredCameras.isNotEmpty()) {
                        Log.i(TAG, "Using selected camera ID: $selectedCameraId (lens facing: $lensFacing)")
                        return filteredSelector
                    }
                }
            } catch (e: Exception) {
                Log.w(TAG, "Selected camera ID $selectedCameraId unavailable", e)
            }
        }

        // Fallback: try to use the selected lens facing
        try {
            val selector = CameraSelector.Builder().requireLensFacing(cameraFacing).build()
            if (provider.hasCamera(selector)) {
                Log.i(TAG, "Using camera with lens facing = $cameraFacing")
                return selector
            }
        } catch (e: Exception) {
            Log.w(TAG, "Camera with lens facing $cameraFacing unavailable", e)
        }

        // Fallback: try other cameras in order
        val orderedLensFacing = listOf(
            CameraSelector.LENS_FACING_BACK,
            CameraSelector.LENS_FACING_FRONT,
            CameraSelector.LENS_FACING_EXTERNAL
        ).filter { it != cameraFacing }

        for (lens in orderedLensFacing) {
            try {
                val selector = CameraSelector.Builder().requireLensFacing(lens).build()
                if (provider.hasCamera(selector)) {
                    Log.i(TAG, "Falling back to camera with lens facing = $lens")
                    cameraFacing = lens
                    return selector
                }
            } catch (e: Exception) {
                Log.w(TAG, "Camera with lens facing $lens unavailable on this device", e)
            }
        }

        // Last resort: use first available camera with ID-based selector
        val firstCamera = provider.availableCameraInfos.firstOrNull()
        if (firstCamera != null) {
            try {
                val camera2Info = Camera2CameraInfo.from(firstCamera)
                val firstCameraId = camera2Info.cameraId
                val lensFacing = camera2Info.getCameraCharacteristic(CameraCharacteristics.LENS_FACING)
                
                Log.i(TAG, "Using first available camera ID: $firstCameraId, lens facing = $lensFacing")
                selectedCameraId = firstCameraId
                cameraFacing = lensFacing ?: CameraCharacteristics.LENS_FACING_EXTERNAL
                
                return CameraSelector.Builder()
                    .addCameraFilter { cameraInfoList ->
                        cameraInfoList.filter { info ->
                            Camera2CameraInfo.from(info).cameraId == firstCameraId
                        }
                    }
                    .build()
            } catch (e: Exception) {
                Log.e(TAG, "Failed to create selector for first camera", e)
            }
        }

        return CameraSelector.DEFAULT_BACK_CAMERA
    }

    private fun detectPose(imageProxy: ImageProxy) {
        // Cache resolution/rotation for UI-thread logging
        lastAnalysisResolution = Size(imageProxy.width, imageProxy.height)
        lastAnalysisRotation = imageProxy.imageInfo.rotationDegrees

        if(this::poseLandmarkerHelper.isInitialized) {
            poseLandmarkerHelper.detectLiveStream(
                imageProxy = imageProxy,
                isFrontCamera = cameraFacing == CameraSelector.LENS_FACING_FRONT
            )
        }
    }

    override fun onConfigurationChanged(newConfig: Configuration) {
        super.onConfigurationChanged(newConfig)
        imageAnalyzer?.targetRotation =
            fragmentCameraBinding.viewFinder.display.rotation
    }

    // Update UI after pose have been detected. Extracts original
    // image height/width to scale and place the landmarks properly through
    // OverlayView
    override fun onResults(
        resultBundle: PoseLandmarkerHelper.ResultBundle
    ) {
        activity?.runOnUiThread {
            if (_fragmentCameraBinding != null) {
                fragmentCameraBinding.bottomSheetLayout.inferenceTimeVal.text =
                    String.format("%d ms", resultBundle.inferenceTime)
                fragmentCameraBinding.bottomSheetLayout.algorithmTimeVal.text =
                    String.format(
                        Locale.US,
                        "pre:%d | native:%d | post:%d ms",
                        resultBundle.preprocessTime,
                        resultBundle.nativeTime,
                        resultBundle.postProcessTime
                    )

                // Pass necessary information to OverlayView for drawing on the canvas
                fragmentCameraBinding.overlay.setResults(
                    resultBundle.results.first(),
                    resultBundle.inputImageHeight,
                    resultBundle.inputImageWidth,
                    RunningMode.LIVE_STREAM,
                    poseLandmarkerHelper.minPoseTrackingConfidence
                )


                if (debugLogsRemaining > 0) {
                    val viewFinderSize = Size(
                        fragmentCameraBinding.viewFinder.width,
                        fragmentCameraBinding.viewFinder.height
                    )
                    val overlaySize = Size(
                        fragmentCameraBinding.overlay.width,
                        fragmentCameraBinding.overlay.height
                    )
                    val analysisSize = lastAnalysisResolution
                    Log.d(
                        TAG,
                        "Debug frame ${6 - debugLogsRemaining}: analyzer=${analysisSize?.width}x${analysisSize?.height} " +
                                "rot=${lastAnalysisRotation}°, inputImage=${resultBundle.inputImageWidth}x${resultBundle.inputImageHeight}, " +
                                "viewFinder=${viewFinderSize.width}x${viewFinderSize.height}, overlay=${overlaySize.width}x${overlaySize.height}, " +
                                "targetResolution=${targetResolution.width}x${targetResolution.height}"
                    )
                    debugLogsRemaining--
                }

                // Force a redraw
                fragmentCameraBinding.overlay.invalidate()
            }
        }
    }

    override fun onError(error: String, errorCode: Int) {
        activity?.runOnUiThread {
            Toast.makeText(requireContext(), error, Toast.LENGTH_SHORT).show()
        }
    }
}
