package com.easyml.camera

import android.Manifest
import android.util.Log
import android.util.Size
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableLongStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.lifecycle.compose.LocalLifecycleOwner
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import com.easyml.detection.Detection
import com.easyml.detection.InferenceMetrics
import com.easyml.detection.ObjectDetector
import com.easyml.detection.toDetectionList
import com.google.accompanist.permissions.ExperimentalPermissionsApi
import com.google.accompanist.permissions.isGranted
import com.google.accompanist.permissions.rememberPermissionState
import com.google.accompanist.permissions.shouldShowRationale
import java.util.concurrent.Executors

/**
 * All-in-one Compose composable for camera preview with real-time object detection.
 *
 * Handles:
 * - Camera permission request
 * - CameraX Preview + ImageAnalysis binding with hardware ISP downscaling to detector input size
 * - Real-time inference via [ObjectDetector] with zero-allocation buffers
 * - Built-in bounding box overlay with configurable styling (rounded corners, labels, confidence)
 * - Temporal box smoothing with Exponential Moving Average (EMA)
 * - Real-time FPS and latency counter
 *
 * Usage:
 * ```kotlin
 * EasyMLCameraView(
 *     detector = myDetector,
 *     onResults = { detections -> /* your custom handling */ },
 *     showOverlay = true,
 *     showFps = true,
 *     showInferenceTime = true,
 *     enableSmoothing = true,
 *     targetFps = 30
 * )
 * ```
 *
 * @param detector The [ObjectDetector] instance
 * @param modifier Compose modifier
 * @param onResults Callback with detection results per frame
 * @param onInferenceTime Optional callback returning per-frame inference duration in milliseconds
 * @param onInferenceMetrics Optional callback returning granular microsecond pipeline profiling metrics
 * @param showOverlay If true, draws built-in bounding boxes. Default: true
 * @param showFps If true, shows FPS badge. Default: true
 * @param showInferenceTime If true, shows inference latency in the FPS badge. Default: true
 * @param showLabels Whether to show class labels on bounding boxes. Default: true
 * @param showConfidence Whether to show confidence % on bounding boxes. Default: true
 * @param targetFps Optional maximum FPS limit (e.g. 30) to preserve battery. Default: null (unlimited)
 * @param enableSmoothing Whether to apply temporal box smoothing. Default: true
 * @param smoothingFactor Smoothing factor (0.0 to 1.0). Default: 0.35
 * @param lensFacing Camera to use. Default: BACK
 * @param overlayColor Color for bounding boxes. Default: #00E676 (vibrant green)
 * @param strokeWidth Bounding box outline width in pixels. Default: 4f
 * @param cornerRadius Rounded corner radius for bounding boxes and labels. Default: 8f
 * @param labelSize Font size for detection labels in sp. Default: 14
 */
@OptIn(ExperimentalPermissionsApi::class)
@Suppress("UnstableCollections")
@Composable
fun EasyMLCameraView(
    detector: ObjectDetector,
    modifier: Modifier = Modifier,
    showOverlay: Boolean = true,
    showFps: Boolean = true,
    showInferenceTime: Boolean = true,
    showLabels: Boolean = true,
    showConfidence: Boolean = true,
    targetFps: Int? = null,
    enableSmoothing: Boolean = true,
    smoothingFactor: Float = 0.35f,
    lensFacing: Int = CameraSelector.LENS_FACING_BACK,
    overlayColor: Color = Color(0xFF00E676),
    strokeWidth: Float = 4f,
    cornerRadius: Float = 8f,
    labelSize: Int = 14,
    onResults: ((List<Detection>) -> Unit)? = null,
    onInferenceTime: ((Long) -> Unit)? = null,
    onInferenceMetrics: ((InferenceMetrics) -> Unit)? = null
) {
    val lifecycleOwner = LocalLifecycleOwner.current
    val cameraPermissionState = rememberPermissionState(Manifest.permission.CAMERA)

    // State
    var detections by remember { mutableStateOf<List<Detection>>(emptyList()) }
    var imageWidth by remember { mutableIntStateOf(1) }
    var imageHeight by remember { mutableIntStateOf(1) }
    var fps by remember { mutableFloatStateOf(0f) }
    var inferenceTimeMs by remember { mutableLongStateOf(0L) }

    LaunchedEffect(Unit) {
        if (!cameraPermissionState.status.isGranted) {
            cameraPermissionState.launchPermissionRequest()
        }
    }

    Box(modifier = modifier.fillMaxSize()) {
        if (!cameraPermissionState.status.isGranted) {
            Box(
                modifier = Modifier
                    .fillMaxSize()
                    .background(Color.Black),
                contentAlignment = Alignment.Center
            ) {
                Text(
                    text = if (cameraPermissionState.status.shouldShowRationale) {
                        "Camera permission is needed for detection.\nPlease grant it in Settings."
                    } else {
                        "Requesting camera permission..."
                    },
                    color = Color.White,
                    fontSize = 16.sp
                )
            }
        } else {
            val cameraExecutor = remember { Executors.newSingleThreadExecutor() }

            DisposableEffect(Unit) {
                onDispose {
                    cameraExecutor.shutdown()
                }
            }

            // Camera preview
            AndroidView(
                modifier = Modifier.fillMaxSize(),
                factory = { ctx ->
                    val previewView = PreviewView(ctx).apply {
                        scaleType = PreviewView.ScaleType.FILL_CENTER
                    }

                    val cameraProviderFuture = ProcessCameraProvider.getInstance(ctx)
                    cameraProviderFuture.addListener({
                        val cameraProvider = cameraProviderFuture.get()

                        val preview = Preview.Builder().build().also {
                            it.surfaceProvider = previewView.surfaceProvider
                        }

                        // Hardware-assisted downscaling: ask CameraX to deliver frames matching detector input size
                        val targetResolution = Size(detector.getInputSize(), detector.getInputSize())
                        val resolutionSelector = ResolutionSelector.Builder()
                            .setResolutionStrategy(
                                ResolutionStrategy(
                                    targetResolution,
                                    ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER
                                )
                            )
                            .build()
                        val imageAnalysis = ImageAnalysis.Builder()
                            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                            .setResolutionSelector(resolutionSelector)
                            .build()

                        val analyzer = EasyMLAnalyzer(
                            detector = detector,
                            targetFps = targetFps,
                            enableSmoothing = enableSmoothing,
                            smoothingFactor = smoothingFactor,
                            onResults = { results, w, h ->
                                detections = results
                                imageWidth = w
                                imageHeight = h
                                onResults?.invoke(results)
                            },
                            onFps = { fps = it },
                            onInferenceTime = { time ->
                                inferenceTimeMs = time
                                onInferenceTime?.invoke(time)
                            },
                            onInferenceMetrics = onInferenceMetrics
                        )

                        imageAnalysis.setAnalyzer(cameraExecutor, analyzer)

                        val cameraSelector = CameraSelector.Builder()
                            .requireLensFacing(lensFacing)
                            .build()

                        try {
                            cameraProvider.unbindAll()
                            cameraProvider.bindToLifecycle(
                                lifecycleOwner,
                                cameraSelector,
                                preview,
                                imageAnalysis
                            )
                        } catch (e: Exception) {
                            Log.e("EasyML", "Camera binding failed: ${e.message}", e)
                        }
                    }, ContextCompat.getMainExecutor(ctx))

                    previewView
                }
            )

            // Detection overlay
            if (showOverlay && detections.isNotEmpty()) {
                DetectionOverlay(
                    detections = detections.toDetectionList(),
                    imageWidth = imageWidth,
                    imageHeight = imageHeight,
                    modifier = Modifier.fillMaxSize(),
                    boxColor = overlayColor,
                    strokeWidth = strokeWidth,
                    cornerRadius = cornerRadius,
                    labelSize = labelSize,
                    showLabels = showLabels,
                    showConfidence = showConfidence,
                    isMirrored = lensFacing == CameraSelector.LENS_FACING_FRONT
                )
            }

            // Performance badge (FPS & Inference Latency)
            if (showFps || showInferenceTime) {
                val badgeText = buildString {
                    if (showFps) append("%.1f FPS".format(fps))
                    if (showFps && showInferenceTime && inferenceTimeMs > 0) append("  •  ")
                    if (showInferenceTime && inferenceTimeMs > 0) append("${inferenceTimeMs}ms")
                }
                if (badgeText.isNotEmpty()) {
                    Text(
                        text = badgeText,
                        color = Color.White,
                        fontSize = 14.sp,
                        modifier = Modifier
                            .align(Alignment.TopStart)
                            .padding(12.dp)
                            .background(Color(0xCC000000), RoundedCornerShape(6.dp))
                            .padding(horizontal = 10.dp, vertical = 6.dp)
                    )
                }
            }
        }
    }
}
