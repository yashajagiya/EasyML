package com.easyml.camera

import android.Manifest
import android.util.Log
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
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
 * - CameraX Preview + ImageAnalysis binding
 * - Real-time inference via [ObjectDetector]
 * - Optional built-in bounding box overlay
 * - FPS counter
 *
 * Usage:
 * ```kotlin
 * EasyMLCameraView(
 *     detector = myDetector,
 *     onResults = { detections -> /* your custom handling */ },
 *     showOverlay = true,   // built-in bounding boxes
 *     showFps = true        // FPS counter
 * )
 * ```
 *
 * @param detector The [ObjectDetector] instance
 * @param modifier Compose modifier
 * @param onResults Callback with detection results per frame (gives you freedom to do anything)
 * @param showOverlay If true, draws built-in bounding boxes. Default: true
 * @param showFps If true, shows FPS counter. Default: true
 * @param lensFacing Camera to use. Default: BACK
 * @param overlayColor Color for bounding boxes
 */
@OptIn(ExperimentalPermissionsApi::class)
@Suppress("UnstableCollections")
@Composable
fun EasyMLCameraView(
    detector: ObjectDetector,
    modifier: Modifier = Modifier,
    showOverlay: Boolean = true,
    showFps: Boolean = true,
    lensFacing: Int = CameraSelector.LENS_FACING_BACK,
    overlayColor: Color = Color(0xFF00E676),
    onResults: ((List<Detection>) -> Unit)? = null
) {
    val lifecycleOwner = LocalLifecycleOwner.current
    val cameraPermissionState = rememberPermissionState(Manifest.permission.CAMERA)

    // State
    var detections by remember { mutableStateOf<List<Detection>>(emptyList()) }
    var imageWidth by remember { mutableIntStateOf(1) }
    var imageHeight by remember { mutableIntStateOf(1) }
    var fps by remember { mutableFloatStateOf(0f) }

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

                    val imageAnalysis = ImageAnalysis.Builder()
                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                        .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                        .build()

                    val analyzer = EasyMLAnalyzer(
                        detector = detector,
                        onResults = { results, w, h ->
                            detections = results
                            imageWidth = w
                            imageHeight = h
                            onResults?.invoke(results)
                        },
                        onFps = { fps = it }
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
                isMirrored = lensFacing == CameraSelector.LENS_FACING_FRONT
            )
        }

        // FPS counter
        if (showFps) {
            Text(
                text = "%.1f FPS".format(fps),
                color = Color.White,
                fontSize = 14.sp,
                modifier = Modifier
                    .align(Alignment.TopStart)
                    .padding(12.dp)
                    .background(Color(0xAA000000), RoundedCornerShape(4.dp))
                    .padding(horizontal = 8.dp, vertical = 4.dp)
            )
        }
    }
}
}
