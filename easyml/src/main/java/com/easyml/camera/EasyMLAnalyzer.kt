package com.easyml.camera

import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import com.easyml.core.ImageUtils
import com.easyml.detection.Detection
import com.easyml.detection.ObjectDetector
import java.util.concurrent.atomic.AtomicBoolean

/**
 * CameraX ImageAnalysis.Analyzer implementation for real-time object detection.
 *
 * Efficiently processes camera frames using the EasyML ObjectDetector.
 * Uses backpressure handling and optional frame rate capping to skip frames,
 * ensuring smooth camera preview without lag.
 *
 * @param detector The ObjectDetector to use for detection
 * @param onResults Callback invoked with detection results for each processed frame
 * @param onFps Optional callback for FPS monitoring
 * @param onInferenceTime Optional callback for per-frame inference latency (ms)
 * @param targetFps Optional maximum FPS limit (e.g. 30) to preserve battery
 */
internal class EasyMLAnalyzer(
    private val detector: ObjectDetector,
    private val onResults: (List<Detection>, Int, Int) -> Unit,
    private val onFps: ((Float) -> Unit)? = null,
    private val onInferenceTime: ((Long) -> Unit)? = null,
    private val targetFps: Int? = null
) : ImageAnalysis.Analyzer {

    private val isProcessing = AtomicBoolean(false)
    private var lastFpsTime = System.currentTimeMillis()
    private var lastAnalyzedFrameTime = 0L
    private var frameCount = 0

    override fun analyze(imageProxy: ImageProxy) {
        val currentTime = System.currentTimeMillis()

        // Frame rate throttling if targetFps is configured
        if (targetFps != null && targetFps > 0) {
            val minInterval = 1000L / targetFps
            if (currentTime - lastAnalyzedFrameTime < minInterval) {
                imageProxy.close()
                return
            }
        }

        // Skip frame if still processing the previous one (backpressure)
        if (!isProcessing.compareAndSet(false, true)) {
            imageProxy.close()
            return
        }

        lastAnalyzedFrameTime = currentTime

        try {
            // Convert CameraX frame to Bitmap
            val bitmap = ImageUtils.imageProxyToBitmap(imageProxy)
            val imageWidth = bitmap.width
            val imageHeight = bitmap.height

            // Run detection
            val detections = detector.detect(bitmap)
            val inferenceTime = detector.lastInferenceTimeMs

            // Recycle bitmap
            bitmap.recycle()

            // Report results
            onResults(detections, imageWidth, imageHeight)
            onInferenceTime?.invoke(inferenceTime)

            // Calculate FPS
            frameCount++
            val elapsed = currentTime - lastFpsTime
            if (elapsed >= 1000) {
                val fps = frameCount * 1000f / elapsed
                android.util.Log.d("EasyML", "FPS: %.1f (inference: %d ms, detections: %d)".format(fps, inferenceTime, detections.size))
                onFps?.invoke(fps)
                frameCount = 0
                lastFpsTime = currentTime
            }
        } catch (e: Exception) {
            android.util.Log.e("EasyML", "Error analyzing frame: ${e.message}", e)
        } finally {
            imageProxy.close()
            isProcessing.set(false)
        }
    }
}
