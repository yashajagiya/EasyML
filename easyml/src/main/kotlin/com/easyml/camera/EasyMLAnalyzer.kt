package com.easyml.camera

import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import com.easyml.core.ImageUtils
import com.easyml.detection.Detection
import com.easyml.detection.DetectionSmoother
import com.easyml.detection.InferenceMetrics
import com.easyml.detection.ObjectDetector
import java.util.concurrent.atomic.AtomicBoolean

/**
 * CameraX ImageAnalysis.Analyzer implementation for real-time object detection.
 *
 * Efficiently processes camera frames using the EasyML ObjectDetector.
 * Uses backpressure handling, optional frame rate capping, and temporal box smoothing.
 *
 * @param detector The ObjectDetector to use for detection
 * @param onResults Callback invoked with detection results for each processed frame
 * @param onFps Optional callback for FPS monitoring
 * @param onInferenceTime Optional callback for per-frame inference latency (ms)
 * @param onInferenceMetrics Optional callback for granular microsecond latency breakdown
 * @param targetFps Optional maximum FPS limit (e.g. 30) to preserve battery
 * @param enableSmoothing Whether to apply temporal EMA box smoothing to reduce jitter
 * @param smoothingFactor EMA weight factor (0.0 to 1.0)
 */
internal class EasyMLAnalyzer(
    private val detector: ObjectDetector,
    private val onResults: (List<Detection>, Int, Int) -> Unit,
    private val onFps: ((Float) -> Unit)? = null,
    private val onInferenceTime: ((Long) -> Unit)? = null,
    private val onInferenceMetrics: ((InferenceMetrics) -> Unit)? = null,
    private val targetFps: Int? = null,
    enableSmoothing: Boolean = true,
    smoothingFactor: Float = 0.35f
) : ImageAnalysis.Analyzer {

    private val isProcessing = AtomicBoolean(false)
    private val smoother: DetectionSmoother? = if (enableSmoothing && !detector.isSmoothingEnabled) {
        DetectionSmoother(smoothingFactor)
    } else {
        null
    }
    private var lastFpsTime = System.currentTimeMillis()
    private var lastAnalyzedFrameTime = 0L
    private var frameCount = 0

    // Pre-allocated reusable structures to eliminate per-frame heap churn
    private val reusableResults = ArrayList<Detection>(20)

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
            // Reliable, opaque frame acquisition with correct orientation
            val bitmap = ImageUtils.imageProxyToBitmap(imageProxy)
            val finalWidth = bitmap.width
            val finalHeight = bitmap.height

            // Run detection on upright bitmap with zero-allocation results buffer
            detector.detect(bitmap, 0, reusableResults)
            val detections = smoother?.update(reusableResults) ?: reusableResults
            val metrics = detector.lastMetrics

            bitmap.recycle()

            // Report results
            onResults(detections, finalWidth, finalHeight)
            onInferenceTime?.invoke(metrics.totalMs.toLong())
            onInferenceMetrics?.invoke(metrics)

            // Calculate FPS based on completed frame completion time
            val completionTime = System.currentTimeMillis()
            frameCount++
            val elapsed = completionTime - lastFpsTime
            if (elapsed >= 1000) {
                val fps = frameCount * 1000f / elapsed
                android.util.Log.d("EasyML", "FPS: %.1f (latency: %.1f ms, detections: %d)".format(fps, metrics.totalMs, detections.size))
                onFps?.invoke(fps)
                frameCount = 0
                lastFpsTime = completionTime
            }
        } catch (e: Exception) {
            android.util.Log.e("EasyML", "Error analyzing frame: ${e.message}", e)
        } finally {
            imageProxy.close()
            isProcessing.set(false)
        }
    }
}
