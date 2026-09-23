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
 * Uses backpressure handling to skip frames if inference is still running,
 * ensuring smooth camera preview without lag.
 *
 * @param detector The ObjectDetector to use for detection
 * @param onResults Callback invoked with detection results for each processed frame
 * @param onFps Optional callback for FPS monitoring
 */
internal class EasyMLAnalyzer(
    private val detector: ObjectDetector,
    private val onResults: (List<Detection>, Int, Int) -> Unit,
    private val onFps: ((Float) -> Unit)? = null
) : ImageAnalysis.Analyzer {

    private val isProcessing = AtomicBoolean(false)
    private var lastFpsTime = System.currentTimeMillis()
    private var frameCount = 0

    override fun analyze(imageProxy: ImageProxy) {
        // Skip frame if still processing the previous one (backpressure)
        if (!isProcessing.compareAndSet(false, true)) {
            imageProxy.close()
            return
        }

        try {
            // Convert CameraX frame to Bitmap
            val bitmap = ImageUtils.imageProxyToBitmap(imageProxy)
            val imageWidth = bitmap.width
            val imageHeight = bitmap.height

            // Run detection
            val detections = detector.detect(bitmap)

            // Recycle bitmap
            bitmap.recycle()

            // Report results
            onResults(detections, imageWidth, imageHeight)

            // Calculate FPS
            frameCount++
            val currentTime = System.currentTimeMillis()
            val elapsed = currentTime - lastFpsTime
            if (elapsed >= 1000) {
                val fps = frameCount * 1000f / elapsed
                android.util.Log.d("EasyML", "FPS: %.1f, detections: %d".format(fps, detections.size))
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
