package com.easyml.detection

import android.graphics.RectF
import kotlin.math.max
import kotlin.math.min

/**
 * Standalone temporal box smoother for real-time video streams (e.g. CameraX).
 *
 * Uses Exponential Moving Average (EMA) interpolation between matching detections in consecutive frames
 * to eliminate high-frequency bounding box jitter while maintaining responsive tracking of moving objects.
 *
 * Decoupled from [ObjectDetector] so detection inference remains pure and stateless.
 *
 * @param smoothingFactor EMA weight applied to new frames: 0.0 (maximum smoothness) to 1.0 (instantaneous)
 * @param iouMatchThreshold Minimum IoU required to associate a detection across consecutive frames (default: 0.35)
 */
class DetectionSmoother(
    val smoothingFactor: Float = 0.3f,
    val iouMatchThreshold: Float = 0.35f
) {

    private var previousDetections: List<Detection> = emptyList()

    /**
     * Smooth detection bounding boxes against the previous frame.
     *
     * @param newDetections The raw detections from the current frame
     * @return List of smoothed detections
     */
    fun update(newDetections: List<Detection>): List<Detection> {
        if (smoothingFactor >= 1f || previousDetections.isEmpty() || newDetections.isEmpty()) {
            previousDetections = newDetections
            return newDetections
        }

        val alpha = smoothingFactor.coerceIn(0.01f, 1f)
        val invAlpha = 1f - alpha

        val smoothed = newDetections.map { newDet ->
            val prevDet = previousDetections.firstOrNull { prev ->
                prev.labelIndex == newDet.labelIndex &&
                    calculateIoU(prev.boundingBox, newDet.boundingBox) > iouMatchThreshold
            }

            if (prevDet != null) {
                val smoothedBox = RectF(
                    newDet.boundingBox.left * alpha + prevDet.boundingBox.left * invAlpha,
                    newDet.boundingBox.top * alpha + prevDet.boundingBox.top * invAlpha,
                    newDet.boundingBox.right * alpha + prevDet.boundingBox.right * invAlpha,
                    newDet.boundingBox.bottom * alpha + prevDet.boundingBox.bottom * invAlpha
                )
                newDet.copy(boundingBox = smoothedBox)
            } else {
                newDet
            }
        }

        previousDetections = smoothed
        return smoothed
    }

    /**
     * Reset the smoother history (e.g., when the camera pauses, switches lenses, or changes scenes).
     */
    fun reset() {
        previousDetections = emptyList()
    }

    private fun calculateIoU(a: RectF, b: RectF): Float {
        val intersectLeft = max(a.left, b.left)
        val intersectTop = max(a.top, b.top)
        val intersectRight = min(a.right, b.right)
        val intersectBottom = min(a.bottom, b.bottom)

        val intersectWidth = max(0f, intersectRight - intersectLeft)
        val intersectHeight = max(0f, intersectBottom - intersectTop)
        val intersectArea = intersectWidth * intersectHeight
        if (intersectArea <= 0f) return 0f

        val aArea = (a.right - a.left) * (a.bottom - a.top)
        val bArea = (b.right - b.left) * (b.bottom - b.top)
        val unionArea = aArea + bArea - intersectArea

        return if (unionArea > 0f) intersectArea / unionArea else 0f
    }
}
