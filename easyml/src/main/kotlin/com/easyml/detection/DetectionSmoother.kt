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
    private val smoothedList = ArrayList<Detection>(30)

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

        smoothedList.clear()
        val n = newDetections.size
        val prevSize = previousDetections.size

        for (i in 0 until n) {
            val newDet = newDetections[i]
            var matchedPrev: Detection? = null

            for (j in 0 until prevSize) {
                val prev = previousDetections[j]
                if (prev.labelIndex == newDet.labelIndex &&
                    calculateIoU(prev.boundingBox, newDet.boundingBox) > iouMatchThreshold
                ) {
                    matchedPrev = prev
                    break
                }
            }

            if (matchedPrev != null) {
                val smoothedBox = RectF(
                    newDet.boundingBox.left * alpha + matchedPrev.boundingBox.left * invAlpha,
                    newDet.boundingBox.top * alpha + matchedPrev.boundingBox.top * invAlpha,
                    newDet.boundingBox.right * alpha + matchedPrev.boundingBox.right * invAlpha,
                    newDet.boundingBox.bottom * alpha + matchedPrev.boundingBox.bottom * invAlpha
                )
                smoothedList.add(newDet.copy(boundingBox = smoothedBox))
            } else {
                smoothedList.add(newDet)
            }
        }

        previousDetections = ArrayList(smoothedList)
        return smoothedList
    }

    /**
     * Reset the smoother history (e.g., when the camera pauses, switches lenses, or changes scenes).
     */
    fun reset() {
        previousDetections = emptyList()
        smoothedList.clear()
    }

    private fun calculateIoU(a: RectF, b: RectF): Float {
        val intersectLeft = max(a.left, b.left)
        val intersectTop = max(a.top, b.top)
        val intersectRight = min(a.right, b.right)
        val intersectBottom = min(a.bottom, b.bottom)

        val intersectWidth = intersectRight - intersectLeft
        if (intersectWidth <= 0f) return 0f
        val intersectHeight = intersectBottom - intersectTop
        if (intersectHeight <= 0f) return 0f

        val intersectArea = intersectWidth * intersectHeight
        val aArea = (a.right - a.left) * (a.bottom - a.top)
        val bArea = (b.right - b.left) * (b.bottom - b.top)
        val unionArea = aArea + bArea - intersectArea

        return if (unionArea > 0f) intersectArea / unionArea else 0f
    }
}
