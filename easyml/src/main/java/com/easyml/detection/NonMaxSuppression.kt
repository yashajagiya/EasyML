package com.easyml.detection

import kotlin.math.max
import kotlin.math.min

/**
 * High-performance Non-Maximum Suppression (NMS) filter.
 *
 * Supports both:
 * - **Class-Aware NMS (default)**: Bounding boxes belonging to different classes (e.g. Person and Bicycle)
 *   do not suppress each other, correctly preserving overlapping multi-class detections.
 * - **Class-Agnostic NMS**: Bounding boxes suppress each other purely based on IoU overlap regardless of class.
 *
 * @param iouThreshold Intersection-over-Union threshold above which overlapping boxes are suppressed (default: 0.45)
 * @param maxResults Maximum number of final detections to retain (default: 20)
 * @param classAgnostic When false (default), only suppresses boxes sharing the same class index
 */
class NonMaxSuppression(
    val iouThreshold: Float = 0.45f,
    val maxResults: Int = 20,
    val classAgnostic: Boolean = false
) {

    /**
     * Run Non-Maximum Suppression on candidates and populate the output list without extra heap allocations.
     */
    fun process(
        candidates: List<DetectionCandidate>,
        outKept: MutableList<DetectionCandidate>
    ) {
        outKept.clear()
        if (candidates.isEmpty()) return

        // Sort descending by confidence, capping at top 300 to bound worst-case O(N^2) comparison
        val count = min(candidates.size, 300)
        val sortedIndices = Array(count) { it }
        // Sort indices based on candidate confidence
        sortedIndices.sortWith { a, b ->
            candidates[b].confidence.compareTo(candidates[a].confidence)
        }

        val suppressed = BooleanArray(count)

        for (i in 0 until count) {
            val idxI = sortedIndices[i]
            if (suppressed[i]) continue

            val current = candidates[idxI]
            outKept.add(current)
            if (outKept.size >= maxResults) break

            for (j in i + 1 until count) {
                if (suppressed[j]) continue
                val idxJ = sortedIndices[j]
                val other = candidates[idxJ]

                // Class-aware check: only suppress if same class or classAgnostic is enabled
                val shouldCompare = classAgnostic || current.labelIndex == other.labelIndex
                if (shouldCompare) {
                    if (calculateIoU(current, other) > iouThreshold) {
                        suppressed[j] = true
                    }
                }
            }
        }
    }

    /**
     * Compute Intersection-over-Union between two detection candidate boxes.
     */
    fun calculateIoU(a: DetectionCandidate, b: DetectionCandidate): Float {
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
