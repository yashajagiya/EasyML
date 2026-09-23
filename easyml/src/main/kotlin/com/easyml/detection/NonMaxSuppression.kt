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

    // Pre-allocated primitive scratch arrays to eliminate heap allocations during NMS
    private val indexPool = IntArray(300)
    private val suppressed = BooleanArray(300)
    private val boxAreas = FloatArray(300)

    /**
     * Run Non-Maximum Suppression on candidates and populate the output list without extra heap allocations.
     *
     * @param candidates Detection candidates pool
     * @param outKept Destination list for kept detections
     */
    fun process(
        candidates: List<DetectionCandidate>,
        outKept: MutableList<DetectionCandidate>
    ) {
        process(candidates, candidates.size, outKept)
    }

    /**
     * Run Non-Maximum Suppression on candidates and populate the output list without extra heap allocations.
     *
     * @param candidates Detection candidates pool
     * @param count Number of active candidates in the pool
     * @param outKept Destination list for kept detections
     */
    fun process(
        candidates: List<DetectionCandidate>,
        count: Int,
        outKept: MutableList<DetectionCandidate>
    ) {
        outKept.clear()
        if (count <= 0 || candidates.isEmpty()) return

        val n = minOf(count, candidates.size, 300)
        for (i in 0 until n) {
            indexPool[i] = i
            suppressed[i] = false
            val c = candidates[i]
            val w = c.right - c.left
            val h = c.bottom - c.top
            boxAreas[i] = if (w > 0f && h > 0f) w * h else 0f
        }

        // In-place primitive quicksort on IntArray (zero boxing, zero lambda allocations)
        sortIndices(indexPool, 0, n - 1, candidates)

        for (i in 0 until n) {
            if (suppressed[i]) continue

            val idxI = indexPool[i]
            val current = candidates[idxI]
            val currentArea = boxAreas[idxI]
            outKept.add(current)
            if (outKept.size >= maxResults) break

            for (j in i + 1 until n) {
                if (suppressed[j]) continue
                val idxJ = indexPool[j]
                val other = candidates[idxJ]

                // Class-aware check: only suppress if same class or classAgnostic is enabled
                val shouldCompare = classAgnostic || current.labelIndex == other.labelIndex
                if (shouldCompare) {
                    val intersectLeft = max(current.left, other.left)
                    val intersectTop = max(current.top, other.top)
                    val intersectRight = min(current.right, other.right)
                    val intersectBottom = min(current.bottom, other.bottom)

                    val intersectWidth = intersectRight - intersectLeft
                    if (intersectWidth <= 0f) continue

                    val intersectHeight = intersectBottom - intersectTop
                    if (intersectHeight <= 0f) continue

                    val intersectArea = intersectWidth * intersectHeight
                    val otherArea = boxAreas[idxJ]
                    val unionArea = currentArea + otherArea - intersectArea

                    if (unionArea > 0f && (intersectArea / unionArea) > iouThreshold) {
                        suppressed[j] = true
                    }
                }
            }
        }
    }

    /**
     * In-place quicksort on primitive int indices sorting candidates by descending confidence.
     */
    private fun sortIndices(indices: IntArray, low: Int, high: Int, candidates: List<DetectionCandidate>) {
        if (low >= high) return
        val pivotConf = candidates[indices[(low + high) ushr 1]].confidence
        var i = low
        var j = high
        while (i <= j) {
            while (candidates[indices[i]].confidence > pivotConf) i++
            while (candidates[indices[j]].confidence < pivotConf) j--
            if (i <= j) {
                val temp = indices[i]
                indices[i] = indices[j]
                indices[j] = temp
                i++
                j--
            }
        }
        if (low < j) sortIndices(indices, low, j, candidates)
        if (i < high) sortIndices(indices, i, high, candidates)
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
