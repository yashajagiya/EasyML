package com.easyml.detection

import android.graphics.RectF
import kotlin.math.max
import kotlin.math.min

/**
 * Post-processor for YOLO model outputs.
 *
 * Handles:
 * - Auto-detection of output tensor format (YOLOv5 vs YOLOv8/v11/v26)
 * - Bounding box decoding from [cx, cy, w, h] to [left, top, right, bottom]
 * - Confidence thresholding
 * - Non-Maximum Suppression (NMS) to remove overlapping boxes
 * - Coordinate scaling back to original image dimensions
 */
internal class YoloPostProcessor(
    private val confidenceThreshold: Float,
    private val iouThreshold: Float,
    private val maxResults: Int,
    private val labels: List<String>
) {

    /**
     * Process the raw model output tensor into a list of [Detection] results.
     *
     * @param output Raw float output from the model
     * @param outputShape Shape of the output tensor (e.g., [1, 84, 8400] or [1, 8400, 84])
     * @param originalWidth Original image width (for coordinate mapping)
     * @param originalHeight Original image height (for coordinate mapping)
     * @param letterboxScale Scale factor from letterbox padding (null if no letterbox)
     * @param letterboxPadX X padding from letterbox (0 if no letterbox)
     * @param letterboxPadY Y padding from letterbox (0 if no letterbox)
     * @param modelWidth Model input width in pixels (e.g. 640)
     * @param modelHeight Model input height in pixels (e.g. 640)
     */
    fun process(
        output: FloatArray,
        outputShape: IntArray,
        originalWidth: Int,
        originalHeight: Int,
        letterboxScale: Float = 1f,
        letterboxPadX: Int = 0,
        letterboxPadY: Int = 0,
        modelWidth: Int = 640,
        modelHeight: Int = 640
    ): List<Detection> {
        // Determine output format and extract detections
        val rawDetections = when (// YOLOv8/v11/v26 format: [1, 4+num_classes, num_detections]
            // (num_classes typically < num_detections)
            outputShape.size) {
            3 if outputShape[1] < outputShape[2] -> {
                processTransposed(output, outputShape, modelWidth, modelHeight)
            }
            // YOLOv5 format: [1, num_detections, 4+1+num_classes] or [1, num_detections, 4+num_classes]
            3 if true -> {
                processStandard(output, outputShape, modelWidth, modelHeight)
            }
            // 2D output: [num_detections, 4+num_classes]
            2 -> {
                processFlat(output, outputShape, modelWidth, modelHeight)
            }

            else -> {
                android.util.Log.w("EasyML", "Unknown output shape: ${outputShape.contentToString()}, attempting standard parse")
                processStandard(output, outputShape, modelWidth, modelHeight)
            }
        }

        // Apply NMS
        val nmsResults = nonMaxSuppression(rawDetections)

        // Map coordinates back to original image space and apply labels
        return nmsResults.take(maxResults).map { det ->
            // Remove letterbox padding and scale to original image
            val left = ((det.boundingBox.left - letterboxPadX) / letterboxScale)
                .coerceIn(0f, originalWidth.toFloat())
            val top = ((det.boundingBox.top - letterboxPadY) / letterboxScale)
                .coerceIn(0f, originalHeight.toFloat())
            val right = ((det.boundingBox.right - letterboxPadX) / letterboxScale)
                .coerceIn(0f, originalWidth.toFloat())
            val bottom = ((det.boundingBox.bottom - letterboxPadY) / letterboxScale)
                .coerceIn(0f, originalHeight.toFloat())

            Detection(
                boundingBox = RectF(left, top, right, bottom),
                label = if (det.labelIndex < labels.size) labels[det.labelIndex] else "class_${det.labelIndex}",
                labelIndex = det.labelIndex,
                confidence = det.confidence
            )
        }
    }

    /**
     * YOLOv8/v11/v26 format: output shape [1, 4+C, N]
     * Where C = num_classes, N = num_candidate_detections
     * Each column is one detection: rows 0-3 are [cx, cy, w, h], rows 4+ are class scores
     */
    private fun processTransposed(output: FloatArray, shape: IntArray, modelWidth: Int, modelHeight: Int): List<Detection> {
        val numAttributes = shape[1]  // 4 + num_classes
        val numDetections = shape[2]  // e.g., 8400
        val numClasses = numAttributes - 4

        // Sequential memory access pass (eliminates 670k cache misses)
        val maxScores = FloatArray(numDetections)
        val maxClassIndices = IntArray(numDetections)

        for (c in 0 until numClasses) {
            val rowOffset = (4 + c) * numDetections
            for (i in 0 until numDetections) {
                val score = output[rowOffset + i]
                if (score > maxScores[i]) {
                    maxScores[i] = score
                    maxClassIndices[i] = c
                }
            }
        }

        // Only compute bounding boxes for detections that exceed confidence threshold
        val detections = mutableListOf<Detection>()
        val row0 = 0
        val row2 = 2 * numDetections
        val row3 = 3 * numDetections

        for (i in 0 until numDetections) {
            val score = maxScores[i]
            if (score < confidenceThreshold) continue

            var cx = output[row0 + i]
            var cy = output[numDetections + i]
            var w = output[row2 + i]
            var h = output[row3 + i]

            // If coordinates are normalized in [0, 1], scale up to model pixel dimensions
            if (cx <= 1.5f && cy <= 1.5f && w <= 1.5f && h <= 1.5f) {
                cx *= modelWidth
                cy *= modelHeight
                w *= modelWidth
                h *= modelHeight
            }

            val halfW = w * 0.5f
            val halfH = h * 0.5f

            detections.add(
                Detection(
                    boundingBox = RectF(cx - halfW, cy - halfH, cx + halfW, cy + halfH),
                    label = "",
                    labelIndex = maxClassIndices[i],
                    confidence = score
                )
            )
        }

        return detections
    }

    /**
     * YOLOv5 format: output shape [1, N, 4+1+C] or [1, N, 4+C]
     * Each row is one detection: [cx, cy, w, h, obj_conf, class_scores...]
     * or [cx, cy, w, h, class_scores...] (no separate objectness)
     */
    private fun processStandard(output: FloatArray, shape: IntArray, modelWidth: Int, modelHeight: Int): List<Detection> {
        val numDetections = shape[1]
        val numAttributes = shape[2]

        // Determine if there's a separate objectness score
        // YOLOv5 has [cx, cy, w, h, obj_conf, class0, class1, ...]
        // Some models have [cx, cy, w, h, class0, class1, ...]
        val hasObjectness = numAttributes > labels.size + 4 || numAttributes == labels.size + 5
        val classOffset = if (hasObjectness) 5 else 4
        val numClasses = numAttributes - classOffset

        val detections = mutableListOf<Detection>()

        for (i in 0 until numDetections) {
            val base = i * numAttributes

            val objConf = if (hasObjectness) output[base + 4] else 1f

            // Find best class
            var maxClassScore = 0f
            var maxClassIndex = 0
            for (c in 0 until numClasses) {
                val score = output[base + classOffset + c] * objConf
                if (score > maxClassScore) {
                    maxClassScore = score
                    maxClassIndex = c
                }
            }

            if (maxClassScore < confidenceThreshold) continue

            var cx = output[base]
            var cy = output[base + 1]
            var w = output[base + 2]
            var h = output[base + 3]

            // If coordinates are normalized in [0, 1], scale up to model pixel dimensions
            if (cx <= 1.5f && cy <= 1.5f && w <= 1.5f && h <= 1.5f) {
                cx *= modelWidth
                cy *= modelHeight
                w *= modelWidth
                h *= modelHeight
            }

            detections.add(
                Detection(
                    boundingBox = RectF(cx - w / 2f, cy - h / 2f, cx + w / 2f, cy + h / 2f),
                    label = "",
                    labelIndex = maxClassIndex,
                    confidence = maxClassScore
                )
            )
        }

        return detections
    }

    /**
     * Flat 2D format: output shape [N, 4+C]
     */
    private fun processFlat(output: FloatArray, shape: IntArray, modelWidth: Int, modelHeight: Int): List<Detection> {
        // Treat as [1, N, attrs] with batch=1
        return processStandard(output, intArrayOf(1, shape[0], shape[1]), modelWidth, modelHeight)
    }

    /**
     * Non-Maximum Suppression (NMS).
     * Removes overlapping detections, keeping only the highest-confidence ones.
     *
     * Optimized: sorts once, uses early termination.
     */
    private fun nonMaxSuppression(detections: List<Detection>): List<Detection> {
        if (detections.isEmpty()) return emptyList()

        // Pre-filter to at most 100 highest-confidence candidates to avoid O(N^2) load
        val sorted = if (detections.size > 100) {
            detections.sortedByDescending { it.confidence }.take(100)
        } else {
            detections.sortedByDescending { it.confidence }
        }
        val kept = mutableListOf<Detection>()
        val suppressed = BooleanArray(sorted.size)

        for (i in sorted.indices) {
            if (suppressed[i]) continue
            kept.add(sorted[i])
            if (kept.size >= maxResults) break

            for (j in i + 1 until sorted.size) {
                if (suppressed[j]) continue
                if (iou(sorted[i].boundingBox, sorted[j].boundingBox) > iouThreshold) {
                    suppressed[j] = true
                }
            }
        }

        return kept
    }

    /**
     * Calculate Intersection over Union (IoU) of two bounding boxes.
     */
    private fun iou(a: RectF, b: RectF): Float {
        val intersectLeft = max(a.left, b.left)
        val intersectTop = max(a.top, b.top)
        val intersectRight = min(a.right, b.right)
        val intersectBottom = min(a.bottom, b.bottom)

        val intersectArea = max(0f, intersectRight - intersectLeft) * max(0f, intersectBottom - intersectTop)
        if (intersectArea == 0f) return 0f

        val aArea = (a.right - a.left) * (a.bottom - a.top)
        val bArea = (b.right - b.left) * (b.bottom - b.top)
        val unionArea = aArea + bArea - intersectArea

        return if (unionArea > 0f) intersectArea / unionArea else 0f
    }
}
