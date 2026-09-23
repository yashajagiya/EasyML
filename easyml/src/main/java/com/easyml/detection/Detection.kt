package com.easyml.detection

import android.graphics.RectF
import androidx.compose.runtime.Immutable

/**
 * A single object detection result.
 *
 * @property boundingBox Bounding box in the original image's pixel coordinates
 * @property label Human-readable class label (e.g., "person", "car")
 * @property labelIndex Index of the class in the labels list
 * @property confidence Detection confidence score [0.0, 1.0]
 */
@Immutable
data class Detection(
    val boundingBox: RectF,
    val label: String,
    val labelIndex: Int,
    val confidence: Float
) {
    override fun toString(): String =
        "$label (${(confidence * 100).toInt()}%) [${boundingBox.left.toInt()},${boundingBox.top.toInt()},${boundingBox.right.toInt()},${boundingBox.bottom.toInt()}]"
}

/**
 * Immutable wrapper for a list of [Detection] objects.
 * Guarantees stability in Jetpack Compose to optimize recomposition.
 */
@Immutable
data class DetectionList(
    val items: List<Detection> = emptyList()
) {
    val size: Int get() = items.size
    operator fun get(index: Int): Detection = items[index]
    operator fun iterator(): Iterator<Detection> = items.iterator()
}

fun List<Detection>.toDetectionList(): DetectionList = DetectionList(this)

