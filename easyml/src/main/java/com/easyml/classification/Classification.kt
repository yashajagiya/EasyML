package com.easyml.classification

import androidx.compose.runtime.Immutable

/**
 * A single classification result.
 *
 * @property label Human-readable class label
 * @property labelIndex Index of the class in the labels list
 * @property confidence Classification confidence score [0.0, 1.0]
 */
@Immutable
data class Classification(
    val label: String,
    val labelIndex: Int,
    val confidence: Float
) {
    override fun toString(): String = "$label (${(confidence * 100).toInt()}%)"
}
