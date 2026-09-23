package com.easyml.detection

import android.graphics.RectF
import androidx.compose.runtime.Immutable
import com.easyml.core.RectFSerializer
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json

/**
 * A single object detection result.
 *
 * @property boundingBox Bounding box in the original image's pixel coordinates
 * @property label Human-readable class label (e.g., "person", "car")
 * @property labelIndex Index of the class in the labels list
 * @property confidence Detection confidence score [0.0, 1.0]
 */
@Immutable
@Serializable
data class Detection(
    @Serializable(with = RectFSerializer::class)
    val boundingBox: RectF,
    val label: String,
    val labelIndex: Int,
    val confidence: Float
) {
    override fun toString(): String =
        "$label (${(confidence * 100).toInt()}%) [${boundingBox.left.toInt()},${boundingBox.top.toInt()},${boundingBox.right.toInt()},${boundingBox.bottom.toInt()}]"

    /** Serialize this detection to a JSON string */
    fun toJson(json: Json = defaultJson): String = json.encodeToString(serializer(), this)

    companion object {
        private val defaultJson = Json { ignoreUnknownKeys = true; prettyPrint = false }

        /** Parse a [Detection] from a JSON string */
        fun fromJson(jsonString: String, json: Json = defaultJson): Detection =
            json.decodeFromString(serializer(), jsonString)

        /** Parse a list of [Detection]s from a JSON string */
        fun fromJsonList(jsonString: String, json: Json = defaultJson): List<Detection> =
            json.decodeFromString(jsonString)
    }
}

/**
 * Immutable wrapper for a list of [Detection] objects.
 * Guarantees stability in Jetpack Compose to optimize recomposition.
 */
@Immutable
@Serializable
data class DetectionList(
    val items: List<Detection> = emptyList()
) {
    val size: Int get() = items.size
    operator fun get(index: Int): Detection = items[index]
    operator fun iterator(): Iterator<Detection> = items.iterator()

    /** Serialize this detection list to a JSON string */
    fun toJson(json: Json = Json): String = json.encodeToString(serializer(), this)

    companion object {
        /** Parse a [DetectionList] from a JSON string */
        fun fromJson(jsonString: String, json: Json = Json): DetectionList =
            json.decodeFromString(serializer(), jsonString)
    }
}

fun List<Detection>.toDetectionList(): DetectionList = DetectionList(this)

/** Extension function to serialize a `List<Detection>` directly to JSON */
fun List<Detection>.toJson(json: Json = Json): String = json.encodeToString(this)


