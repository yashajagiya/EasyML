package com.easyml.classification

import androidx.compose.runtime.Immutable
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json

/**
 * A single classification result.
 *
 * @property label Human-readable class label
 * @property labelIndex Index of the class in the labels list
 * @property confidence Classification confidence score [0.0, 1.0]
 */
@Immutable
@Serializable
data class Classification(
    val label: String,
    val labelIndex: Int,
    val confidence: Float
) {
    override fun toString(): String = "$label (${(confidence * 100).toInt()}%)"

    /** Serialize this classification to a JSON string */
    fun toJson(json: Json = defaultJson): String = json.encodeToString(serializer(), this)

    companion object {
        private val defaultJson = Json { ignoreUnknownKeys = true; prettyPrint = false }

        /** Parse a [Classification] from a JSON string */
        fun fromJson(jsonString: String, json: Json = defaultJson): Classification =
            json.decodeFromString(serializer(), jsonString)

        /** Parse a list of [Classification]s from a JSON string */
        fun fromJsonList(jsonString: String, json: Json = defaultJson): List<Classification> =
            json.decodeFromString(jsonString)
    }
}

/** Extension function to serialize a `List<Classification>` directly to JSON */
fun List<Classification>.toJson(json: Json = Json): String = json.encodeToString(this)

