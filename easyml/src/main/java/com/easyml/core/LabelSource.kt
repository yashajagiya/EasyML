package com.easyml.core

import android.content.Context
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.int
import kotlinx.serialization.json.intOrNull
import kotlinx.serialization.json.jsonPrimitive
import java.io.BufferedReader
import java.io.File
import java.io.InputStreamReader

/**
 * Represents the source of class labels for a model.
 *
 * Usage:
 * ```kotlin
 * LabelSource.Asset("labels.txt")            // From assets (one label per line)
 * LabelSource.JsonAsset("labels.json")       // From assets JSON array ["cat","dog"] or map {"0":"cat"}
 * LabelSource.JsonString(jsonContent)        // Direct JSON string
 * LabelSource.ExternalFile(File("/path"))    // From device storage
 * LabelSource.StringList(listOf("cat","dog"))// Direct list
 * ```
 */
sealed class LabelSource {
    /** Load labels from the app's assets folder (one label per line). */
    data class Asset(val fileName: String) : LabelSource()

    /** Load labels from a JSON file in the app's assets folder (supports array and index-mapped object). */
    data class JsonAsset(val fileName: String) : LabelSource()

    /** Load labels directly from a JSON string (supports array and index-mapped object). */
    data class JsonString(val jsonContent: String) : LabelSource()

    /** Load labels from an external file (one label per line). */
    data class ExternalFile(val file: File) : LabelSource()

    /** Provide labels directly as a list. */
    data class StringList(val labels: List<String>) : LabelSource()

    /**
     * Resolves labels into a List<String>.
     */
    internal fun resolve(context: Context): List<String> {
        return when (this) {
            is Asset -> {
                val reader = BufferedReader(InputStreamReader(context.assets.open(fileName)))
                val labels = reader.readLines().filter { it.isNotBlank() }
                reader.close()
                labels
            }
            is JsonAsset -> {
                val jsonText = context.assets.open(fileName).bufferedReader().use { it.readText() }
                parseJsonLabels(jsonText)
            }
            is JsonString -> {
                parseJsonLabels(jsonContent)
            }
            is ExternalFile -> {
                file.readLines().filter { it.isNotBlank() }
            }
            is StringList -> labels
        }
    }

    companion object {
        /**
         * Parses a JSON string containing either a list of labels `["cat", "dog"]`
         * or an index-mapped dictionary `{"0": "cat", "1": "dog"}` / `{"cat": 0, "dog": 1}`.
         */
        fun parseJsonLabels(jsonContent: String): List<String> {
            val element = Json.parseToJsonElement(jsonContent)
            return when (element) {
                is JsonArray -> {
                    element.map { it.jsonPrimitive.content }
                }
                is JsonObject -> {
                    // Check if keys are integer indexes (e.g. {"0": "person", "1": "car"})
                    val isIntKeys = element.keys.isNotEmpty() && element.keys.all { it.toIntOrNull() != null }
                    if (isIntKeys) {
                        element.entries
                            .sortedBy { it.key.toInt() }
                            .map { it.value.jsonPrimitive.content }
                    } else {
                        // Check if values are integer indexes (e.g. {"person": 0, "car": 1})
                        val isIntValues = element.values.isNotEmpty() && element.values.all {
                            try { it.jsonPrimitive.intOrNull != null } catch (_: Throwable) { false }
                        }
                        if (isIntValues) {
                            element.entries
                                .sortedBy { it.value.jsonPrimitive.int }
                                .map { it.key }
                        } else {
                            element.values.map { it.jsonPrimitive.content }
                        }
                    }
                }
                else -> emptyList()
            }
        }
    }
}

