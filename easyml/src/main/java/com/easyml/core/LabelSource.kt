package com.easyml.core

import android.content.Context
import java.io.BufferedReader
import java.io.File
import java.io.InputStreamReader

/**
 * Represents the source of class labels for a model.
 *
 * Usage:
 * ```kotlin
 * LabelSource.Asset("labels.txt")           // From assets (one label per line)
 * LabelSource.ExternalFile(File("/path"))    // From device storage
 * LabelSource.StringList(listOf("cat","dog"))// Direct list
 * ```
 */
sealed class LabelSource {
    /** Load labels from the app's assets folder (one label per line). */
    data class Asset(val fileName: String) : LabelSource()

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
            is ExternalFile -> {
                file.readLines().filter { it.isNotBlank() }
            }
            is StringList -> labels
        }
    }
}
