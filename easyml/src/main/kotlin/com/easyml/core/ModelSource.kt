package com.easyml.core

import android.content.Context
import java.io.File
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.channels.FileChannel

/**
 * Represents the source of a TFLite model file.
 *
 * Usage:
 * ```kotlin
 * ModelSource.Asset("model.tflite")       // From assets folder
 * ModelSource.ExternalFile(File("/path"))  // From device storage
 * ModelSource.Buffer(byteBuffer)          // From a pre-loaded ByteBuffer
 * ```
 */
sealed class ModelSource {
    /** Load model from the app's assets folder. */
    data class Asset(val fileName: String) : ModelSource()

    /** Load model from an external file path. */
    data class ExternalFile(val file: File) : ModelSource()

    /** Use a pre-loaded ByteBuffer directly. */
    data class Buffer(val buffer: ByteBuffer) : ModelSource()

    /**
     * Resolves this ModelSource into a MappedByteBuffer suitable for the TFLite Interpreter.
     * Uses memory-mapped I/O for zero-copy loading — the most efficient approach.
     */
    internal fun resolve(context: Context): ByteBuffer {
        return when (this) {
            is Asset -> {
                val assetFd = context.assets.openFd(fileName)
                val inputStream = FileInputStream(assetFd.fileDescriptor)
                val mappedBuffer = inputStream.channel.map(
                    FileChannel.MapMode.READ_ONLY,
                    assetFd.startOffset,
                    assetFd.declaredLength
                )
                inputStream.close()
                mappedBuffer
            }
            is ExternalFile -> {
                val inputStream = FileInputStream(file)
                val channel = inputStream.channel
                val mappedBuffer = channel.map(
                    FileChannel.MapMode.READ_ONLY,
                    0,
                    channel.size()
                )
                inputStream.close()
                mappedBuffer
            }
            is Buffer -> buffer
        }
    }
}
