package com.easyml.core

import android.graphics.RectF
import kotlinx.serialization.KSerializer
import kotlinx.serialization.Serializable
import kotlinx.serialization.descriptors.SerialDescriptor
import kotlinx.serialization.encoding.Decoder
import kotlinx.serialization.encoding.Encoder

/**
 * Custom [KSerializer] for Android framework class [android.graphics.RectF].
 *
 * Encodes [RectF] as a structured JSON object with left, top, right, bottom coordinates:
 * ```json
 * {
 *   "left": 10.0,
 *   "top": 20.0,
 *   "right": 100.0,
 *   "bottom": 150.0
 * }
 * ```
 */
object RectFSerializer : KSerializer<RectF> {

    @Serializable
    private data class RectFSurrogate(
        val left: Float,
        val top: Float,
        val right: Float,
        val bottom: Float
    )

    override val descriptor: SerialDescriptor = RectFSurrogate.serializer().descriptor

    override fun serialize(encoder: Encoder, value: RectF) {
        val surrogate = RectFSurrogate(value.left, value.top, value.right, value.bottom)
        encoder.encodeSerializableValue(RectFSurrogate.serializer(), surrogate)
    }

    override fun deserialize(decoder: Decoder): RectF {
        val surrogate = decoder.decodeSerializableValue(RectFSurrogate.serializer())
        return RectF(surrogate.left, surrogate.top, surrogate.right, surrogate.bottom)
    }
}
