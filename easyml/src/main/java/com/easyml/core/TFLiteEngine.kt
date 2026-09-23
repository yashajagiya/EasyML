package com.easyml.core

import android.content.Context
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.Tensor
import org.tensorflow.lite.gpu.GpuDelegate
import java.io.Closeable
import java.nio.ByteBuffer

/**
 * Core TFLite interpreter wrapper. Manages the interpreter lifecycle,
 * provides thread-safe inference, and handles resource cleanup.
 *
 * This is an internal component — consumers use [com.easyml.detection.ObjectDetector],
 * [com.easyml.classification.ImageClassifier], or [com.easyml.raw.RawRunner] instead.
 */
internal class TFLiteEngine(
    context: Context,
    modelSource: ModelSource,
    device: InferenceDevice = InferenceDevice.CPU,
    numThreads: Int = 4,
    useFp16: Boolean = true
) : Closeable {

    private val interpreter: Interpreter
    private val gpuDelegate: GpuDelegate?
    private val lock = Any()

    /** Input tensor shape (e.g., [1, 640, 640, 3] or [1, 3, 640, 640]) */
    val inputShape: IntArray
        get() = interpreter.getInputTensor(0).shape()

    /** Output tensor shape (e.g., [1, 84, 8400] or [1, 300, 6]) */
    val outputShape: IntArray
        get() = interpreter.getOutputTensor(0).shape()

    /** Number of output tensors */
    val outputTensorCount: Int
        get() = interpreter.outputTensorCount

    /** Get shape of a specific output tensor */
    fun getOutputShape(index: Int): IntArray =
        interpreter.getOutputTensor(index).shape()

    /** Get data type of input tensor */
    val inputDataType: DataType
        get() = interpreter.getInputTensor(0).dataType()

    /** Get data type of primary output tensor */
    val outputDataType: DataType
        get() = interpreter.getOutputTensor(0).dataType()

    /** Quantization parameters for the primary output tensor */
    val outputQuantizationParams: Tensor.QuantizationParams
        get() = interpreter.getOutputTensor(0).quantizationParams()

    init {
        val modelBuffer = modelSource.resolve(context)
        var delegate: GpuDelegate? = null
        var interp: Interpreter? = null

        try {
            val options = Interpreter.Options()
            delegate = device.configure(options, numThreads, useFp16)
            interp = Interpreter(modelBuffer, options)
        } catch (e: Exception) {
            if (device == InferenceDevice.GPU_STRICT) {
                throw e
            }
            android.util.Log.w("EasyML", "Delegate initialization with $device failed (${e.message}), falling back to CPU")
            delegate?.close()
            delegate = null
            val fallbackOptions = Interpreter.Options().apply {
                setUseXNNPACK(true)
                val availableCores = Runtime.getRuntime().availableProcessors()
                setNumThreads(minOf(availableCores, numThreads.coerceIn(1, 4)))
            }
            interp = Interpreter(modelBuffer, fallbackOptions)
        }

        this.gpuDelegate = delegate
        this.interpreter = interp
    }

    /**
     * Run inference with a single input/output (most common case).
     * Thread-safe via synchronized block.
     */
    fun run(input: ByteBuffer, output: ByteBuffer) {
        synchronized(lock) {
            interpreter.run(input, output)
        }
    }

    /**
     * Run inference with a single input and single output array.
     */
    fun run(input: Any, output: Any) {
        synchronized(lock) {
            interpreter.run(input, output)
        }
    }

    /**
     * Run inference with multiple inputs/outputs.
     * Use for models with multiple output heads.
     */
    fun runMultiOutput(inputs: Array<Any>, outputs: MutableMap<Int, Any>) {
        synchronized(lock) {
            interpreter.runForMultipleInputsOutputs(inputs, outputs)
        }
    }

    /**
     * Read and dequantize output byte buffer into target float array.
     *
     * Automatically applies (val - zeroPoint) * scale for INT8/UINT8 quantized models.
     */
    fun readOutput(byteBuffer: ByteBuffer, floatArray: FloatArray) {
        byteBuffer.rewind()
        when (outputDataType) {
            DataType.FLOAT32 -> {
                byteBuffer.asFloatBuffer().get(floatArray)
            }
            DataType.INT8 -> {
                val params = outputQuantizationParams
                val scale = params.scale
                val zeroPoint = params.zeroPoint
                val count = minOf(floatArray.size, byteBuffer.remaining())
                for (i in 0 until count) {
                    val byteVal = byteBuffer.get().toInt()
                    floatArray[i] = (byteVal - zeroPoint) * scale
                }
            }
            DataType.UINT8 -> {
                val params = outputQuantizationParams
                val scale = params.scale
                val zeroPoint = params.zeroPoint
                val count = minOf(floatArray.size, byteBuffer.remaining())
                for (i in 0 until count) {
                    val ubyteVal = byteBuffer.get().toInt() and 0xFF
                    floatArray[i] = (ubyteVal - zeroPoint) * scale
                }
            }
            else -> {
                throw UnsupportedOperationException(
                    "Unsupported output tensor data type: $outputDataType. EasyML supports FLOAT32, INT8, and UINT8."
                )
            }
        }
    }

    override fun close() {
        interpreter.close()
        gpuDelegate?.close()
    }
}
