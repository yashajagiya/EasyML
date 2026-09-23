package com.easyml.detection

import android.graphics.RectF
import com.easyml.classification.Classification
import com.easyml.classification.toJson
import com.easyml.core.LabelSource
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class SerializationTest {

    @Test
    fun testDetectionSerializationRoundTrip() {
        val original = Detection(
            boundingBox = RectF(12.5f, 25.0f, 150.75f, 200.5f),
            label = "person",
            labelIndex = 0,
            confidence = 0.95f
        )

        val jsonString = original.toJson()
        assertTrue(jsonString.contains("\"left\":12.5") || jsonString.contains("\"left\": 12.5"))
        assertTrue(jsonString.contains("\"label\":\"person\"") || jsonString.contains("\"label\": \"person\""))
        assertTrue(jsonString.contains("\"confidence\":0.95") || jsonString.contains("\"confidence\": 0.95"))

        val deserialized = Detection.fromJson(jsonString)
        assertEquals(original.label, deserialized.label)
        assertEquals(original.labelIndex, deserialized.labelIndex)
        assertEquals(original.confidence, deserialized.confidence, 0.0001f)
        assertEquals(original.boundingBox.left, deserialized.boundingBox.left, 0.0001f)
        assertEquals(original.boundingBox.top, deserialized.boundingBox.top, 0.0001f)
        assertEquals(original.boundingBox.right, deserialized.boundingBox.right, 0.0001f)
        assertEquals(original.boundingBox.bottom, deserialized.boundingBox.bottom, 0.0001f)
    }

    @Test
    fun testDetectionListSerialization() {
        val list = listOf(
            Detection(RectF(0f, 0f, 50f, 50f), "car", 2, 0.88f),
            Detection(RectF(100f, 100f, 200f, 250f), "dog", 16, 0.76f)
        )

        val jsonString = list.toJson()
        val deserializedList = Detection.fromJsonList(jsonString)

        assertEquals(2, deserializedList.size)
        assertEquals("car", deserializedList[0].label)
        assertEquals("dog", deserializedList[1].label)

        val detectionList = list.toDetectionList()
        val dListJson = detectionList.toJson()
        val deserializedDList = DetectionList.fromJson(dListJson)

        assertEquals(2, deserializedDList.size)
        assertEquals("car", deserializedDList[0].label)
        assertEquals("dog", deserializedDList[1].label)
    }

    @Test
    fun testClassificationSerialization() {
        val original = Classification(
            label = "tabby_cat",
            labelIndex = 281,
            confidence = 0.92f
        )

        val jsonString = original.toJson()
        val deserialized = Classification.fromJson(jsonString)

        assertEquals(original.label, deserialized.label)
        assertEquals(original.labelIndex, deserialized.labelIndex)
        assertEquals(original.confidence, deserialized.confidence, 0.0001f)

        val list = listOf(original, Classification("persian_cat", 282, 0.05f))
        val listJson = list.toJson()
        val deserializedList = Classification.fromJsonList(listJson)
        assertEquals(2, deserializedList.size)
        assertEquals("tabby_cat", deserializedList[0].label)
        assertEquals("persian_cat", deserializedList[1].label)
    }

    @Test
    fun testInferenceMetricsSerialization() {
        val original = InferenceMetrics(
            preprocessMs = 2.4,
            inferenceMs = 8.1,
            postprocessMs = 1.3,
            totalMs = 11.8
        )

        val jsonString = original.toJson()
        val deserialized = InferenceMetrics.fromJson(jsonString)

        assertEquals(original.preprocessMs, deserialized.preprocessMs, 0.001)
        assertEquals(original.inferenceMs, deserialized.inferenceMs, 0.001)
        assertEquals(original.postprocessMs, deserialized.postprocessMs, 0.001)
        assertEquals(original.totalMs, deserialized.totalMs, 0.001)
        assertEquals(original.fps, deserialized.fps, 0.01)
    }

    @Test
    fun testJsonLabelsParsingArray() {
        val jsonArray = """["person", "bicycle", "car", "motorcycle"]"""
        val labels = LabelSource.parseJsonLabels(jsonArray)
        assertEquals(listOf("person", "bicycle", "car", "motorcycle"), labels)
    }

    @Test
    fun testJsonLabelsParsingIndexMappedObject() {
        val jsonObject = """{"0": "person", "2": "car", "1": "bicycle"}"""
        val labels = LabelSource.parseJsonLabels(jsonObject)
        assertEquals(listOf("person", "bicycle", "car"), labels)
    }

    @Test
    fun testJsonLabelsParsingReversedMap() {
        val jsonObject = """{"person": 0, "car": 2, "bicycle": 1}"""
        val labels = LabelSource.parseJsonLabels(jsonObject)
        assertEquals(listOf("person", "bicycle", "car"), labels)
    }
}
