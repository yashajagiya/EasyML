package com.easyml.detection

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class NonMaxSuppressionTest {

    @Test
    fun testClassAwareNms_preservesOverlappingDifferentClasses() {
        val nms = NonMaxSuppression(iouThreshold = 0.45f, maxResults = 10, classAgnostic = false)

        // Box 1: Person (class 0) at (10, 10, 100, 100) with 0.90 conf
        val person = DetectionCandidate(10f, 10f, 100f, 100f, labelIndex = 0, confidence = 0.90f)
        // Box 2: Bicycle (class 1) almost identical box at (12, 12, 98, 98) with 0.85 conf
        val bicycle = DetectionCandidate(12f, 12f, 98f, 98f, labelIndex = 1, confidence = 0.85f)

        val candidates = listOf(person, bicycle)
        val result = mutableListOf<DetectionCandidate>()
        nms.process(candidates, result)

        // Both person and bicycle MUST survive because they belong to different classes!
        assertEquals("Both different-class detections must survive class-aware NMS", 2, result.size)
        assertEquals(0, result[0].labelIndex)
        assertEquals(1, result[1].labelIndex)
    }

    @Test
    fun testClassAwareNms_suppressesOverlappingSameClass() {
        val nms = NonMaxSuppression(iouThreshold = 0.45f, maxResults = 10, classAgnostic = false)

        // Two overlapping Person detections
        val personHigh = DetectionCandidate(10f, 10f, 100f, 100f, labelIndex = 0, confidence = 0.95f)
        val personLow = DetectionCandidate(12f, 12f, 98f, 98f, labelIndex = 0, confidence = 0.70f)

        val candidates = listOf(personHigh, personLow)
        val result = mutableListOf<DetectionCandidate>()
        nms.process(candidates, result)

        assertEquals("Lower confidence same-class detection must be suppressed", 1, result.size)
        assertEquals(0.95f, result[0].confidence, 0.001f)
    }

    @Test
    fun testClassAgnosticNms_suppressesAcrossDifferentClasses() {
        val nms = NonMaxSuppression(iouThreshold = 0.45f, maxResults = 10, classAgnostic = true)

        val person = DetectionCandidate(10f, 10f, 100f, 100f, labelIndex = 0, confidence = 0.90f)
        val bicycle = DetectionCandidate(12f, 12f, 98f, 98f, labelIndex = 1, confidence = 0.85f)

        val candidates = listOf(person, bicycle)
        val result = mutableListOf<DetectionCandidate>()
        nms.process(candidates, result)

        // Under class-agnostic NMS, the bicycle is suppressed
        assertEquals(1, result.size)
        assertEquals(0, result[0].labelIndex)
    }

    @Test
    fun testIoU_calculation() {
        val nms = NonMaxSuppression()
        val a = DetectionCandidate(0f, 0f, 100f, 100f)
        val b = DetectionCandidate(0f, 0f, 100f, 100f)
        assertEquals(1.0f, nms.calculateIoU(a, b), 0.001f)

        val c = DetectionCandidate(200f, 200f, 300f, 300f)
        assertEquals(0.0f, nms.calculateIoU(a, c), 0.001f)
    }
}
