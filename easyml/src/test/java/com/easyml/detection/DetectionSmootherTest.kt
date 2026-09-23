package com.easyml.detection

import android.graphics.RectF
import org.junit.Assert.assertEquals
import org.junit.Test

class DetectionSmootherTest {

    @Test
    fun testDetectionSmoother_interpolatesCoordinates() {
        val smoother = DetectionSmoother(smoothingFactor = 0.5f, iouMatchThreshold = 0.30f)

        // Frame 1: Person at (0, 0, 100, 100)
        val frame1 = listOf(
            Detection(
                boundingBox = RectF(0f, 0f, 100f, 100f),
                label = "person",
                labelIndex = 0,
                confidence = 0.90f
            )
        )
        val smoothed1 = smoother.update(frame1)
        assertEquals(0f, smoothed1[0].boundingBox.left, 0.001f)

        // Frame 2: Person moves slightly to (10, 10, 110, 110)
        val frame2 = listOf(
            Detection(
                boundingBox = RectF(10f, 10f, 110f, 110f),
                label = "person",
                labelIndex = 0,
                confidence = 0.92f
            )
        )
        val smoothed2 = smoother.update(frame2)

        // With alpha=0.5, new position should be 0.5 * 10 + 0.5 * 0 = 5.0
        assertEquals(5f, smoothed2[0].boundingBox.left, 0.01f)
        assertEquals(5f, smoothed2[0].boundingBox.top, 0.01f)
        assertEquals(105f, smoothed2[0].boundingBox.right, 0.01f)
        assertEquals(105f, smoothed2[0].boundingBox.bottom, 0.01f)
    }

    @Test
    fun testDetectionSmoother_resetClearsHistory() {
        val smoother = DetectionSmoother(smoothingFactor = 0.5f)

        val frame1 = listOf(
            Detection(RectF(0f, 0f, 100f, 100f), "person", 0, 0.9f)
        )
        smoother.update(frame1)

        smoother.reset()

        val frame2 = listOf(
            Detection(RectF(50f, 50f, 150f, 150f), "person", 0, 0.9f)
        )
        val result = smoother.update(frame2)

        // Because of reset, it should not interpolate against frame1
        assertEquals(50f, result[0].boundingBox.left, 0.001f)
    }
}
