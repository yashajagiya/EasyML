package com.easyml.detection

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class YoloDecodersTest {

    @Test
    fun testYolo26EndToEndDecoder() {
        val decoder = Yolo26EndToEndDecoder(coordinateFormat = CoordinateFormat.PIXEL_SPACE)
        assertTrue(decoder.isNmsFree)

        // 3 detections, 6 attributes: [x1, y1, x2, y2, conf, class_id]
        val output = floatArrayOf(
            10f, 20f, 100f, 200f, 0.95f, 0f, // Detection 0: Person
            50f, 60f, 150f, 250f, 0.80f, 2f, // Detection 1: Car
            0f, 0f, 10f, 10f, 0.15f, 1f       // Detection 2: Low confidence
        )
        val shape = intArrayOf(1, 3, 6)
        val candidates = mutableListOf<DetectionCandidate>()

        decoder.decode(output, shape, 640, 640, confidenceThreshold = 0.50f, candidates)

        assertEquals("Only 2 candidates above 0.50 conf should be retained", 2, candidates.size)

        // Verify Candidate 0
        assertEquals(10f, candidates[0].left, 0.01f)
        assertEquals(20f, candidates[0].top, 0.01f)
        assertEquals(100f, candidates[0].right, 0.01f)
        assertEquals(200f, candidates[0].bottom, 0.01f)
        assertEquals(0, candidates[0].labelIndex)
        assertEquals(0.95f, candidates[0].confidence, 0.01f)

        // Verify Candidate 1
        assertEquals(50f, candidates[1].left, 0.01f)
        assertEquals(2, candidates[1].labelIndex)
        assertEquals(0.80f, candidates[1].confidence, 0.01f)
    }

    @Test
    fun testYoloV8Decoder_transposed() {
        val decoder = YoloV8Decoder(isTransposed = true, coordinateFormat = CoordinateFormat.PIXEL_SPACE)

        // Shape: [1, 6, 2] -> 4 coordinates + 2 classes, 2 candidate detections
        // Rows:
        // row 0 (cx): [100f, 200f]
        // row 1 (cy): [150f, 250f]
        // row 2 (w):  [50f, 60f]
        // row 3 (h):  [70f, 80f]
        // row 4 (class0): [0.90f, 0.10f]
        // row 5 (class1): [0.05f, 0.85f]
        val output = floatArrayOf(
            100f, 200f, // cx
            150f, 250f, // cy
            50f, 60f,   // w
            70f, 80f,   // h
            0.90f, 0.10f, // class 0
            0.05f, 0.85f  // class 1
        )
        val shape = intArrayOf(1, 6, 2)
        val candidates = mutableListOf<DetectionCandidate>()

        decoder.decode(output, shape, 640, 640, confidenceThreshold = 0.50f, candidates)

        assertEquals(2, candidates.size)

        // Candidate 0: cx=100, cy=150, w=50, h=70 -> left=75, top=115, right=125, bottom=185
        assertEquals(75f, candidates[0].left, 0.01f)
        assertEquals(115f, candidates[0].top, 0.01f)
        assertEquals(125f, candidates[0].right, 0.01f)
        assertEquals(185f, candidates[0].bottom, 0.01f)
        assertEquals(0, candidates[0].labelIndex)
        assertEquals(0.90f, candidates[0].confidence, 0.01f)

        // Candidate 1: cx=200, cy=250, w=60, h=80 -> left=170, top=210, right=230, bottom=290
        assertEquals(170f, candidates[1].left, 0.01f)
        assertEquals(210f, candidates[1].top, 0.01f)
        assertEquals(1, candidates[1].labelIndex)
        assertEquals(0.85f, candidates[1].confidence, 0.01f)
    }

    @Test
    fun testYoloV5Decoder_withObjectness() {
        val decoder = YoloV5Decoder(hasObjectness = true, coordinateFormat = CoordinateFormat.PIXEL_SPACE)

        // Shape: [1, 2, 7] -> 4 coords + 1 obj_conf + 2 classes
        // Det 0: cx=100, cy=100, w=40, h=40, obj=0.9, c0=0.8, c1=0.1 -> score = 0.9 * 0.8 = 0.72
        // Det 1: cx=200, cy=200, w=50, h=50, obj=0.5, c0=0.2, c1=0.4 -> score = 0.5 * 0.4 = 0.20 (below 0.50 threshold)
        val output = floatArrayOf(
            100f, 100f, 40f, 40f, 0.9f, 0.8f, 0.1f,
            200f, 200f, 50f, 50f, 0.5f, 0.2f, 0.4f
        )
        val shape = intArrayOf(1, 2, 7)
        val candidates = mutableListOf<DetectionCandidate>()

        decoder.decode(output, shape, 640, 640, confidenceThreshold = 0.50f, candidates)

        assertEquals("Only Det 0 should pass confidence threshold", 1, candidates.size)
        assertEquals(0.72f, candidates[0].confidence, 0.01f)
        assertEquals(0, candidates[0].labelIndex)
        assertEquals(80f, candidates[0].left, 0.01f)
        assertEquals(120f, candidates[0].right, 0.01f)
    }
}
