package com.easyml.camera

import androidx.compose.ui.graphics.Color
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertNotNull
import org.junit.Test

class OverlayPaletteTest {

    @Test
    fun testPaletteDistinctColors() {
        val colors = mutableSetOf<Color>()
        for (i in 0 until 10) {
            val c = DetectionColorPalette.getColorForClass(i)
            assertNotNull(c)
            colors.add(c)
        }
        assertEquals("First 10 classes should all have distinct colors", 10, colors.size)
    }

    @Test
    fun testPaletteConsistency() {
        val c1 = DetectionColorPalette.getColorForClass(3)
        val c2 = DetectionColorPalette.getColorForClass(3)
        assertEquals("Same class index should consistently yield the same color", c1, c2)
    }

    @Test
    fun testGoldenRatioFallbackForLargeIndices() {
        // Classes beyond Vibrant.size
        val cLarge1 = DetectionColorPalette.getColorForClass(50)
        val cLarge2 = DetectionColorPalette.getColorForClass(51)
        assertNotNull(cLarge1)
        assertNotNull(cLarge2)
        assertNotEquals("Adjacent large classes should have different hues", cLarge1, cLarge2)
    }

    @Test
    fun testSafeHandlingOfNegativeIndex() {
        val c = DetectionColorPalette.getColorForClass(-1)
        assertNotNull(c)
    }
}
