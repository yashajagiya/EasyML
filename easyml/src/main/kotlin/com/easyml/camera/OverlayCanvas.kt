package com.easyml.camera

import androidx.compose.foundation.Canvas
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.CornerRadius
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.drawText
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.rememberTextMeasurer
import androidx.compose.ui.unit.sp
import com.easyml.detection.Detection
import com.easyml.detection.DetectionList

/**
 * Curated high-contrast palette for real-time object detection visual distinction.
 */
object DetectionColorPalette {
    val Vibrant = listOf(
        Color(0xFF00E5FF), // Electric Cyan
        Color(0xFF00E676), // Neon Emerald
        Color(0xFFFF9100), // Sunset Orange
        Color(0xFFFF1744), // Cyber Coral Red
        Color(0xFF7C4DFF), // Electric Violet
        Color(0xFFFFD600), // Vivid Amber Gold
        Color(0xFFD500F9), // Cyberpunk Magenta
        Color(0xFF1DE9B6), // Bright Mint Teal
        Color(0xFF2979FF), // Royal Azure Blue
        Color(0xFFFF4081), // Neon Pink
        Color(0xFFAEEA00), // Electric Lime
        Color(0xFF00B0FF), // Sky Blue
        Color(0xFFFF6E40), // Bright Tangerine
        Color(0xFF651FFF), // Deep Purple
        Color(0xFFF50057), // Crimson Rose
        Color(0xFF00E6B8)  // Aquamarine
    )

    /**
     * Returns a dynamic, consistent color for a given class index.
     * Uses the vibrant palette, with golden-ratio hue distribution fallback for classes beyond the palette.
     */
    fun getColorForClass(labelIndex: Int): Color {
        val safeIndex = if (labelIndex >= 0) labelIndex else -labelIndex
        if (safeIndex in Vibrant.indices) {
            return Vibrant[safeIndex]
        }
        val hue = ((safeIndex * 137.508f) % 360f)
        return Color.hsl(hue = hue, saturation = 0.90f, lightness = 0.55f)
    }
}

/**
 * Compose Canvas overlay that draws dynamic bounding boxes and badges on top of the camera preview.
 *
 * Automatically maps detection coordinates from image space to screen space.
 * Features:
 * - Dynamic per-class color differentiation from [DetectionColorPalette]
 * - Translucent box fill for modern visual depth
 * - Frosted pill badges with confidence indicator
 * - Full support for custom color providers and front-camera mirroring
 *
 * @param detections Current detection results (wrapped in [DetectionList] for compile-time stability)
 * @param imageWidth Width of the analyzed image
 * @param imageHeight Height of the analyzed image
 * @param modifier Compose modifier
 * @param boxColor Optional override color for all bounding boxes. If null, dynamic per-class colors are used.
 * @param colorProvider Optional custom function returning a [Color] for any [Detection]
 * @param strokeWidth Width of box outlines (default: 3.5f)
 * @param cornerRadius Rounded corner radius for bounding boxes and label tags (default: 8f)
 * @param labelSize Font size for labels (default: 13sp)
 * @param showLabels Whether to show class name on the bounding box (default: true)
 * @param showConfidence Whether to show confidence percentage (default: true)
 * @param showFill Whether to draw a subtle translucent fill inside the box (default: true)
 * @param isMirrored Set true for front camera to flip X coordinates
 */
@Composable
fun DetectionOverlay(
    detections: DetectionList,
    imageWidth: Int,
    imageHeight: Int,
    modifier: Modifier = Modifier,
    boxColor: Color? = null,
    colorProvider: ((Detection) -> Color)? = null,
    strokeWidth: Float = 3.5f,
    cornerRadius: Float = 8f,
    labelSize: Int = 13,
    showLabels: Boolean = true,
    showConfidence: Boolean = true,
    showFill: Boolean = true,
    isMirrored: Boolean = false
) {
    val textMeasurer = rememberTextMeasurer()
    val labelStyle = TextStyle(
        color = Color.White,
        fontSize = labelSize.sp,
        fontWeight = FontWeight.SemiBold
    )
    val pillBgColor = Color(0xEE141414)

    Canvas(modifier = modifier) {
        if (imageWidth <= 0 || imageHeight <= 0) return@Canvas

        // Map from analyzed image coordinates to PreviewView FILL_CENTER space
        val scale = maxOf(size.width / imageWidth.toFloat(), size.height / imageHeight.toFloat())
        val offsetX = (size.width - imageWidth * scale) * 0.5f
        val offsetY = (size.height - imageHeight * scale) * 0.5f

        for (detection in detections) {
            val box = detection.boundingBox

            // Map coordinates to canvas space
            var left = box.left * scale + offsetX
            var right = box.right * scale + offsetX
            val top = box.top * scale + offsetY
            val bottom = box.bottom * scale + offsetY

            // Mirror for front camera
            if (isMirrored) {
                val tempLeft = size.width - right
                right = size.width - left
                left = tempLeft
            }

            val rectWidth = right - left
            val rectHeight = bottom - top

            // Skip zero or inverted boxes
            if (rectWidth <= 0f || rectHeight <= 0f) continue

            // Determine dynamic color for this object
            val themeColor = colorProvider?.invoke(detection)
                ?: boxColor
                ?: DetectionColorPalette.getColorForClass(detection.labelIndex)

            // 1. Subtle translucent fill inside bounding box
            if (showFill) {
                if (cornerRadius > 0f) {
                    drawRoundRect(
                        color = themeColor.copy(alpha = 0.12f),
                        topLeft = Offset(left, top),
                        size = Size(rectWidth, rectHeight),
                        cornerRadius = CornerRadius(cornerRadius, cornerRadius)
                    )
                } else {
                    drawRect(
                        color = themeColor.copy(alpha = 0.12f),
                        topLeft = Offset(left, top),
                        size = Size(rectWidth, rectHeight)
                    )
                }
            }

            // 2. High-contrast bounding box outline
            if (cornerRadius > 0f) {
                drawRoundRect(
                    color = themeColor,
                    topLeft = Offset(left, top),
                    size = Size(rectWidth, rectHeight),
                    cornerRadius = CornerRadius(cornerRadius, cornerRadius),
                    style = Stroke(width = strokeWidth)
                )
            } else {
                drawRect(
                    color = themeColor,
                    topLeft = Offset(left, top),
                    size = Size(rectWidth, rectHeight),
                    style = Stroke(width = strokeWidth)
                )
            }

            // 3. Modern pill badge for label & confidence
            if (showLabels || showConfidence) {
                val labelText = when {
                    showLabels && showConfidence -> "${detection.label} • ${(detection.confidence * 100).toInt()}%"
                    showLabels -> detection.label
                    else -> "${(detection.confidence * 100).toInt()}%"
                }

                val textLayoutResult = textMeasurer.measure(labelText, labelStyle)
                val textWidth = textLayoutResult.size.width.toFloat()
                val textHeight = textLayoutResult.size.height.toFloat()
                val padH = 10f
                val padV = 5f
                val pillWidth = textWidth + padH * 2 + 10f // Extra space for color bullet
                val pillHeight = textHeight + padV * 2
                val pillRadius = CornerRadius(pillHeight * 0.5f, pillHeight * 0.5f)

                // Position badge above box, or just inside if near screen top
                val labelTop = if (top - pillHeight - 4f >= 0f) {
                    top - pillHeight - 4f
                } else {
                    top + 4f
                }
                val labelLeft = left.coerceIn(0f, (size.width - pillWidth).coerceAtLeast(0f))

                // Pill background
                drawRoundRect(
                    color = pillBgColor,
                    topLeft = Offset(labelLeft, labelTop),
                    size = Size(pillWidth, pillHeight),
                    cornerRadius = pillRadius
                )

                // Pill subtle accent border
                drawRoundRect(
                    color = themeColor.copy(alpha = 0.70f),
                    topLeft = Offset(labelLeft, labelTop),
                    size = Size(pillWidth, pillHeight),
                    cornerRadius = pillRadius,
                    style = Stroke(width = 1.5f)
                )

                // Color accent dot / bullet
                val dotRadius = 3.5f
                val dotCenterY = labelTop + pillHeight * 0.5f
                val dotCenterX = labelLeft + padH + dotRadius
                drawCircle(
                    color = themeColor,
                    radius = dotRadius,
                    center = Offset(dotCenterX, dotCenterY)
                )

                // Text
                drawText(
                    textMeasurer = textMeasurer,
                    text = labelText,
                    style = labelStyle,
                    topLeft = Offset(dotCenterX + dotRadius + 5f, labelTop + padV)
                )
            }
        }
    }
}
