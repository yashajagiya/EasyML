package android.graphics

/**
 * Lightweight mock of android.graphics.RectF for local JVM unit testing.
 * Prevents unit test failures caused by unmocked Android stubs.
 */
class RectF(
    @JvmField var left: Float = 0f,
    @JvmField var top: Float = 0f,
    @JvmField var right: Float = 0f,
    @JvmField var bottom: Float = 0f
) {
    constructor(r: RectF?) : this(
        left = r?.left ?: 0f,
        top = r?.top ?: 0f,
        right = r?.right ?: 0f,
        bottom = r?.bottom ?: 0f
    )

    fun width(): Float = right - left
    fun height(): Float = bottom - top

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is RectF) return false
        return left == other.left && top == other.top && right == other.right && bottom == other.bottom
    }

    override fun hashCode(): Int {
        var result = left.hashCode()
        result = 31 * result + top.hashCode()
        result = 31 * result + right.hashCode()
        result = 31 * result + bottom.hashCode()
        return result
    }

    override fun toString(): String = "RectF($left, $top, $right, $bottom)"
}
