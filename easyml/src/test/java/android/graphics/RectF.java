package android.graphics;

/**
 * Lightweight mock of android.graphics.RectF for local JVM unit testing.
 * Prevents unit test failures caused by unmocked Android stubs.
 */
public class RectF {
    public float left;
    public float top;
    public float right;
    public float bottom;

    public RectF() {}

    public RectF(float left, float top, float right, float bottom) {
        this.left = left;
        this.top = top;
        this.right = right;
        this.bottom = bottom;
    }

    public RectF(RectF r) {
        if (r != null) {
            this.left = r.left;
            this.top = r.top;
            this.right = r.right;
            this.bottom = r.bottom;
        }
    }

    public float width() {
        return right - left;
    }

    public float height() {
        return bottom - top;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        RectF rectF = (RectF) o;
        return Float.compare(rectF.left, left) == 0 &&
               Float.compare(rectF.top, top) == 0 &&
               Float.compare(rectF.right, right) == 0 &&
               Float.compare(rectF.bottom, bottom) == 0;
    }

    @Override
    public String toString() {
        return "RectF(" + left + ", " + top + ", " + right + ", " + bottom + ")";
    }
}
