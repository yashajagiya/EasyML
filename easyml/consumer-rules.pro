# Consumer ProGuard rules for EasyML
# Keep all public API classes
-keep class com.easyml.** { *; }
-keep class org.tensorflow.lite.** { *; }
-dontwarn org.tensorflow.lite.**

