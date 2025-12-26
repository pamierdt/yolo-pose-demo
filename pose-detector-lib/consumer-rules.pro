# Keep native methods
-keepclasseswithmembernames class * {
    native <methods>;
}

# Keep RknnRunner class and its methods
-keep class com.yolo.pose.detector.RknnRunner { *; }





