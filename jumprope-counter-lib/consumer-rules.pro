# Keep native methods
-keepclasseswithmembernames class * {
    native <methods>;
}

# Keep JumpRopeCounter class and its methods
-keep class com.yolo.jumprope.JumpRopeCounter { *; }






