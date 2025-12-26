#!/bin/bash

# Test build script for library modules
echo "======================================"
echo "Testing library module builds"
echo "======================================"

# Try building jumprope-counter-lib
echo ""
echo "Building jumprope-counter-lib..."
./gradlew :jumprope-counter-lib:assembleDebug

if [ $? -eq 0 ]; then
    echo "✓ jumprope-counter-lib build SUCCESS"
else
    echo "✗ jumprope-counter-lib build FAILED"
fi

# Try building pose-detector-lib
echo ""
echo "Building pose-detector-lib..."
./gradlew :pose-detector-lib:assembleDebug

if [ $? -eq 0 ]; then
    echo "✓ pose-detector-lib build SUCCESS"
else
    echo "✗ pose-detector-lib build FAILED"
fi

# Try building both
echo ""
echo "Building all modules..."
./gradlew assembleDebug

if [ $? -eq 0 ]; then
    echo "✓ All modules build SUCCESS"
else
    echo "✗ Some modules build FAILED"
fi

echo ""
echo "======================================"
echo "Build test complete"
echo "======================================"






