# JNI 目录说明

`src/main/jni` 直接作为 Gradle 的 `jniLibs` 目录，用来打包预编译的原生库。目录结构保持如下：

- `arm64-v8a/`、`armeabi-v7a/`：放置对应 ABI 的 `librknnrt.so`（如有 `librga.so` 也可一并放入）。
- `include/`：放置 `rknn_api.h`、`rknn_custom_op.h`、`rknn_matmul_api.h` 等头文件，便于后续接 JNI/NDK 代码。
- `sync_rknn.sh`：从 RKNN runtime 发布包或 `rknn_model_zoo/3rdparty/rknpu2` 目录自动同步上述文件。

## 快速同步

1. 将 RKNN runtime（或 `rknn_model_zoo` 仓库中的 `3rdparty/rknpu2`）解压到本地。
2. 运行脚本同步：

```bash
cd app/src/main/jni
./sync_rknn.sh /path/to/rknpu2
```

也可以提前导出 `RKNNPU2_ROOT=/path/to/rknpu2`，然后直接执行脚本。

脚本会：
- 校验 `include` 和 `Android/<abi>/librknnrt.so` 是否存在。
- 为 `arm64-v8a`、`armeabi-v7a` 拷贝 `librknnrt.so`，并在存在时附带 `librga.so`。
- 同步 RKNN 头文件到 `include/`。

若需要额外 ABI 或其它依赖，可按同样方式在对应子目录中放入 `.so` 文件。
