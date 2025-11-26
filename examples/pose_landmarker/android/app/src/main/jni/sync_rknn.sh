#!/usr/bin/env bash

# Synchronize RKNN runtime headers and shared libraries into src/main/jni.
# Usage:
#   ./sync_rknn.sh /path/to/rknpu2
# or set RKNNPU2_ROOT beforehand.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
RKNN_ROOT="${1:-${RKNNPU2_ROOT:-}}"

if [[ -z "${RKNN_ROOT}" ]]; then
  echo "Usage: $0 /path/to/rknpu2 (or export RKNNPU2_ROOT)" >&2
  exit 1
fi

if [[ ! -d "${RKNN_ROOT}" ]]; then
  echo "RKNN root not found: ${RKNN_ROOT}" >&2
  exit 1
fi

headers_dir="${RKNN_ROOT}/include"
if [[ ! -d "${headers_dir}" ]]; then
  echo "Missing header directory: ${headers_dir}" >&2
  exit 1
fi

echo "Using RKNN root: ${RKNN_ROOT}"

mkdir -p "${ROOT}/include"
copied_header=false
for header in rknn_api.h rknn_custom_op.h rknn_matmul_api.h; do
  if [[ -f "${headers_dir}/${header}" ]]; then
    cp -f "${headers_dir}/${header}" "${ROOT}/include/"
    copied_header=true
  else
    echo "Warning: header not found, skipped ${headers_dir}/${header}" >&2
  fi
done

if [[ "${copied_header}" == "false" ]]; then
  echo "No RKNN headers copied; please check ${headers_dir}." >&2
  exit 1
fi

missing_required=0
for abi in arm64-v8a armeabi-v7a; do
  dest="${ROOT}/${abi}"
  mkdir -p "${dest}"

  lib_rknn="${RKNN_ROOT}/Android/${abi}/librknnrt.so"
  if [[ -f "${lib_rknn}" ]]; then
    cp -f "${lib_rknn}" "${dest}/"
    echo "Copied librknnrt.so for ${abi}"
  else
    echo "Missing required librknnrt.so for ${abi} under ${RKNN_ROOT}/Android/${abi}" >&2
    missing_required=1
  fi

  lib_rga="${RKNN_ROOT}/Android/${abi}/librga.so"
  if [[ -f "${lib_rga}" ]]; then
    cp -f "${lib_rga}" "${dest}/"
    echo "Copied optional librga.so for ${abi}"
  else
    echo "Info: librga.so not found for ${abi}, skipped (only needed when using RGA helpers)." >&2
  fi
done

if [[ "${missing_required}" -ne 0 ]]; then
  echo "Sync finished with errors: missing librknnrt.so for one or more ABIs." >&2
  exit 1
fi

echo "JNI libs/headers refreshed under ${ROOT}"
