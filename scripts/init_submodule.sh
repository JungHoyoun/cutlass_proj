#!/bin/bash
# CUTLASS 서브모듈 초기화 스크립트

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "CUTLASS 서브모듈 초기화 중..."

cd "$PROJECT_ROOT"

# 서브모듈 초기화
if [ ! -d "3rd_party/cutlass" ]; then
    echo "CUTLASS 서브모듈 추가 중..."
    git submodule add https://github.com/NVIDIA/cutlass.git 3rd_party/cutlass
fi

# 서브모듈 업데이트
git submodule update --init --recursive

echo "CUTLASS 서브모듈 초기화 완료!"

