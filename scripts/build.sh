#!/bin/bash
# 프로젝트 빌드 스크립트

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

# CUTLASS 서브모듈 확인
if [ ! -d "3rd_party/cutlass/include" ]; then
    echo "CUTLASS 서브모듈이 없습니다. 초기화 중..."
    bash scripts/init_submodule.sh
fi

# Python 패키지 설치
echo "Python 패키지 빌드 및 설치 중..."
pip install -e .

echo "빌드 완료!"

