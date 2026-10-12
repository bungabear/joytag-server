# JoyTag Server

JoyTag ONNX 모델을 HTTP API로 제공하는 CPU Docker 서버입니다. 공식 모델과
태그 목록이 이미지에 포함되어 있어 모델을 별도로 내려받거나 마운트할 필요가
없습니다. 기존 Twi JoyTag 클라이언트의 `/analyze` API와 호환됩니다.

## 실행

이미지는 `linux/amd64`용이며 GPU 없이 실행합니다.

```sh
docker run -d --name joytag-server --restart unless-stopped \
  -p 127.0.0.1:18001:8000 \
  ghcr.io/bungabear/joytag-server:latest
```

또는 이 저장소의 `compose.yaml`을 사용합니다.

```sh
docker compose up -d
```

API 문서: <http://127.0.0.1:18001/docs> · 상태 확인: <http://127.0.0.1:18001/health>

GHCR 패키지의 접근 권한이 private이면 `read:packages` 권한의 토큰으로 먼저
`docker login ghcr.io`를 수행해야 합니다.

## 이미지 분석

```sh
curl -F "files=@frame1.jpg" -F "files=@frame2.jpg" \
  "http://127.0.0.1:18001/analyze?threshold=0.4&top_k=15"
```

PowerShell에서는 `curl.exe`를 사용하세요. `files`는 여러 이미지를 받으며
`image`(단일 이미지), `images`(여러 이미지)도 지원합니다.

| 항목 | 동작 |
| --- | --- |
| `GET /health` | 모델 정보, 리비전·해시, CPU 스레드, 실행 환경, 로딩 시간 |
| `POST /analyze` | multipart 이미지별 태그와 전처리·추론·후처리 시간 |
| `threshold` | 0~1, 기본 0.4. `tags`에는 이 값보다 높은 확률만 포함 |
| `top_k` | 1~500, 기본 50. `top_tags`에는 임계값과 무관하게 상위 K개 포함 |

응답의 `results` 순서는 입력 순서와 같으며, 각 결과는 `filename`,
`width`, `height`, `tags`, `top_tags`, `preprocess_sec`, `inference_sec`,
`postprocess_sec`, `analysis_sec`를 포함합니다. 태그 항목은
`{"tag":"1girl","score":0.98}` 형태입니다.

요청당 최대 16개 이미지, 이미지 파일당 최대 20MiB, 최대 20백만 픽셀과
가로·세로 각각 최대 8192px를 허용합니다. 영상 파일은 받지 않습니다.
영상별 프레임 추출·태그 취합은 호출하는 앱에서 수행합니다.

RGB 이미지에 흰색 정사각 패딩을 넣고 448×448로 리사이즈·정규화합니다.
ONNX 그래프의 최종 연산을 확인해 sigmoid를 한 번만 적용합니다.
하나의 모델 세션과 API 워커를 사용하며 분석 요청은 순차 처리합니다.

## 설정

| 환경변수 | 기본값 |
| --- | --- |
| `ORT_INTRA_THREADS` | `4` |
| `ORT_INTER_THREADS` | `1` |
| `MODEL_DIR` | `/models` |
| `MODEL_OUTPUT_KIND` | `auto` |

모델 경로를 바꾸려면 `model.onnx`, `config.json`, `top_tags.txt`를 포함하는
폴더를 읽기 전용으로 마운트하고 `MODEL_DIR`을 지정합니다. 컨테이너는
UID/GID 10001로 실행하므로 마운트 파일은 해당 사용자에게 읽기 가능해야 합니다.

## 로컬 빌드와 검증

```sh
docker build -t joytag-server:local .
docker run -d --name joytag-test -p 127.0.0.1:18003:8000 joytag-server:local
docker cp tests/smoke.py joytag-test:/tmp/smoke.py
docker exec joytag-test python /tmp/smoke.py --url http://127.0.0.1:8000 --wait-seconds 180
docker rm -f joytag-test
```

다운로더 검증은 Python 표준 라이브러리만 사용합니다.

```sh
python -m unittest discover -s tests -p 'test_*.py' -v
```

모델은 Docker의 별도 빌드 단계에서 공식 Hugging Face 저장소의 고정 리비전으로
내려받고 SHA-256을 검증합니다. Docker 레이어 캐시를 재사용하며, 모델 파일은
Git과 로컬 빌드 컨텍스트에 포함하지 않습니다.

## GitHub Actions

`.github/workflows/docker.yml`은 다음 순서로 동작합니다.

1. 다운로더 테스트, Python 구문 검사, Docker 이미지 빌드
2. 이미지에 포함된 실제 모델로 상태 확인·다중 이미지 분석·오류 응답 검사
3. 검증 성공 후 GitHub Container Registry에 게시

`main` push와 수동 실행은 `latest` 및 `sha-<commit>` 태그를 게시합니다.
`v1.2.3` 같은 버전 태그를 push하면 `1.2.3`, `1.2`, `sha-<commit>` 태그를
게시합니다. `latest`는 `main` 빌드만 갱신합니다. Pull request는 검증만 수행합니다.
수동 실행은 `main` 또는 버전 태그에서만 이미지를 게시합니다.

게시 작업은 내장 `GITHUB_TOKEN`의 `packages: write` 권한을 사용합니다.
별도 registry secret은 필요하지 않습니다. 처음 생성되는 GHCR 패키지는 private일
수 있으므로 공개 다운로드가 필요하면 GitHub 패키지 설정에서 visibility를 확인하세요.

## 모델과 라이선스

- 공식 모델: [fancyfeast/joytag](https://huggingface.co/fancyfeast/joytag)
- 원 프로젝트: [fpgaminer/joytag](https://github.com/fpgaminer/joytag)
- 고정 리비전: `6b7f16331a6ccf0fdce37d5a9564715f6e772b22`
- 모델, config, 태그 목록의 SHA-256: `download_models.py`
- Apache License 2.0: `LICENSE`, 출처 표시: `NOTICE`
