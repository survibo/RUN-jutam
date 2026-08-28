# 정렬 Grokking 사용 설명서

이 문서는 설치, 데이터 생성, 학습, sweep, 체크포인트 재개와 그래프 생성 절차만 다룬다. 실험 설계의 배경과 지표 해석은 [`PRINCIPLES_KO.md`](PRINCIPLES_KO.md)를 참고한다.

## 1. 설치와 검사

RunPod에서는 PyTorch가 포함된 템플릿을 권장한다.

```bash
pip install -r requirements.txt
python -c "import torch; print(torch.__version__); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no CUDA')"

python sortdata.py --selftest
python sortformer.py --smoke
python sweep.py --selftest
python plot_grokking.py --selftest
```

`--smoke`는 작은 CPU 학습으로 설치와 입출력 경로를 검사한다. 실제 실험 결과로 사용하지 않는다.

## 2. 빠른 시작

### 데이터 생성

```bash
python sortdata.py \
  --n 50 --m 5 --modulus 5 \
  --train-count 128 \
  --split-strategy relation-complete \
  --seed 0 --n-test 20000 \
  --out data/n50_m5_tc128_rc
```

### 단일 과제 학습

```bash
python sortformer.py \
  --data data/n50_m5_tc128_rc \
  --task ascending \
  --output-constraint free \
  --steps 200000 \
  --log-csv runs/ascending_rc_free.csv \
  --out-dir runs/ascending_rc_free \
  --ckpt-every 10000
```

### 그래프 생성

```bash
python plot_grokking.py "runs/*.csv" \
  --out runs/grokking.png \
  --title "Sorting Grokking"
```

## 3. 데이터 생성

`sortdata.py`는 `train.txt`, `test.txt`, `metadata.json`을 생성한다. 현재 형식은 FORMAT v2이며 FORMAT v1은 지원하지 않는다. 예전 데이터는 다시 생성해야 한다.

각 데이터 행에는 하나의 입력과 세 과제의 target이 함께 저장된다.

```text
1 4 6 12 19 -> asc: 1 4 6 12 19 | mod: 1 6 12 4 19 | alt: 1 19 4 12 6
```

### 과제 이름

| `--task` | 출력 순서 |
| --- | --- |
| `ascending` | 값 오름차순 |
| `mod` | `(x % modulus, x)` 오름차순 |
| `alternating` | 최소, 최대, 두 번째 최소, 두 번째 최대, ... |

### 주요 옵션

| 옵션 | 동작 |
| --- | --- |
| `--n N` | 값 범위를 `0..N-1`로 설정 |
| `--m M` | 입력 조합의 원소 수 |
| `--modulus K` | `mod` 과제의 modulus |
| `--train-count K` | train을 정확히 K개 생성 |
| `--train-percent P` | 전체 조합의 P%를 train으로 생성 |
| `--split-strategy random` | seed 기반 무작위 split |
| `--split-strategy relation-complete` | 필요한 순서 관계를 덮는 train basis를 먼저 구성 |
| `--seed S` | split seed |
| `--n-test K` | test를 정확히 K개 생성 |
| `--out DIR` | 출력 디렉터리 |
| `--preview N` | 생성 후 N개 행 출력, 0이면 비활성화 |

`--train-count`와 `--train-percent`는 동시에 사용할 수 없다. 둘 다 생략하면 `sortdata.py`는 `--train-percent 30`을 사용한다. 재현 가능한 크기 비교에는 `--train-count`를 권장한다.

`--n-test -1`은 남은 조합이 안전한 크기이면 전부 사용하고, 아니면 최대 50,000개를 사용한다. 평가 비용을 고정하려면 `--n-test 20000`처럼 명시한다.

train과 test 합계는 최대 500,000행, 저장되는 integer cell은 최대 5,000,000개다. `relation-complete`의 최소 basis보다 작은 train 크기는 오류 메시지와 함께 거부된다.

### 생성 결과 확인

명령 출력과 `metadata.json`에서 다음 항목을 확인한다.

| 경로 | 확인할 값 |
| --- | --- |
| `config` | n, m, modulus, seed, sizing, split 전략 |
| `counts` | 전체 조합 수, train 수, test 수 |
| `coverage.elements` | train 원소 coverage와 빈도 |
| `coverage.pairs` | pair coverage 또는 `skipped` 상태 |
| `coverage.adjacent_pairs` | 과제별 인접 pair coverage |
| `coverage.order_identifiable` | 과제별 식별 가능 여부 |
| `coverage.test_strata` | 과제별 direct/transitive/unresolved 개수 |

pair 분석 규모가 내부 한도를 넘으면 `coverage.pairs.status`가 `skipped`가 된다. 데이터 생성 실패가 아니며 `reason`과 관련 상한이 함께 기록된다.

## 4. 학습

한 데이터 디렉터리를 여러 과제가 공유하도록 각 과제를 따로 실행한다.

```bash
python sortformer.py --data data/n50_m5_tc128_rc --task ascending \
  --output-constraint free --steps 200000 \
  --log-csv runs/ascending.csv --out-dir runs/ascending

python sortformer.py --data data/n50_m5_tc128_rc --task mod \
  --output-constraint free --steps 200000 \
  --log-csv runs/mod.csv --out-dir runs/mod

python sortformer.py --data data/n50_m5_tc128_rc --task alternating \
  --output-constraint free --steps 200000 \
  --log-csv runs/alternating.csv --out-dir runs/alternating
```

파일을 만들지 않고 generated data를 바로 사용할 수도 있다.

```bash
python sortformer.py \
  --n 50 --m 5 --modulus 5 \
  --train-count 128 --split-strategy relation-complete \
  --data-seed 0 --n-test 20000 \
  --task ascending --output-constraint free \
  --steps 200000 \
  --log-csv runs/generated.csv \
  --out-dir runs/generated
```

`--data`를 지정하면 해당 디렉터리의 metadata를 사용하며 generated-data sizing 옵션은 사용하지 않는다. generated-data mode에서 sizing을 생략하면 train count는 128이다.

### 출력 제약

| 값 | 허용되는 다음 출력 |
| --- | --- |
| `permutation` | 입력 중 아직 출력하지 않은 값 |
| `input-only` | 입력에 있는 값, 중복 허용 |
| `free` | 전체 vocabulary |

CLI 기본값은 `permutation`이다. 전체 생성 과정을 평가하는 기준 실행에는 `--output-constraint free`를 사용한다. `--label-smoothing`은 `free`에서만 사용할 수 있다.

### 주요 기본값

| 옵션 | 기본값 |
| --- | ---: |
| `--n-embd`, `--n-head` | 128, 4 |
| `--n-enc-layer`, `--n-layer` | 2, 2 |
| `--batch-size` | 512 |
| `--lr`, `--weight-decay` | `1e-3`, `1.0` |
| `--warmup`, `--lr-schedule` | 100, `constant` |
| `--steps` | 100000 |
| `--eval-every` | 250 |
| `--n-eval`, `--eval-batch` | 4096, 1024 |
| `--seed` | 42 |

`--batch-size -1`은 full-batch다. `--n-eval -1`은 train/test 전체를 평가한다. test 20,000개 전체의 strata 지표가 필요하면 `--n-eval 20000` 또는 `-1`을 사용한다.

## 5. RunPod GPU

A100, A40, RTX 4090 등 bf16 GPU의 예시는 다음과 같다.

```bash
python sortformer.py \
  --data data/n50_m5_tc128_rc \
  --task ascending --output-constraint free \
  --device auto --dtype bfloat16 --compile \
  --batch-size 2048 --eval-batch 4096 \
  --steps 200000 --eval-every 250 --n-eval 20000 \
  --log-csv runs/ascending_rc_free.csv \
  --out-dir runs/ascending_rc_free \
  --ckpt-every 10000
```

bf16 미지원 GPU는 `--dtype float16`을 사용한다. 메모리가 부족하면 `--batch-size`와 `--eval-batch`를 먼저 낮춘다. `--compile`은 첫 호출 비용이 있으므로 짧은 실행에서는 생략할 수 있다. `--device auto`는 `cuda`, `mps`, `cpu` 순으로 선택한다.

## 6. 체크포인트 재개

`--out-dir`을 지정하면 종료 시 `ckpt_final.pt`가 저장된다. `--ckpt-every K`를 지정하면 `ckpt_00010000.pt` 형식의 중간 체크포인트도 저장된다.

```bash
python sortformer.py \
  --data data/n50_m5_tc128_rc \
  --task ascending --output-constraint free \
  --device auto --dtype bfloat16 --compile \
  --batch-size 2048 --eval-batch 4096 \
  --steps 200000 \
  --resume runs/ascending_rc_free/ckpt_00010000.pt \
  --log-csv runs/ascending_rc_free.csv \
  --out-dir runs/ascending_rc_free \
  --ckpt-every 10000
```

`--steps`는 추가 실행 횟수가 아니라 최종 step이다. model config, dataset, task, constraint, seed가 기존 실행과 다르면 재개가 거부된다. CSV schema 또는 signature가 다르거나 CSV의 마지막 step이 체크포인트보다 뒤에 있어도 거부된다. 새 실험은 새 CSV와 출력 디렉터리를 사용한다.

## 7. Sweep

여러 train count, split, task, seed를 순차 실행한다.

```bash
python sweep.py \
  --n 50 --m 5 --modulus 5 --data-seed 0 --n-test 20000 \
  --train-counts 32 64 128 256 \
  --split-strategies relation-complete random \
  --tasks ascending mod alternating \
  --seeds 42 43 44 \
  --weight-decays 1.0 \
  --output-constraints free \
  --steps 200000 \
  --out-dir sweeps/n50_m5_counts \
  -- \
  --device auto --dtype bfloat16 \
  --batch-size 2048 --eval-batch 4096 \
  --eval-every 250 --n-eval 20000
```

고정 데이터셋 sweep:

```bash
python sweep.py \
  --data data/n50_m5_tc128_rc \
  --tasks ascending --seeds 42 43 44 \
  --weight-decays 0.1 1.0 3.0 \
  --output-constraints free permutation \
  --steps 200000 \
  --out-dir sweeps/fixed_rc \
  -- \
  --device auto --dtype bfloat16 \
  --batch-size 2048 --eval-batch 4096
```

`--train-counts`와 `--train-percents`는 동시에 사용할 수 없다. generated-data mode에서 둘 다 생략하면 count 128 하나를 사용한다. 고정 `--data` mode에서 sizing 또는 split 목록을 지정할 때는 각각 한 값만 허용된다.

`--` 뒤 옵션은 각 `sortformer.py` 실행에 전달된다. sweep이 관리하는 data, task, sizing, split, seed, weight decay, constraint, steps, log와 checkpoint 옵션은 뒤에서 다시 지정할 수 없다.

| 옵션 | 동작 |
| --- | --- |
| `--dry-run` | 학습 없이 명령과 manifest 생성 |
| `--skip-existing` | CSV와 최종 체크포인트가 모두 있는 run 생략 |
| `--continue-on-error` | 한 run 실패 후 다음 run 계속 |
| `--selftest` | 조합과 집계 자체 검사 |

출력 디렉터리에는 `sweep_manifest.json`, 각 run의 CSV/checkpoint, `sweep_summary.csv`가 생성된다. 부분 artifact만 존재하면 자동으로 덮어쓰지 않고 오류를 낸다.

## 8. 그래프

```bash
python plot_grokking.py "runs/*.csv" \
  --out runs/grokking.png \
  --title "Sorting Grokking"
```

그래프에는 exact accuracy, 생성 지표, strata exact accuracy, loss, parameter norm의 다섯 panel이 표시된다. 기본 x축은 log이며 `--linear-x`로 선형 축을 사용한다. 예전 schema의 CSV는 지원하지 않으므로 현재 코드로 다시 생성해야 한다.

## 9. 출력 파일 참고

### 학습 CSV 주요 열

| 열 | 내용 |
| --- | --- |
| `step`, `lr`, `weight_norm` | step, 학습률, parameter L2 norm |
| `train_loss`, `test_loss` | train/test loss |
| `*_token_acc` | teacher-forced token accuracy |
| `*_gen_in_set_token_acc` | 생성 token의 입력 집합 포함 비율 |
| `*_set_acc` | 생성 결과와 입력 집합 일치 비율 |
| `*_exact_acc` | 생성 순서 전체 일치 비율 |
| `test_direct_exact_acc` | direct stratum exact accuracy |
| `test_transitive_exact_acc` | transitive stratum exact accuracy |
| `test_unresolved_exact_acc` | unresolved stratum exact accuracy |
| `test_*_count` | 평가 subset의 각 stratum 개수 |
| `run_signature_sha256` | 실행 일관성 확인용 signature |

stratum count가 0이면 해당 accuracy cell은 비어 있다.

### Sweep summary 주요 열

| 열 | 내용 |
| --- | --- |
| `runs` | 집계된 run 수 |
| `successful_test90` | overall test exact 0.90 도달 run 수 |
| `successful_transitive90` | transitive exact 0.90 도달 run 수 |
| `median_grokking_gap` | seed별 grokking gap의 median |
| `median_final_*` | 최종 지표의 median |
| `csv_paths`, `checkpoint_paths` | 집계에 포함된 artifact 경로 |

지표의 의미와 권장 비교 순서는 [`PRINCIPLES_KO.md`](PRINCIPLES_KO.md)를 참고한다.

## 10. 자주 발생하는 오류

| 오류 상황 | 조치 |
| --- | --- |
| FORMAT v1 거부 | `sortdata.py`로 데이터 재생성 |
| relation-complete 최소 크기 오류 | 메시지에 표시된 최소 train count 이상 사용 |
| CUDA 메모리 부족 | batch size와 eval batch 축소 |
| checkpoint signature 불일치 | 동일 설정을 사용하거나 새 출력 경로에서 시작 |
| CSV schema 불일치 | 예전 CSV를 분리하고 새 CSV 사용 |
| sweep partial artifact 오류 | 기존 run 상태를 확인하고 다른 output directory 사용 |
