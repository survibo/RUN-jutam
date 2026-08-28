# 정렬 Grokking 실행 가이드

이 문서는 데이터 생성부터 세 정렬 과제 학습, RunPod GPU 실행, 체크포인트 재개, 학습 sweep과 그래프 생성까지의 사용법을 설명한다.


| 파일                                   | 역할                                     |
| ---------------------------------------- | ------------------------------------------ |
| [`sortdata.py`](sortdata.py)           | 조합 split, 정답, coverage metadata 생성 |
| [`sortformer.py`](sortformer.py)       | Transformer 학습, 평가, 체크포인트       |
| [`sweep.py`](sweep.py)                 | 여러 seed와 설정의 순차 실행 및 집계     |
| [`plot_grokking.py`](plot_grokking.py) | CSV 곡선과 grokking gap 시각화           |

## 1. 설치

RunPod에서는 PyTorch가 포함된 템플릿을 사용하는 것을 권장한다.

```bash
pip install -r requirements.txt
```

GPU 인식 여부를 확인한다.

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

## 2. 동작 점검

본격적인 실험 전에 다음 자체 검사를 실행한다.

```bash
python sortdata.py --selftest
python sortformer.py --smoke
python sweep.py --selftest
python plot_grokking.py --selftest
```

`--smoke`는 작은 데이터로 학습, 평가, 자유 생성, CSV 기록 경로를 확인한다. 설치 검사용이므로 해당 결과를 grokking 실험 결과로 사용하지 않는다.

## 3. 데이터 생성

다음 예시는 `[0, 50)`에서 서로 다른 5개 값을 선택하는 모든 조합 중 2%를 train으로 사용한다.

```bash
python sortdata.py \
  --n 0050 \
  --m 5 \
  --train-percent 2 \
  --modulus 7 \
  --seed 0 \
  --n-test 5000 \
  --out data/m50_k5
```

생성 파일은 다음과 같다.

```text
data/m50_k5/train.txt
data/m50_k5/test.txt
data/m50_k5/metadata.json
```

각 데이터 행에는 하나의 입력 집합과 세 과제의 정답이 함께 저장된다.

```text
1 4 6 12 19 -> asc: ... | mod: ... | alt: ...
```

### 정렬 과제


| task          | 규칙                                        |
| --------------- | --------------------------------------------- |
| `ascending`   | 일반적인 값 오름차순                        |
| `mod`         | `(x % modulus, x)` 기준 오름차순            |
| `alternating` | 최소, 최대, 두 번째 최소, 두 번째 최대, ... |

### 데이터 옵션


| 옵션                |    기본값 | 의미                                  |
| --------------------- | ----------: | --------------------------------------- |
| `--n`               |        20 | 정수 범위`[0, n)`                     |
| `--m`               |         4 | 한 입력 집합의 원소 개수              |
| `--train-percent`   |        30 | 전체`C(n,m)` 중 train 비율            |
| `--modulus`         |         3 | mod 정렬에서 사용하는 값              |
| `--seed`            |         0 | train/test 분할 seed                  |
| `--n-test`          |        -1 | test 개수,`-1`이면 가능한 나머지 전체 |
| `--enumerate-limit` | 5,000,000 | 전체 열거와 rank sampling의 전환 기준 |
| `--out`             |      없음 | 데이터 저장 디렉터리                  |
| `--preview`         |         5 | 출력할 미리보기 행 수                 |
| `--selftest`        |      꺼짐 | 데이터 생성기 자체 검사               |

train 크기는 다음 식으로 정확히 결정된다.

```text
round(C(n,m) * train_percent / 100)
```

조합 공간이 큰 상태에서 `--n-test -1`을 사용하면 test는 최대 50,000개로 제한된다. 동일한 옵션과 seed를 사용하면 동일한 split이 생성된다.

### Coverage metadata

`metadata.json`의 `coverage.elements`에는 train 원소의 정확한 coverage와 등장 빈도 `frequency_min`, `frequency_max`, `frequency_mean`, `frequency_std`가 항상 기록된다.

pair 분석은 다음 값이 안전 한도 1,000,000 이하일 때만 정확히 계산한다.

```text
min(C(n,2), train_count * C(m,2)) + min(C(n,2), test_count * C(m,2))
```

각 항은 해당 split에서 추적할 수 있는 unique pair 수의 상한이며, train과 test 양쪽을 합산한다. `--n-test`를 크게 잡으면 두 번째 항 때문에 한도를 넘을 수 있다.

이때 `coverage.pairs`에는 train pair coverage와 같은 빈도 요약 외에 다음 값이 기록된다.

- test의 전체 pair occurrence와 그중 train에서 본 occurrence 수 및 비율
- test의 unique pair 수와 그중 train에서 본 unique pair 수 및 비율

한도를 넘으면 `coverage.pairs.status`는 `skipped`다. 이 경우 `reason`, `possible`, `potential_train_occurrences`, `potential_test_occurrences`를 기록하며 pair coverage나 test 비율을 추정하지 않는다.

## 4. 세 과제 학습

비교 실험에서는 한 번 생성한 데이터 디렉터리를 세 과제가 공유해야 한다.

### 오름차순

```bash
python sortformer.py \
  --data data/m50_k5 \
  --task ascending \
  --steps 200000 \
  --log-csv runs/ascending.csv \
  --out-dir runs/ascending \
  --ckpt-every 10000
```

### mod 기반 오름차순

```bash
python sortformer.py \
  --data data/m50_k5 \
  --task mod \
  --steps 200000 \
  --log-csv runs/mod.csv \
  --out-dir runs/mod \
  --ckpt-every 10000
```

mod 정렬의 정확한 키는 다음과 같다.

```python
key=lambda x: (x % modulus, x)
```

### 최소·최대 교대 정렬

```bash
python sortformer.py \
  --data data/m50_k5 \
  --task alternating \
  --steps 200000 \
  --log-csv runs/alternating.csv \
  --out-dir runs/alternating \
  --ckpt-every 10000
```

## 5. RunPod GPU 권장 명령

A100, A40, RTX 4090 등 bf16 지원 GPU에서는 다음 설정부터 시작한다.

```bash
python sortformer.py \
  --data data/m50_k5 \
  --task ascending \
  --device auto \
  --dtype bfloat16 \
  --compile \
  --batch-size 2048 \
  --eval-batch 4096 \
  --steps 200000 \
  --eval-every 500 \
  --log-csv runs/ascending.csv \
  --out-dir runs/ascending \
  --ckpt-every 10000
```

bf16을 지원하지 않는 GPU에서는 다음 옵션을 사용한다.

```bash
--dtype float16
```

GPU 메모리가 부족하면 먼저 학습 및 평가 배치를 낮춘다.

```bash
--batch-size 512 --eval-batch 1024
```

`--compile`은 첫 호출 때 컴파일 시간이 필요하다. 짧은 시험 실행에서는 빼는 편이 빠를 수 있다.

## 6. 모델 옵션


| 옵션                  |      기본값 | 의미                                |
| ----------------------- | ------------: | ------------------------------------- |
| `--task`              |   ascending | `ascending`, `mod`, `alternating`   |
| `--n-embd`            |         128 | embedding 및 hidden 차원            |
| `--n-head`            |           4 | attention head 수                   |
| `--n-enc-layer`       |           2 | 순열 불변 집합 인코더 층 수         |
| `--n-layer`           |           2 | 자기회귀 디코더 층 수               |
| `--dropout`           |           0 | dropout 비율                        |
| `--tie`               |        꺼짐 | 출력 head와 입력 embedding 공유     |
| `--init-std`          |        0.02 | 초기 파라미터 표준편차              |
| `--init-scale`        |         1.0 | 초기 파라미터 전체 배율             |
| `--output-constraint` | permutation | `permutation`, `input-only`, `free` |

### 출력 제약

`--output-constraint`의 세 모드는 다음과 같다.


| 값            | 생성 가능한 값                                                  |
| --------------- | ----------------------------------------------------------------- |
| `permutation` | 입력에 있고 아직 출력하지 않은 값. 결과는 항상 입력의 순열이다. |
| `input-only`  | 입력에 있는 값. 같은 값을 다시 출력할 수 있다.                  |
| `free`        | 전체 vocabulary의 값. 입력 밖의 값도 출력할 수 있다.            |

기본값 `permutation`은 모델이 순서 규칙에 집중하게 하지만 과제를 쉽게 만들어 grokking 구간을 줄일 수 있다. `input-only`와 `free`를 대조군으로 함께 비교하는 것이 좋다.

```bash
python sortformer.py \
  --data data/m50_k5 \
  --task ascending \
  --output-constraint free \
  --steps 200000 \
  --log-csv runs/ascending_free.csv
```

## 7. 학습 옵션


| 옵션                |   기본값 | 의미                                    |
| --------------------- | ---------: | ----------------------------------------- |
| `--steps`           |   100000 | optimizer update 횟수                   |
| `--batch-size`      |      512 | optimizer step당 학습 표본 수           |
| `--batch-size -1`   |        - | 전체 train 데이터를 사용하는 full-batch |
| `--lr`              |     1e-3 | 최대 학습률                             |
| `--beta1`           |      0.9 | AdamW beta1                             |
| `--beta2`           |     0.98 | AdamW beta2                             |
| `--weight-decay`    |      1.0 | AdamW weight decay                      |
| `--grad-clip`       |      1.0 | gradient clipping                       |
| `--warmup`          |      100 | linear warmup step 수                   |
| `--lr-schedule`     | constant | `constant`, `cosine`, `linear`          |
| `--label-smoothing` |        0 | label smoothing 값                      |

고전적인 grokking 설정은 다음과 같이 시작할 수 있다.

```bash
--batch-size -1 \
--lr 0.001 \
--weight-decay 1.0 \
--lr-schedule constant \
--dropout 0
```

label smoothing은 `free` 모드에서만 사용할 수 있다.

```bash
--output-constraint free --label-smoothing 0.1
```

## 8. 평가와 로그


| 옵션           | 기본값 | 의미                             |
| ---------------- | -------: | ---------------------------------- |
| `--eval-every` |    250 | 평가 간격                        |
| `--n-eval`     |   4096 | 각 split에서 평가할 최대 예제 수 |
| `--n-eval -1`  |      - | train/test 전체 평가             |
| `--eval-batch` |   1024 | 평가 배치 크기                   |
| `--log-csv`    |   없음 | CSV 로그 경로                    |
| `--seed`       |     42 | 모델 초기화 및 학습 seed         |

CSV의 주요 지표는 다음과 같다.


| 지표                   | 의미                                                      |
| ------------------------ | ----------------------------------------------------------- |
| `loss`                 | teacher forcing cross-entropy loss                        |
| `token_acc`            | teacher forcing 다음 token 정확도                         |
| `gen_in_set_token_acc` | 자유 생성 위치 중 token이 입력 집합에 포함된 비율         |
| `set_acc`              | 자유 생성 전체가 순서를 무시했을 때 입력 집합과 같은 비율 |
| `exact_acc`            | 자유 생성 전체가 target 순서까지 같은 비율                |
| `weight_norm`          | 전체 파라미터 L2 norm                                     |

CSV 열 이름은 split prefix를 붙인 `train_gen_in_set_token_acc`, `test_gen_in_set_token_acc`, `train_set_acc`, `test_set_acc`, `train_exact_acc`, `test_exact_acc`다. 생성 지표 세 개를 함께 보면 입력 원소 사용, 중복/누락 없는 집합 완성, 올바른 정렬 순서를 차례로 구분할 수 있다. `permutation`에서는 in-set과 set 지표가 구조적으로 1이고, `input-only`에서는 in-set만 구조적으로 1이므로 constraint 간 비교에서 이 점을 고려한다.

teacher-forced `loss`와 `token_acc`도 각 constraint의 후보 공간에서 계산된다. 특히 `permutation`의 마지막 위치는 남은 값이 하나라 구조적으로 정답이다. 따라서 이 두 값은 서로 다른 constraint의 난이도를 직접 비교하기보다 동일 constraint 안에서 학습 추세를 보는 용도로 사용한다.

CSV의 `run_signature_sha256`은 resume 시 task, output constraint, 데이터셋이 같은 실행인지 검증하는 식별자다.

실제 grokking 판정에서는 `train_exact_acc`와 `test_exact_acc`를 사용한다. 전이 시점을 정확히 측정하려면 다음 옵션으로 split 전체를 평가한다.

```bash
--n-eval -1
```

test가 매우 크면 전체 자유 생성 평가 시간이 길어질 수 있다.

## 9. 체크포인트 재개

다음 옵션으로 중간 체크포인트를 저장한다.

```bash
--out-dir runs/ascending --ckpt-every 10000
```

예상 파일:

```text
runs/ascending/ckpt_00010000.pt
runs/ascending/ckpt_00020000.pt
runs/ascending/ckpt_final.pt
```

10,000 step 체크포인트에서 최종 200,000 step까지 재개하는 예시는 다음과 같다.

```bash
python sortformer.py \
  --data data/m50_k5 \
  --task ascending \
  --device auto \
  --dtype bfloat16 \
  --compile \
  --batch-size 2048 \
  --eval-batch 4096 \
  --steps 200000 \
  --resume runs/ascending/ckpt_00010000.pt \
  --log-csv runs/ascending.csv \
  --out-dir runs/ascending \
  --ckpt-every 10000
```

`--steps`는 추가 학습량이 아니라 최종 도달 step이다. 체크포인트에는 모델, optimizer, AMP scaler, 난수 상태, task, 데이터셋 fingerprint가 저장된다. 현재 task 또는 데이터셋이 다르면 실행을 거부한다.

CSV가 체크포인트보다 뒤 step까지 기록돼 있으면 서로 다른 학습 trajectory가 섞이지 않도록 새 CSV를 사용한다.

```bash
--log-csv runs/ascending_resumed.csv
```

## 10. 그래프 생성

단일 결과:

```bash
python plot_grokking.py runs/ascending.csv \
  --out runs/ascending.png
```

세 과제 비교:

```bash
python plot_grokking.py \
  runs/ascending.csv \
  runs/mod.csv \
  runs/alternating.csv \
  --out runs/all_tasks.png \
  --title "Sorting Grokking"
```

glob 패턴도 사용할 수 있다.

```bash
python plot_grokking.py "runs/*.csv" --out runs/grokking.png
```

그래프는 네 panel로 train/test exact accuracy, test 생성 분해(in-set token/set/exact), loss, parameter L2 norm을 표시한다. 기본 x축은 로그 스케일이다. 선형 x축은 다음 옵션을 사용한다.

```bash
--linear-x
```

요약표의 `gap x`는 다음 비율이다.

```text
test exact >= 0.90 최초 step / train exact >= 0.99 최초 step
```

- `gap x`가 1에 가까우면 암기와 일반화가 거의 동시에 일어난다.
- `gap x`가 10보다 크면 긴 암기 구간 이후 일반화가 나타난 것이다.
- test 임계점이 없으면 학습 기간 내에 grokking이 관측되지 않은 것이다.

한 CSV에는 하나의 task만 기록해야 한다. 과제를 바꾸면 별도 CSV를 사용한다.

## 11. Sweep 실행

### 고정 데이터에서 여러 seed, weight decay, 출력 제약 비교

```bash
python sweep.py \
  --data data/m50_k5 \
  --tasks ascending \
  --seeds 42 43 44 \
  --weight-decays 0.1 1.0 3.0 \
  --output-constraints permutation input-only free \
  --steps 200000 \
  --out-dir sweeps/fixed_data \
  -- \
  --device auto \
  --dtype bfloat16 \
  --batch-size 2048 \
  --eval-batch 4096
```

`--data`를 사용하면 모든 run이 같은 split을 공유한다. 이 모드에서 `--train-percents`는 sweep하지 않으며, 지정한다면 값 하나만 허용된다.

### 생성 데이터의 train 비율 비교

```bash
python sweep.py \
  --n 20 --m 4 --modulus 5 --data-seed 0 --n-test -1 \
  --train-percents 0.5 1 2 5 \
  --tasks ascending \
  --seeds 42 43 44 \
  --weight-decays 1.0 \
  --output-constraints permutation free \
  --steps 100000 \
  --out-dir sweeps/train_percent \
  -- \
  --batch-size -1 --eval-every 250 --n-eval -1 --device auto
```

`--` 뒤의 옵션은 모든 `sortformer.py` 실행에 전달된다. sweep이 직접 관리하는 데이터, task, seed, weight decay, 출력 제약, step, 로그 및 체크포인트 경로는 뒤에서 다시 지정할 수 없다.

출력 디렉터리에는 run별 CSV와 체크포인트 디렉터리 외에 다음 파일이 생긴다.

```text
sweep_manifest.json
sweep_summary.csv
```

manifest는 각 명령, 설정, 상태와 artifact 경로를 기록한다. summary는 완료된 run을 task, 데이터 또는 train 비율, weight decay, 출력 제약별로 묶어 test 90% 도달 수, grokking gap, 최종 exact accuracy를 seed에 걸쳐 집계한다.


| 옵션                  | 동작                                                                            |
| ----------------------- | --------------------------------------------------------------------------------- |
| `--dry-run`           | 학습하지 않고 생성될 명령과 manifest를 확인한다.                                |
| `--skip-existing`     | CSV와`ckpt_final.pt`가 모두 있는 완전한 run을 건너뛴다. 부분 artifact는 오류다. |
| `--continue-on-error` | 한 run이 실패해도 나머지 run을 계속한다.                                        |
| `--selftest`          | run 조합, 전달 인자, summary 집계를 자체 검사한다.                              |

## 12. 첫 실험 추천

작은 기준 실험은 다음과 같이 시작한다.

```bash
python sortdata.py \
  --n 20 \
  --m 4 \
  --train-percent 2 \
  --modulus 5 \
  --seed 0 \
  --out data/baseline

python sortformer.py \
  --data data/baseline \
  --task ascending \
  --batch-size -1 \
  --steps 100000 \
  --eval-every 250 \
  --n-eval -1 \
  --weight-decay 1.0 \
  --log-csv runs/baseline.csv \
  --out-dir runs/baseline
```

전이가 너무 빠르면 `--train-percent`를 낮추거나 `--output-constraint free`를 사용한다. train은 암기하지만 test가 오르지 않으면 train 비율이나 총 step을 늘린다. 유망한 설정을 찾은 뒤 seed를 최소 3개 사용해 우연한 전이와 구분하는 것이 좋다.
