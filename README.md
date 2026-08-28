# 정렬에서의 Grokking

정수 집합을 입력받아 정해진 순열을 자기회귀적으로 생성하는 PyTorch 실험 코드다.
Karpathy의 [`microgpt_org.py`](microgpt_org.py)에서 사용한 pre-norm, RMSNorm, bias 없는
attention/MLP 구조를 바탕으로 집합 인코더와 causal 디코더를 구성했다.

## 과제

`[0, n)`에서 중복 없이 뽑은 `m`개 값에 대해 세 과제를 독립적으로 학습한다.

| task | 규칙 | 예시 (`p=4`) |
| --- | --- | --- |
| `ascending` | 값 오름차순 | `0 3 7 9 -> 0 3 7 9` |
| `mod` | `(x % p, x)` 오름차순 | `0 3 7 9 -> 0 9 3 7` |
| `alternating` | 최소, 최대, 두 번째 최소, 두 번째 최대, ... | `0 3 7 9 -> 0 9 3 7` |

집합 인코더에는 위치 임베딩이 없어서 입력 순서에 불변이다. 디코더는 이전 출력을 causal
self-attention으로 보고, 매 단계 입력 집합 전체를 cross-attention으로 참조한다. 학습 데이터에서
유사 예제를 검색하는 retrieval은 사용하지 않는다. 그런 방식은 홀드아웃 정보 누출과 암기 측정을
혼동시키기 쉽다.

## 파일

| 파일 | 역할 |
| --- | --- |
| [`sortdata.py`](sortdata.py) | `C(n,m)` 조합 분할 및 세 과제 정답 생성 |
| [`sortformer.py`](sortformer.py) | 자기회귀 Transformer 학습, 평가, 체크포인트 |
| [`sweep.py`](sweep.py) | seed, weight decay, 출력 제약, train 비율 순차 sweep |
| [`plot_grokking.py`](plot_grokking.py) | CSV 곡선 시각화와 grokking gap 계산 |
| [`microgpt_org.py`](microgpt_org.py) | 원본 microGPT 참고 코드 |

## 설치

```bash
pip install -r requirements.txt
```

CUDA가 설치된 RunPod PyTorch 이미지에서는 별도 CUDA 코드를 설정할 필요가 없다. 설치된 PyTorch가
GPU를 인식하는지는 다음 명령으로 확인한다.

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## 빠른 검증

```bash
python sortdata.py --selftest
python sortformer.py --smoke
python sweep.py --selftest
python plot_grokking.py --selftest
```

`--smoke`는 CPU에서 작은 데이터로 전체 학습, 자유 생성, CSV 기록 경로를 검사한다. 정확한
benchmark가 아니라 설치와 코드 경로를 확인하는 용도다.

## 데이터 생성

```bash
python sortdata.py --n 50 --m 5 --train-percent 2 \
  --modulus 7 --seed 0 --n-test 5000 --out data/m50_k5
```

출력 파일은 `train.txt`, `test.txt`, `metadata.json`이다. 각 텍스트 행은 세 target을 함께 담는다.

```text
1 4 6 12 19 -> asc: 1 4 6 12 19 | mod: 1 19 4 12 6 | alt: 1 19 4 12 6
```

- 분할 단위는 입력 순서가 아니라 조합 identity다.
- train 크기는 `round(C(n,m) * train_percent / 100)`로 정확히 정한다.
- 동일 seed는 동일 파일을 생성한다.
- 작은 공간은 rank 전체를 섞고, 큰 공간은 조합을 열거하지 않고 rank를 직접 샘플링한다.
- `--n-test -1`은 가능한 경우 나머지 전체를 test로 사용하고, 큰 공간에서는 최대 50,000개를 쓴다.
- 안전을 위해 한 번에 materialize하는 행은 최대 5,000,000개, 정수 cell은 최대 50,000,000개다.

`metadata.json`의 `coverage`에는 train에서 각 원소가 얼마나 등장했는지 정확한 coverage와 빈도
요약이 항상 기록된다. train과 test에서 추적할 pair ID의 보수적 합계가 1,000,000개 이하일 때는 train pair의
coverage 및 빈도 최소/최대/평균/표준편차와, test의 pair occurrence 및 unique pair 중 train에서
본 비율도 기록한다. 한도를 넘으면 `pairs.status`가 `skipped`가 되고 이유, 가능한 pair 수,
예상 train/test pair occurrence를 기록하며 근삿값을 만들지 않는다.

모델에 `--data`를 생략하면 같은 옵션으로 데이터를 메모리에서 직접 만들 수도 있다. 여러 과제를
비교할 때는 txt를 한 번 생성해 같은 split을 공유하는 편이 안전하다.

## 세 과제 학습

```bash
python sortformer.py --data data/m50_k5 --task ascending \
  --steps 200000 --log-csv runs/ascending.csv --out-dir runs/ascending

python sortformer.py --data data/m50_k5 --task mod \
  --steps 200000 --log-csv runs/mod.csv --out-dir runs/mod

python sortformer.py --data data/m50_k5 --task alternating \
  --steps 200000 --log-csv runs/alternating.csv --out-dir runs/alternating
```

`--output-constraint`는 생성 후보를 다음 세 방식으로 정한다.

| 값 | 동작 |
| --- | --- |
| `permutation` | 입력 원소 중 아직 생성하지 않은 값만 허용한다. 출력은 항상 입력의 순열이다. |
| `input-only` | 입력 원소만 허용하되 중복 생성을 허용한다. |
| `free` | 전체 vocabulary를 허용한다. |

기본값 `permutation`은 모델이 순서 규칙에 집중하게 하지만 grokking 구간을 줄일 수 있다. 알고리즘
전체가 스스로 출현하는지를 보려면 `input-only`와 `free`를 대조군으로 실행한다.

```bash
python sortformer.py --data data/m50_k5 --task ascending \
  --output-constraint free --steps 200000 --log-csv runs/ascending_free.csv
```

### 주요 옵션

| 옵션 | 기본값 | 설명 |
| --- | ---: | --- |
| `--n-embd` | 128 | embedding 차원 |
| `--n-head` | 4 | attention head 수 |
| `--n-enc-layer` | 2 | 양방향 집합 인코더 층 수 |
| `--n-layer` | 2 | causal 디코더 층 수 |
| `--output-constraint` | permutation | `permutation`, `input-only`, `free` |
| `--batch-size` | 512 | optimizer step당 표본 수, `-1`은 full-batch |
| `--lr` | 1e-3 | AdamW 학습률 |
| `--weight-decay` | 1.0 | grokking 실험용 강한 weight decay |
| `--warmup` | 100 | linear warmup step 수 |
| `--lr-schedule` | constant | `constant`, `cosine`, `linear` |
| `--eval-every` | 250 | train/test 평가 간격 |
| `--n-eval` | 4096 | 주기적 평가 표본 수, `-1`은 전체 |
| `--ckpt-every` | 0 | 중간 체크포인트 간격, `0`은 마지막만 |
| `--resume` | - | 체크포인트에서 재개 |

## RunPod / GPU

장치를 지정하지 않으면 `cuda`, `mps`, `cpu` 순으로 자동 선택한다. CUDA에서는 기본적으로 bf16
autocast를 사용한다. 구형 GPU가 bf16을 지원하지 않으면 `--dtype float16`을 사용한다.

```bash
python sortformer.py --data data/m50_k5 --task ascending \
  --device auto --dtype bfloat16 --compile \
  --batch-size 2048 --eval-batch 4096 --steps 200000 \
  --log-csv runs/ascending.csv --out-dir runs/ascending --ckpt-every 10000
```

중단 후 같은 데이터와 모델 옵션으로 재개한다.

```bash
python sortformer.py --data data/m50_k5 --task ascending \
  --device auto --dtype bfloat16 --compile --batch-size 2048 --eval-batch 4096 \
  --resume runs/ascending/ckpt_00010000.pt --steps 200000 \
  --log-csv runs/ascending.csv --out-dir runs/ascending
```

체크포인트는 task와 데이터셋 fingerprint를 검증하고 optimizer, AMP scaler, 난수 상태를 함께
복원한다. 재개 지점보다 뒤 step이 이미 CSV에 있으면 서로 다른 trajectory가 섞이지 않도록 실행을
거부하므로, 그 경우 새 CSV 경로를 지정한다.

GPU 메모리가 부족하면 우선 `--batch-size`와 `--eval-batch`를 낮춘다. `--compile`의 첫 호출은
컴파일 때문에 느리며, 작은 실험에서는 오히려 손해일 수 있다.

## Sweep

이미 생성한 split을 고정하고 여러 model seed, weight decay, 출력 제약을 비교한다.

```bash
python sweep.py --data data/m50_k5 --tasks ascending \
  --seeds 42 43 44 --weight-decays 0.1 1.0 3.0 \
  --output-constraints permutation input-only free --steps 200000 \
  --out-dir sweeps/fixed_data -- \
  --device auto --dtype bfloat16 --batch-size 2048 --eval-batch 4096
```

데이터를 각 run에서 같은 `data-seed`로 생성하면서 train 비율을 sweep할 수도 있다.

```bash
python sweep.py --n 20 --m 4 --modulus 5 --data-seed 0 --n-test -1 \
  --train-percents 0.5 1 2 5 --tasks ascending --seeds 42 43 44 \
  --weight-decays 1.0 --output-constraints permutation free --steps 100000 \
  --out-dir sweeps/train_percent -- \
  --batch-size -1 --eval-every 250 --n-eval -1 --device auto
```

`--` 뒤 인자는 각 `sortformer.py` 실행에 전달된다. sweep이 관리하는 데이터, task, seed,
weight decay, 출력 제약, step 및 출력 경로 옵션은 뒤에 다시 지정할 수 없다. 실행 상태와 명령은
`sweep_manifest.json`, 완료 run의 exact-accuracy 요약은 `sweep_summary.csv`에 기록된다.
`--dry-run`은 명령만 확인하고, `--skip-existing`은 CSV와 최종 체크포인트가 모두 있는 run만 건너뛴다.
`--continue-on-error`는 실패 후 다음 run을 계속하며, `--selftest`는 sweep 구성과 집계를 검사한다.

## 로그와 그래프

CSV에는 다음 값이 기록된다.

```text
step,lr,weight_norm,
train_loss,train_token_acc,train_gen_in_set_token_acc,train_set_acc,train_exact_acc,
test_loss,test_token_acc,test_gen_in_set_token_acc,test_set_acc,test_exact_acc,
elapsed_seconds,task,output_constraint,run_signature_sha256,train_eval_count,test_eval_count
```

- `token_acc`: teacher forcing에서 다음 token 정확도
- `gen_in_set_token_acc`: 자유 생성 위치 중 출력 token이 입력 집합에 포함된 비율
- `set_acc`: 자유 생성 전체가 순서를 무시했을 때 입력 집합과 정확히 같은 비율
- `exact_acc`: 자유 생성 전체가 target 순서까지 정확히 같은 비율
- `weight_norm`: 전체 파라미터 L2 norm
- `train_eval_count`, `test_eval_count`: 해당 행의 평가에 실제 사용한 예제 수
- `run_signature_sha256`: resume 시 task, constraint, 데이터셋이 같은 run인지 확인하는 식별자

세 생성 지표는 "입력 원소 사용 -> 중복/누락 없는 집합 완성 -> 올바른 순서"를 분리한다.
`permutation`에서는 앞의 두 지표가 구조적으로 1이고, `input-only`에서는 in-set만 구조적으로 1이다.
실제 정렬 일반화 성능은 `test_exact_acc`를 기준으로 본다.
teacher-forced loss와 `token_acc`는 constraint가 적용된 후보 공간에서 계산된다. 특히
`permutation`의 마지막 위치는 정답 하나만 남아 구조적으로 맞으므로 constraint 간 절대값 비교보다
동일 constraint 안에서의 학습 추세 확인에 사용한다.

```bash
python plot_grokking.py "runs/*.csv" --out runs/grokking.png \
  --title "Set sorting grokking"
```

그래프의 네 panel은 train/test exact accuracy, test 생성 분해(in-set token/set/exact), loss,
parameter norm이다. 기본 x축은 로그 스케일이다. 요약표의 `gap x`는
`test exact >= 0.90` 최초 step을 `train exact >= 0.99` 최초 step으로 나눈 값이다.
1에 가까우면 즉시 일반화이고, 값이 클수록 긴 암기 구간 뒤 일반화가 나타난 것이다.

## 실험 권장 순서

1. `n=20, m=4`에서 `train-percent`를 `0.5, 1, 2, 5`로 탐색한다.
2. 전이가 너무 빠르면 `n` 또는 `m`을 키우거나 `--output-constraint free` 대조군을 사용한다.
3. train은 암기하지만 test가 오르지 않으면 100,000~500,000 step까지 늘린다.
4. `weight-decay`를 `0.1, 1.0, 3.0`으로 비교한다.
5. 후보 설정을 찾은 뒤 seed를 최소 3개 사용해 우연한 전이와 구분한다.

Grokking은 특정 optimizer나 weight decay만으로 보장되지 않는다. train 분할 크기, 모델 크기,
초기화, 출력 제약이 암기 해와 일반화 해의 상대적 난이도를 바꾸므로 한 번에 하나씩 통제해서
비교해야 한다.
