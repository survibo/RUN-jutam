# 정렬에서의 Grokking

정수 집합을 입력받아 정해진 순열을 자기회귀적으로 생성하는 PyTorch 실험 코드다. Karpathy의 [`microgpt_org.py`](microgpt_org.py)에서 사용한 pre-norm, RMSNorm, bias 없는 attention/MLP 구조를 바탕으로 집합 인코더와 causal 디코더를 구성했다.

## 과제와 모델

`[0, n)`에서 중복 없이 뽑은 `m`개 값에 대해 세 과제를 독립적으로 학습한다.


| task          | 규칙                                        | 예시 (`modulus=4`)   |
| --------------- | --------------------------------------------- | ---------------------- |
| `ascending`   | 값 오름차순                                 | `0 3 7 9 -> 0 3 7 9` |
| `mod`         | `(x % modulus, x)` 오름차순                 | `0 3 7 9 -> 0 9 3 7` |
| `alternating` | 최소, 최대, 두 번째 최소, 두 번째 최대, ... | `0 3 7 9 -> 0 9 3 7` |

값은 연속적인 수치가 아니라 학습되는 categorical embedding이다. 인코더에는 위치 임베딩이 없고 학습 때 각 입력 행도 섞으므로 입력 순서에 불변이다. 디코더에는 이전 출력 token embedding, BOS, 출력 위치 embedding이 있으며, causal self-attention과 인코더 전체에 대한 cross-attention을 사용한다. `--tie`를 지정하면 출력 head와 인코더 값 embedding을 공유한다.

따라서 모델은 숫자의 크기나 나머지 연산을 입력 좌표에서 직접 받지 않는다. train에 어떤 값 쌍의 관계가 제시됐는지, 그 관계들로 전체 순서를 식별할 수 있는지가 핵심 실험 조건이다. retrieval이나 홀드아웃 검색은 사용하지 않는다.

## 파일


| 파일                                   | 역할                                                 |
| ---------------------------------------- | ------------------------------------------------------ |
| [`sortdata.py`](sortdata.py)           | FORMAT v2 조합 split, 세 target, 관계 coverage 생성  |
| [`sortformer.py`](sortformer.py)       | 자기회귀 Transformer 학습, strata 평가, 체크포인트   |
| [`sweep.py`](sweep.py)                 | train count, split 전략, seed와 하이퍼파라미터 sweep |
| [`plot_grokking.py`](plot_grokking.py) | CSV의 다섯 panel 시각화와 grokking gap 계산          |
| [`microgpt_org.py`](microgpt_org.py)   | 원본 microGPT 참고 코드                              |
| [`USAGE_KO.md`](USAGE_KO.md)           | 설치, 실행 옵션과 운영 절차                          |
| [`PRINCIPLES_KO.md`](PRINCIPLES_KO.md) | 실험 설계 원리와 결과 해석                           |

## 설치와 검증

```bash
pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

python sortdata.py --selftest
python sortformer.py --smoke
python sweep.py --selftest
python plot_grokking.py --selftest
```

`--smoke`는 CPU에서 작은 데이터로 학습, 평가, 자유 생성, CSV 기록 경로를 검사한다. benchmark가 아니라 설치 검사용이다.

## FORMAT v2 데이터

권장 기준 데이터는 정확한 train 개수와 관계 완전 split으로 만든다.

```bash
python sortdata.py --n 50 --m 5 --modulus 5 \
  --train-count 128 --split-strategy relation-complete \
  --seed 0 --n-test 20000 --out data/n50_m5_tc128_rc
```

`train.txt`, `test.txt`, `metadata.json`이 생성된다. 각 텍스트 파일의 첫 줄에는 `format_version=2`, sizing mode, split 전략을 포함한 JSON header가 있고, 각 행에는 같은 입력의 세 target이 있다.

```text
1 4 6 12 19 -> asc: 1 4 6 12 19 | mod: 1 6 12 4 19 | alt: 1 19 4 12 6
```

FORMAT v1 데이터는 현재 loader가 지원하지 않는다. 기존 데이터 디렉터리는 `sortdata.py`로 다시 생성해야 한다.

### 크기와 split

`--train-count`와 `--train-percent`는 상호 배타적이다.

- `sortdata.py`: 둘 다 생략하면 `--train-percent 30`이다.
- `sortformer.py`: `--data` 없이 둘 다 생략하면 `--train-count 128`이다.
- `sweep.py`: generated-data mode에서 `--train-counts`와 `--train-percents`를 둘 다 생략하면 train count 128이다.
- `--train-percent P`는 `round(C(n,m) * P / 100)`을 round-half-even으로 계산한다. 재현 가능한 비교에는 정확한 `--train-count`를 권장한다.
- `--n-test -1`은 가능한 나머지를 모두 사용하되 큰 공간에서는 50,000개로 제한한다. 권장 명령은 평가 비용을 고정하려고 `--n-test 20000`을 사용한다.

`--split-strategy`는 다음 중 하나다.

- `random`: seed로 train 조합을 무작위 선택한다.
- `relation-complete`: numeric order의 인접 관계와 `(x % modulus, x)` order의 인접 관계의 합집합을 모두 덮는 deterministic basis를 먼저 구성한다. numeric order는 `ascending`과 `alternating`이 공유한다. 같은 seed로 tie-break가 재현되며, basis 뒤에는 무작위 조합을 중복 없이 채워 요청한 train count를 정확히 맞춘다. 요청 count가 basis보다 작거나 test가 0이면 오류다.

두 전략 모두 train/test는 조합 identity 기준으로 분리되고 정확한 요청 크기이며 서로 겹치지 않는다. 작은 공간은 전체 rank를 섞고 큰 공간은 조합 전체를 열거하지 않고 rank를 직접 샘플링한다. 한 번에 materialize할 수 있는 한도는 500,000행과 5,000,000 integer cell이다.

### 관계 completeness와 strata

categorical 값에는 내장된 대소 관계가 없으므로, 학습 관계 그래프가 target order를 유일하게 정하는지 확인하지 않으면 일반화 실패와 식별 불가능한 문제를 구분할 수 없다. `relation-complete`는 각 target order의 모든 인접 관계를 train에서 직접 덮어 전체 순서를 식별 가능하게 만든다. test는 train 관계에 따라 다음으로 분류된다.

- `direct`: 예제 안의 모든 필요한 ordered pair가 train에서 직접 함께 관측됐다.
- `transitive`: 직접 관측되지 않은 pair가 하나 이상 있지만, 모든 pair를 train 관계의 transitive closure로 추론할 수 있다.
- `unresolved`: closure로도 추론할 수 없는 pair가 하나 이상 있다.

relation-complete 데이터에서는 모든 과제의 `unresolved`가 0이다. 그래서 `test_transitive_exact_acc`는 직접 본 pair의 단순 재사용과 구별되는 핵심 일반화 지표다. random split은 같은 count에서도 unresolved가 생길 수 있어 identifiability 대조군으로 사용한다.

`metadata.json`의 `coverage`는 다음을 포함한다.

- `elements`: train 원소 coverage와 빈도 요약.
- `pairs`: 정확한 전체 pair coverage와 test pair 관측률. train/test에서 추적 가능한 unique pair ID 상한의 합이 1,000,000을 넘으면 근삿값 대신 `status: skipped`, 이유와 occurrence 상한을 기록한다.
- `adjacent_pairs`: `ascending`, `mod`, `alternating` key별 인접 pair coverage/빈도 사전.
- `order_identifiable`: 같은 세 task key별 boolean 사전.
- `test_strata`: 같은 세 task key 아래 `direct`, `transitive`, `unresolved` test 개수를 담은 사전.

## 학습

한 번 생성한 split을 세 과제가 공유하면 데이터 차이 없이 비교할 수 있다.

```bash
python sortformer.py --data data/n50_m5_tc128_rc --task ascending \
  --output-constraint free --steps 200000 \
  --log-csv runs/ascending_rc_free.csv --out-dir runs/ascending_rc_free \
  --ckpt-every 10000

python sortformer.py --data data/n50_m5_tc128_rc --task mod \
  --output-constraint free --steps 200000 \
  --log-csv runs/mod_rc_free.csv --out-dir runs/mod_rc_free

python sortformer.py --data data/n50_m5_tc128_rc --task alternating \
  --output-constraint free --steps 200000 \
  --log-csv runs/alternating_rc_free.csv --out-dir runs/alternating_rc_free
```

파일 없이 generated data를 쓰면 기본 train count는 128이다. 정확히 명시하는 예시는 다음과 같다.

```bash
python sortformer.py --n 50 --m 5 --modulus 5 --train-count 128 \
  --split-strategy relation-complete --data-seed 0 --n-test 20000 \
  --task ascending --output-constraint free --steps 200000 \
  --log-csv runs/generated.csv --out-dir runs/generated
```

### 출력 제약


| 값            | 동작                                                                        |
| --------------- | ----------------------------------------------------------------------------- |
| `permutation` | 입력 원소 중 아직 생성하지 않은 값만 허용한다. 출력은 항상 입력의 순열이다. |
| `input-only`  | 입력 원소만 허용하되 중복 생성을 허용한다.                                  |
| `free`        | 전체 vocabulary를 허용한다.                                                 |

기본값 `permutation`은 순서 규칙에 집중하게 하지만 과제를 구조적으로 쉽게 만들어 grokking 구간을 줄일 수 있다. 알고리즘 전체를 보려는 권장 기준은 `free`이고, 다른 두 모드는 출력 제약 대조군이다. label smoothing은 `free`에서만 허용된다.

주요 기본값은 embedding 128, head 4, encoder/decoder 각 2층, batch 512, AdamW learning rate `1e-3`, weight decay `1.0`, warmup 100, constant schedule, 평가 간격 250, `--n-eval 4096`이다. `--batch-size -1`과 `--n-eval -1`은 각각 full-batch 학습과 split 전체 평가다.

## RunPod / GPU

장치를 지정하지 않으면 `cuda`, `mps`, `cpu` 순으로 선택한다. CUDA 기본 dtype은 bf16이며, 지원하지 않는 GPU는 `--dtype float16`을 사용한다.

```bash
python sortformer.py --data data/n50_m5_tc128_rc --task ascending \
  --output-constraint free --device auto --dtype bfloat16 --compile \
  --batch-size 2048 --eval-batch 4096 --steps 200000 \
  --log-csv runs/ascending_rc_free.csv --out-dir runs/ascending_rc_free \
  --ckpt-every 10000
```

메모리가 부족하면 `--batch-size`와 `--eval-batch`부터 낮춘다. `--compile` 첫 호출은 느리므로 짧은 실험에서는 생략할 수 있다.

체크포인트는 모델, optimizer, AMP scaler, 난수 상태, 모델 설정과 run signature를 저장한다. `--steps`는 추가량이 아니라 최종 step이다.

```bash
python sortformer.py --data data/n50_m5_tc128_rc --task ascending \
  --output-constraint free --device auto --dtype bfloat16 --compile \
  --batch-size 2048 --eval-batch 4096 --steps 200000 \
  --resume runs/ascending_rc_free/ckpt_00010000.pt \
  --log-csv runs/ascending_rc_free.csv --out-dir runs/ascending_rc_free
```

재개 시 모델 설정과 task, output constraint, model seed, 데이터셋 fingerprint가 일치해야 한다. CSV schema/signature도 검증하며 CSV가 체크포인트보다 뒤 step까지 있으면 trajectory 혼합을 막기 위해 거부한다.

## Sweep

권장 sweep은 `n=50`, `m=5`, `modulus=5`, train count `32/64/128/256`, test 20,000에서 relation-complete/free를 실행하고 동일 조건 random을 비교한다.

```bash
python sweep.py --n 50 --m 5 --modulus 5 --data-seed 0 --n-test 20000 \
  --train-counts 32 64 128 256 \
  --split-strategies relation-complete random \
  --tasks ascending mod alternating --seeds 42 43 44 \
  --weight-decays 1.0 --output-constraints free --steps 200000 \
  --out-dir sweeps/n50_m5_counts -- \
  --device auto --dtype bfloat16 --batch-size 2048 \
  --eval-batch 4096 --eval-every 250 --n-eval 20000
```

`--train-counts`와 `--train-percents`는 상호 배타적이며 generated-data 기본은 count 128이다. `--split-strategies`는 하나 이상을 받아 각 sizing과 곱집합으로 실행한다. `--data`를 사용하면 이미 고정된 split이므로 sizing/strategy 목록은 여러 값을 sweep할 수 없고 generated-data 옵션은 실제 command에 전달되지 않는다.

`--` 뒤에는 sweep이 관리하지 않는 `sortformer.py` 옵션만 전달한다. `--dry-run`, `--skip-existing`, `--continue-on-error`, `--selftest`를 지원한다. `sweep_manifest.json`에는 command/config/status/artifact가, `sweep_summary.csv`에는 다음 group별 집계가 기록된다.

- task, `train_count` 또는 `train_percent`, `split_strategy`, dataset, weight decay, output constraint.
- run 수, overall test exact 90% 성공 수, transitive exact 90% 성공 수.
- grokking gap의 median/min/max.
- 최종 train/test exact median과 최종 transitive exact median.
- CSV와 checkpoint 경로.

## CSV와 그래프

CSV schema는 다음 순서다.

```text
step,lr,weight_norm,
train_loss,train_token_acc,train_gen_in_set_token_acc,train_set_acc,train_exact_acc,
test_loss,test_token_acc,test_gen_in_set_token_acc,test_set_acc,test_exact_acc,
elapsed_seconds,task,output_constraint,run_signature_sha256,
train_eval_count,test_eval_count,
test_direct_exact_acc,test_transitive_exact_acc,test_unresolved_exact_acc,
test_direct_count,test_transitive_count,test_unresolved_count
```

overall train/test 지표와 함께 각 test stratum의 자유 생성 exact accuracy와 실제 평가 count를 기록한다. 평가 subset에 특정 stratum이 없으면 해당 accuracy는 빈 값이고 count는 0이다. `run_signature_sha256`은 task, output constraint, model seed, train/test 입력 fingerprint로 만든 run 식별자다.

`gen_in_set_token_acc`, `set_acc`, `exact_acc`는 각각 입력 원소 사용, 중복/누락 없는 집합 완성, target 순서를 구분한다. `permutation`에서는 앞의 두 값이 구조적으로 1이고 `input-only`에서는 in-set이 구조적으로 1이다. teacher-forced loss/token accuracy도 constraint 후보 공간에서 계산되므로 서로 다른 constraint의 절대 난이도 비교보다 같은 constraint의 추세에 사용한다.

```bash
python plot_grokking.py "runs/*.csv" --out runs/grokking.png \
  --title "Set sorting grokking"
```

다섯 panel은 overall train/test exact, test 생성 분해, test direct/transitive/unresolved exact, loss, parameter L2 norm이다. 기본 x축은 log이며 `--linear-x`로 바꾼다. `gap x`는 `test exact >= 0.90` 최초 step을 `train exact >= 0.99` 최초 step으로 나눈 값이다.

## 권장 실험 순서

1. 위 sweep으로 count `32, 64, 128, 256`의 relation-complete/free를 비교한다.
2. 같은 count와 seed의 random split을 비교해 unresolved와 identifiability 영향을 확인한다.
3. overall뿐 아니라 `test_transitive_exact_acc`와 해당 count를 기준으로 관계 합성 일반화를 판단한다.
4. 전이가 너무 빠르면 `free`, 모델 크기 또는 count를 조정하고, train만 암기하면 100,000~500,000 step과 weight decay `0.1, 1.0, 3.0`을 비교한다.
5. 후보 설정은 최소 3개 model seed로 반복한다.

Grokking은 optimizer나 weight decay 하나로 보장되지 않는다. train count, split의 관계 식별 가능성, 모델 크기, 초기화와 출력 제약을 한 번에 하나씩 통제해야 한다.
