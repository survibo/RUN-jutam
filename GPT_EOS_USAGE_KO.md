# EOS·Label Masking GPT 정렬 모델 사용법

`sortformer_gpt_eos.py`는 숫자 입력과 정렬 결과를 하나의 causal language-model
sequence로 연결해 학습하는 GPT-2-style decoder-only Transformer다. 숫자마다 정확히
하나의 token을 사용하며 encoder, cross-attention, 외부 메모리, 정렬 전용 layer나
출력 hard mask는 사용하지 않는다.

기존 `sortformer_gpt.py`와 달리 다음을 명시적으로 지원한다.

- `BOS`, 입력, `SEP`, 정렬 결과, `EOS`를 포함하는 단일 sequence
- 정렬 결과와 EOS에만 loss를 주는 label masking
- 서로 겹치지 않는 train/validation/test 3-way split
- validation 지표를 포함한 장기 grokking CSV 기록
- AdamW, Adam, SGD 선택과 checkpoint 재개

## 빠른 확인

프로젝트 루트에서 다음을 실행한다.

```bash
python sortformer_gpt_eos.py --smoke
```

`--smoke`는 CPU에서 작은 조합 공간을 20 step 학습하며 모델 forward, label
masking, 자유 생성과 세 split 평가를 확인한다. 성능 benchmark는 아니다.

사용 가능한 옵션 전체는 다음으로 확인한다.

```bash
python sortformer_gpt_eos.py --help
```

## Sequence와 label masking

길이 `k`의 입력은 다음처럼 직렬화된다.

```text
[BOS, input_1, ..., input_k, SEP, sorted_1, ..., sorted_k, EOS]
```

예를 들어 입력이 `1, 23, 15`이면 full sequence와 label은 다음과 같다.

```text
input_ids = [BOS, 1, 23, 15, SEP, 1, 15, 23, EOS]
labels    = [  X, X,  X,  X,   X, 1, 15, 23, EOS]
```

`X`는 PyTorch의 `ignore_index=-100`이다. Causal shift 후 `SEP` 위치의 hidden
state가 첫 정답 `1`을 예측하며, 이후에는 teacher forcing으로 이전 정답 token을
조건으로 다음 정답을 예측한다. 한 example에서 loss에 포함되는 token 수는 정렬된
숫자 `k`개와 EOS를 합친 `k+1`개다.

Inference에서는 `[BOS, inputs..., SEP]`만 prompt로 주고 greedy decoding으로
`k+1`개 위치를 생성한다. 이전에 생성한 token은 다음 단계 context에 다시 들어간다.

## 기본 학습 실행

아래 값들은 예시일 뿐이며 숫자 범위, `k`, split 크기와 모든 모델·optimizer
설정은 CLI에서 바꿀 수 있다.

```bash
python sortformer_gpt_eos.py \
  --n 30 \
  --minimum 0 \
  --k 5 \
  --train-count 500 \
  --validation-count 1000 \
  --test-count 1000 \
  --split-strategy random \
  --data-seed 0 \
  --n-layer 4 \
  --n-embd 256 \
  --n-head 8 \
  --dropout 0.0 \
  --optimizer adamw \
  --lr 1e-3 \
  --weight-decay 1.0 \
  --steps 500000 \
  --batch-size 256 \
  --eval-every 500 \
  --n-eval -1 \
  --eval-batch 1024 \
  --device auto \
  --dtype bfloat16 \
  --log-csv runs/gpt_eos/metrics.csv \
  --out-dir runs/gpt_eos \
  --ckpt-every 10000
```

`--m`은 `--k`, `--hidden-dim`은 `--n-embd`, `--attention-heads`는
`--n-head`, `--layers`는 `--n-layer`, `--n-test`는 `--test-count`의 alias다.
기존 GPT 실행 명령을 옮길 때 사용할 수 있다. Decoder-only 구조를 명확히 하기
위한 호환 옵션 `--n-enc-layer`는 `0`만 허용한다.

## Vocabulary와 숫자 범위

`--n N --minimum A`는 실제 숫자 `A, A+1, ..., A+N-1`을 만든다. 각 숫자는
BPE나 문자열 분해 없이 하나의 token ID에 대응한다. 그 뒤에 `BOS`, `SEP`, `EOS`
세 special token이 배치된다.

예를 들어 `--n 24 --minimum 0`이면 다음과 같다.

```text
number token IDs: 0 ... 23
BOS: 24
SEP: 25
EOS: 26
model vocabulary size: 27
```

입력은 `N`개 숫자 중 중복 없이 `k`개를 뽑은 조합이다. 전체 가능한 조합 수는
`C(N, k)`이며, 같은 underlying combination의 다른 입력 순서가 서로 다른 split에
들어가지 않는다.

## Train/validation/test split

Train 크기는 다음 중 하나만 지정한다.

```bash
--train-count 500
```

```bash
--train-percent 10
```

둘 다 생략하면 기존 실행 관례와 같이 train count 128을 사용한다. Validation과
test는 `--validation-count`, `--test-count`로 지정한다.

- 둘 다 `-1`이면 train을 제외한 나머지를 절반씩 나눈다.
- 하나만 `-1`이면 명시된 split을 제외한 나머지를 해당 split에 배정한다.
- 둘 다 양수이면 정확히 지정한 수만 사용하며 남은 조합은 materialize하지 않는다.
- 세 split은 모두 적어도 한 example을 포함해야 한다.

큰 조합 공간에서 나머지 전체를 자동 배정하면 안전 한도인
`--max-materialized-examples`를 넘을 수 있다. 대규모 실험에서는 validation/test
개수를 명시하는 것이 좋다.

Split 전략은 다음 두 가지다.

| 값 | 동작 |
| --- | --- |
| `random` | `--data-seed`로 조합 rank를 비복원 무작위 추출하고 세 split으로 나눈다. |
| `lexicographic` | 조합 rank 앞부분부터 train, validation, test 순으로 나눈다. 분포 이동 기준선으로 사용할 수 있다. |

Train/validation/test split의 행 순서와 각 행 내부의 입력 숫자 순서는 dataset 생성
시점에 seed로 한 번만 결정된다. 학습이 시작된 뒤에는 어느 split도 다시 섞거나
새 permutation을 만들지 않는다. Target은 항상 오름차순이다.

## 모델 설정

모델은 learned token embedding과 absolute positional embedding을 더한 뒤, 동일한
causal mask를 사용하는 GPT block을 통과한다. 각 block은 pre-LayerNorm,
multi-head self-attention, GELU MLP, residual connection으로 구성된다.

주요 옵션은 다음과 같다.

| 옵션 | 의미 |
| --- | --- |
| `--n-layer` | GPT block 수 |
| `--n-embd` | hidden/embedding 차원 |
| `--n-head` | attention head 수; `n-embd`를 나누어야 함 |
| `--dropout` | embedding, attention, MLP dropout |
| `--no-bias` | attention과 MLP의 linear bias 제거 |
| `--tie` | token embedding과 LM head weight 공유 |
| `--init-std` | linear/embedding 정규분포 초기화 표준편차 |
| `--layer-norm-epsilon` | LayerNorm epsilon |

`--output-constraint`는 외형적 CLI 호환성을 위해 존재하지만 `free`만 허용한다.
따라서 모델은 입력에 있는 숫자나 아직 생성하지 않은 숫자로 출력 후보를 제한받지
않는다. 학습 초기에 숫자 대신 BOS나 SEP를 생성할 수 있으며, 이를 피하는 법도
training data에서 학습해야 한다.

## Optimizer와 batch

`--optimizer`는 `adamw`, `adam`, `sgd`를 지원한다.

| 옵션 | 적용 대상 |
| --- | --- |
| `--lr`, `--weight-decay` | 모든 optimizer |
| `--beta1`, `--beta2` | AdamW와 Adam |
| `--momentum` | SGD |
| `--grad-clip` | gradient norm clipping; `0`이면 비활성화 |
| `--warmup` | learning-rate warmup step |
| `--lr-schedule` | `constant`, `cosine`, `linear` |
| `--label-smoothing` | output 영역의 causal LM loss |

양수 `--batch-size B`는 train split의 고정된 행 순서를 바꾸지 않고 연속한 `B`개씩
순환한다. 마지막 행을 지나면 첫 행부터 이어지며 random shuffle이나 복원추출은
하지 않는다. `--batch-size -1`은 매 step에서 고정된 train split 전체를 같은
순서로 사용하는 full-batch다.
CSV의 `epoch`은 누적 처리 example 수를 train split 크기로 나눈 값이다.

## 평가 지표와 CSV

평가는 step 1, `--eval-every` 간격, 마지막 step에 수행한다.
`--n-eval -1`은 각 split 전체를 평가하고, 양수 값은 split 앞부분에서 해당 개수만
평가한다. `--eval-batch`는 평가 chunk 크기다.

각 지표의 정의는 다음과 같다.

| 지표 | 정의 |
| --- | --- |
| `*_loss` | teacher forcing으로 정답 `k`개와 EOS에 계산한 평균 cross-entropy |
| `*_token_acc` | 같은 output 영역의 teacher-forced next-token accuracy |
| `*_exact_acc` | 자유 생성한 `sorted_1 ... sorted_k EOS` 전체가 일치한 example 비율 |

Grokking 판단에는 teacher-forced token accuracy보다 오류 누적을 포함하는
`train_exact_acc`, `validation_exact_acc`, `test_exact_acc`를 우선 확인한다.
일반적으로 train exact가 먼저 상승한 뒤 validation/test exact가 늦게 상승하는지를
장기간 추적한다.

CSV에는 다음 열이 기록된다.

```text
step,epoch,lr,weight_norm,
train_loss,train_token_acc,train_exact_acc,
validation_loss,validation_token_acc,validation_exact_acc,
test_loss,test_token_acc,test_exact_acc,
elapsed_seconds,train_eval_count,validation_eval_count,test_eval_count,
run_signature_sha256
```

## Grokking 그래프

`plot_grokking.py`가 CSV schema를 자동 감지하므로 기존 모델과 같은 명령으로
그래프를 만들 수 있다.

```bash
python plot_grokking.py runs/gpt_eos/metrics.csv \
  --out runs/gpt_eos/grokking.png \
  --title "GPT-2 sorting with EOS"
```

EOS 모델의 그래프는 다음 네 panel로 구성된다.

1. Train/validation/test autoregressive exact accuracy
2. Train/validation/test teacher-forced token accuracy
3. Train/validation/test loss
4. Parameter L2 norm

터미널 요약에는 train exact 0.99, validation/test exact 0.90과 0.99에 최초로
도달한 step, 최종 accuracy와 grokking gap이 표시된다. `gap x`는 기존 정의와
같이 `test exact >= 0.90` 최초 step을 `train exact >= 0.99` 최초 step으로 나눈
값이다.

여러 seed 또는 설정은 glob으로 함께 그릴 수 있다.

```bash
python plot_grokking.py "runs/gpt_eos_seed*/metrics.csv" \
  --out runs/gpt_eos_comparison.png
```

기존 모델 CSV와 EOS 모델 CSV를 같은 glob에 넣는 것도 지원한다. 이 경우 exact,
loss와 norm은 공통 panel에 겹쳐 그리고, EOS token accuracy와 기존 모델의 생성
분해 및 strata panel을 함께 추가한다. 기본 x축은 log scale이고 `--linear-x`로
선형 축을 사용할 수 있다.

## GPU 실행

`--device auto`는 CUDA, MPS, CPU 순으로 사용 가능한 장치를 선택한다. CUDA에서는
기본 예시처럼 bf16을 사용할 수 있다. GPU가 bf16을 지원하지 않으면
`--dtype float16`을 사용한다.

```bash
python sortformer_gpt_eos.py \
  --n 50 --k 5 \
  --train-count 128 --validation-count 10000 --test-count 20000 \
  --n-layer 4 --n-embd 256 --n-head 8 \
  --device auto --dtype bfloat16 --compile \
  --batch-size 8192 --eval-batch 4096 \
  --steps 500000 --eval-every 500 \
  --log-csv runs/gpt_eos_gpu.csv \
  --out-dir runs/gpt_eos_gpu --ckpt-every 10000
```

메모리가 부족하면 `--batch-size`, 그다음 `--eval-batch`를 줄인다. `--compile`은
CUDA에서만 적용되며 첫 호출 때 compilation 시간이 든다.

## Checkpoint 저장과 재개

`--out-dir`을 지정하면 다음 파일이 만들어진다.

```text
out-dir/
  config.json
  ckpt_00010000.pt
  ckpt_00020000.pt
  ...
  ckpt_final.pt
```

주기 checkpoint는 `--ckpt-every`가 양수일 때 저장한다. `ckpt_final.pt`는
`--out-dir`을 지정한 모든 정상 종료 실행에서 저장된다. Checkpoint에는 model,
optimizer, AMP scaler, 난수 상태, model/dataset config와 run signature가 포함된다.

다음처럼 동일 설정으로 최종 step만 늘려 재개한다.

```bash
python sortformer_gpt_eos.py \
  --n 30 --k 5 \
  --train-count 500 --validation-count 1000 --test-count 1000 \
  --data-seed 0 --seed 42 \
  --n-layer 4 --n-embd 256 --n-head 8 \
  --optimizer adamw --lr 1e-3 --weight-decay 1.0 \
  --steps 750000 \
  --resume runs/gpt_eos/ckpt_00500000.pt \
  --log-csv runs/gpt_eos/metrics.csv \
  --out-dir runs/gpt_eos
```

`--steps`는 추가 학습량이 아니라 도달할 최종 step이다. Model 또는 dataset config,
split fingerprint, vocabulary, optimizer 종류, seed 또는 batch size가 다르면 재개를 거부한다.
기존 CSV의 signature가 다르거나 마지막 행이 checkpoint보다 뒤에 있어도 trajectory
혼합 방지를 위해 거부한다.

## Python API 사용

Vocabulary, split, serialization과 모델은 독립적으로 사용할 수 있다.

```python
import torch

from sortformer_gpt_eos import (
    DatasetConfig,
    GPTSortTransformer,
    ModelConfig,
    NumberVocabulary,
    create_dataset_splits,
    examples_to_tensors,
    serialize_batch,
)

vocabulary = NumberVocabulary.contiguous(size=24, minimum=0)
splits = create_dataset_splits(
    DatasetConfig(
        n=24,
        set_size=3,
        train_count=500,
        validation_count=200,
        test_count=200,
        split_strategy="random",
        seed=0,
    )
)

inputs, targets = examples_to_tensors(splits.train)
input_ids, labels = serialize_batch(inputs[:8], targets[:8], vocabulary)

model = GPTSortTransformer(
    ModelConfig(
        vocab_size=vocabulary.number_token_count,
        set_size=3,
        n_embd=128,
        n_head=4,
        n_layer=2,
    )
)
logits = model(input_ids)
generated = model.generate(inputs[:8])
```

`generated`의 shape은 `[batch, k + 1]`이며 마지막 기대 token은 EOS다.
