# Entity–Value KB 정렬 Grokking 모델 사용법

`sortformer_gpt_eos.py`는 하나의 GPT-2-style causal Transformer에 다음 두 task를
동시에 학습시킨다.

1. 모든 entity의 atomic fact: `entity + ATTR → value`
2. ID entity에서의 sorting: entity들을 KB value 오름차순으로 재배열

OOD entity도 atomic fact 학습에는 포함되지만 sorting training에는 절대 포함되지
않는다. 최종 목표는 모델이 이미 암기한 OOD atomic knowledge와 ID에서 학습한 정렬
규칙을 합성하여 OOD entity tuple을 정렬하는지 측정하는 것이다.

## 빠른 검증

```bash
python sortformer_gpt_eos.py --smoke
python plot_grokking.py --selftest
```

`--smoke`는 entity 20개, ID/OOD 16/4, `k=3`인 작은 CPU 실험이다. Dataset
integrity, padding, 두 task의 loss, free-running 생성과 평가 경로를 확인하기 위한
것이며 성능 benchmark가 아니다.

전체 옵션은 다음으로 확인한다.

```bash
python sortformer_gpt_eos.py --help
```

## Vocabulary

Entity, value, special token은 서로 겹치지 않는 ID 영역을 사용한다.

```text
Entity: E0000 ... E0999
Value:  V0 ... V100
Special: BOS SEP EOS PAD ATOM SORT ATTR
```

Entity token ID는 entity 이름만 나타낸다. 실행 시작 시 seed로 생성한 무작위 KB
mapping이 value를 결정하므로 entity token ID 순서와 value 순서는 무관하다. 서로
다른 entity가 같은 value를 갖는 것은 허용한다.

## 두 causal-LM sequence

Atomic fact는 다음처럼 직렬화한다.

```text
input_ids = [BOS, ATOM, ATTR, E001, SEP, V23, EOS]
labels    = [  X,    X,    X,    X,   X, V23, EOS]
```

Sorting example은 value token을 입력에 넣지 않는다.

```text
KB(E10)=23, KB(E42)=1, KB(E81)=15

input_ids = [BOS, SORT, ATTR, E10, E42, E81, SEP, E42, E81, E10, EOS]
labels    = [  X,    X,    X,   X,   X,   X,   X, E42, E81, E10, EOS]
```

`X`는 `IGNORE_INDEX=-100`이다. Loss는 SEP 이후의 정답과 EOS에만 계산한다.
Atomic과 sorting sequence 길이가 다르므로 shorter sequence는 오른쪽 PAD로 맞추고,
PAD는 attention mask와 label mask 모두에서 제외한다.

## 기본 실행

```bash
python sortformer_gpt_eos.py \
  --n-entities 1000 \
  --id-fraction 0.9 \
  --value-min 0 \
  --value-max 100 \
  --k 3 \
  --phi 3.6 \
  --validation-count 2000 \
  --test-count 5000 \
  --training-mode combined \
  --n-layer 8 \
  --n-embd 512 \
  --n-head 16 \
  --dropout 0.0 \
  --optimizer adamw \
  --lr 1e-3 \
  --weight-decay 1.0 \
  --steps 100000 \
  --batch-size 256 \
  --eval-every 50 \
  --device auto \
  --dtype bfloat16 \
  --log-csv runs/entity_kb/metrics.csv \
  --out-dir runs/entity_kb \
  --ckpt-every 10000
```

각 수치는 CLI로 변경할 수 있다. `--m`, `--hidden-dim`, `--attention-heads`,
`--layers`, `--n-test`는 각각 `--k`, `--n-embd`, `--n-head`, `--n-layer`,
`--test-count`의 alias다. `--n-enc-layer`는 호환 옵션이며 `0`만 허용한다.

## KB와 ID/OOD split

`--data-seed` 하나에서 서로 독립적인 deterministic seed stream을 파생하여 다음을
실행 시작 시 한 번만 생성한다.

- 각 entity의 value를 replacement sampling한 KB
- entity를 무작위 permutation한 뒤 나눈 ID/OOD membership
- ID sorting train과 unseen-combination validation tuple
- OOD-only sorting test tuple
- 각 sorting tuple의 고정 input permutation

ID 개수는 `round(n_entities × id_fraction)`이다. ID/OOD 목록은 token 번호 구간으로
자르지 않는다. `config.json`에는 실제 KB 전체와 ID/OOD entity 목록, seed와 dataset
fingerprint가 저장된다.

Sorting tuple 안에서 value가 중복되는 조합은 제외한다. 요청한 example 수가 가능한
distinct-value 조합 수를 넘으면 학습 전에 오류가 발생한다.

## φ와 sorting 데이터 수

이 구현에서 φ는 다음과 같이 정확히 정의한다.

```text
phi = ID sorting training example 수 / ID atomic fact 수
sorting_train_count = round(ID entity 수 × phi)
```

따라서 ID entity가 900개일 때 `--phi 3.6`은 3240개, `--phi 7.2`는
6480개의 sorting training example을 만든다. 정확한 개수를 직접 지정하려면 다음을
사용한다.

```bash
--sorting-train-count 3240
```

`--phi`와 `--sorting-train-count`는 상호 배타적이다. 둘 다 생략하면 φ=3.6을
사용한다. Validation은 sorting train에서 사용하지 않은 ID combination이고 test는
OOD entity로만 이루어진 combination이다.

## 고정 데이터와 input permutation

기본값에서는 KB, split, example 행 순서와 각 sorting input의 entity 순서를 시작
시점에 한 번 고정한다. 학습 중 재표본화하거나 다시 섞지 않는다. 양수 batch는 이
고정된 행 순서를 cyclic하게 읽으며 `--batch-size -1`은 combined pool 전체를 매
step 사용한다.

Input permutation augmentation이 필요한 별도 대조 실험에서만 다음을 지정한다.

```bash
--dynamic-input-permutation
```

이 옵션은 선택된 sorting training input의 entity 순서만 step seed로 다시 섞는다.
Validation/test와 target은 항상 고정이다. 기본값은 false이며 설정은 config, CSV와
checkpoint signature에 기록된다.

## Atomic/sorting task mixture

두 방식 모두 같은 Transformer parameter와 causal-LM loss를 사용한다.

### Combined dataset

```bash
--training-mode combined
```

Atomic example 전체와 ID sorting training example을 하나의 pool에 합친 뒤 seed로
한 번만 섞어 고정한다. Pool의 실제 atomic 비율이 effective atomic fraction이다.
기본 방식이다.

### Controlled task mixture

```bash
--training-mode controlled --atomic-fraction 0.25 --batch-size 256
```

각 batch의 25%를 atomic, 나머지를 sorting example로 구성한다. 두 dataset stream과
batch 내 task 위치는 처음 한 번 고정되며 이후 cyclic하게 순회한다. 반올림 후 각
task가 최소 한 행 포함되어야 하므로 controlled mode에서는 양수 batch size가
필요하다.

## Dataset integrity 검사

학습 시작 전에 다음을 강제로 검사하며 하나라도 위반되면 `ValueError`를 발생시킨다.

1. ID/OOD intersection이 비어 있고 모든 entity가 정확히 한 group에 속함
2. 모든 entity, 특히 모든 OOD entity의 올바른 atomic fact가 training에 존재함
3. Sorting train과 ID validation에는 ID entity만 존재함
4. OOD sorting test에는 OOD entity만 존재함
5. Sorting train과 ID validation combination이 겹치지 않음
6. 각 split 내부 combination이 중복되지 않음
7. 한 sorting tuple의 KB value가 모두 다름
8. Target entity가 input entity의 정확한 permutation임
9. Target이 실제 KB value 오름차순임
10. Serialized sorting input에 value token이 없음

Run 시작 시 동일한 조건을 사람이 확인할 수 있도록 dataset summary와 atomic OOD,
ID sorting train, OOD sorting test example도 출력한다. Preview 수는 `--preview`로
조절하며 `0`이면 example 출력을 생략한다.

## Evaluation과 CSV

모든 exact accuracy는 teacher forcing 없는 greedy autoregressive generation으로
계산한다. Atomic은 `[BOS ATOM ATTR entity SEP]` 뒤에 `value, EOS`를 생성하고,
sorting은 `[BOS SORT ATTR entities... SEP]` 뒤에 `k`개 entity와 EOS를 생성한다.

매 평가 시 다음을 별도로 측정한다.

- Atomic 전체/ID/OOD loss, teacher-forced token accuracy, free-running exact
- ID sorting train loss/token/exact
- Unseen ID-combination validation sorting loss/token/exact
- OOD-only test sorting loss/token/exact
- Invalid token, early EOS, duplicate entity, input에 없는 entity 생성 비율

호환용 기본 열은 sorting curve를 다음처럼 나타낸다.

| 호환 열 | 의미 |
| --- | --- |
| `train_exact_acc` | ID sorting train exact |
| `validation_exact_acc` | unseen ID-combination sorting exact |
| `test_exact_acc` | OOD sorting exact |
| `train_loss` | mixture 비율로 가중한 total training loss |

`atomic_ood_exact_acc`가 충분히 높지 않으면 낮은 OOD sorting 성능을 composition
실패로 단정하면 안 된다. 핵심 grokking 패턴은 atomic OOD와 sorting train exact가
먼저 1에 가까워지고 OOD sorting exact가 뒤늦게 상승하는 것이다.

`--n-eval`은 세 sorting split의 평가 example 수를 제한한다. Atomic ID/OOD는
knowledge가 실제로 학습됐는지 확인하기 위해 항상 전체 entity를 평가한다.

## 그래프

`plot_grokking.py`가 entity-KB CSV를 자동 감지한다.

```bash
python plot_grokking.py runs/entity_kb/metrics.csv \
  --out runs/entity_kb/grokking.png \
  --title "Entity-KB OOD sorting"
```

Entity-KB 그래프에는 다음 panel이 포함된다.

1. Sorting train / ID validation / OOD test exact
2. 같은 세 split의 teacher-forced token accuracy
3. Atomic ID / Atomic OOD free-running exact
4. Total / atomic / sorting training loss 분해
5. Sorting validation/test loss
6. Parameter L2 norm

여러 seed는 glob으로 함께 비교할 수 있고 기존 모델 CSV와 섞어서 전달할 수도 있다.

```bash
python plot_grokking.py "runs/entity_kb_seed*/metrics.csv" \
  --out runs/entity_kb_comparison.png
```

기본 x축은 log scale이며 `--linear-x`로 선형 축을 사용한다.

## Checkpoint와 resume

`--out-dir`에는 `config.json`, 주기 checkpoint와 `ckpt_final.pt`가 저장된다.
Checkpoint에는 model/optimizer/scaler, 난수 상태, dataset/model config와 dataset
fingerprint가 포함된다.

```bash
python sortformer_gpt_eos.py \
  --n-entities 1000 --id-fraction 0.9 --value-min 0 --value-max 100 \
  --k 3 --phi 3.6 --validation-count 2000 --test-count 5000 \
  --training-mode combined --batch-size 256 \
  --n-layer 8 --n-embd 512 --n-head 16 \
  --optimizer adamw --lr 1e-3 --weight-decay 1.0 \
  --steps 200000 \
  --resume runs/entity_kb/ckpt_00100000.pt \
  --log-csv runs/entity_kb/metrics.csv \
  --out-dir runs/entity_kb
```

`--steps`는 추가량이 아니라 최종 step이다. Dataset/model config, fingerprint,
optimizer와 LR schedule, precision, training mode, mixture, dynamic permutation,
batch size 또는 seed가 달라지면 재개를 거부한다.

## GPU 실행 참고

`--device auto`는 CUDA, MPS, CPU 순으로 선택한다. CUDA에서 bf16을 지원하지 않으면
`--dtype float16`을 사용한다. OOM이면 `--batch-size`, 그다음 `--eval-batch`와
`--n-eval`을 줄인다. `--compile`은 CUDA에서만 적용된다.
