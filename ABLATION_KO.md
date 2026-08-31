# 어텐션 Ablation 실행 설명서

`sortformer.py`의 어텐션은 세 곳이다. 이 세 개를 하나씩만 남긴 파일이 `sortformer_attn1.py`, `sortformer_attn2.py`, `sortformer_attn3.py`다. 어텐션 외의 코드는 `sortformer.py`와 동일하므로 CLI, CSV schema, 체크포인트 형식이 모두 같다.

데이터 생성, sweep, 그래프의 일반 절차는 [`USAGE_KO.md`](USAGE_KO.md)를 따른다. 이 문서는 세 ablation 파일을 돌리는 명령만 다룬다.

## 1. 파일별 구조

| 파일 | 남긴 어텐션 | 인코더 | 디코더 |
| --- | --- | --- | --- |
| `sortformer_attn1.py` | 인코더 self-attention | self-attn + MLP | MLP만 |
| `sortformer_attn2.py` | 디코더 self-attention (causal) | MLP만 | self-attn + MLP |
| `sortformer_attn3.py` | 디코더 cross-attention | MLP만 | cross-attn + MLP |
| `sortformer.py` | 세 개 전부 (baseline) | self-attn + MLP | self-attn + cross-attn + MLP |

제거된 어텐션은 모듈 정의, 정규화 레이어, forward 인자(`memory`, `causal_mask`)까지 전부 삭제했다. 각 파일에 `Attention` 인스턴스가 정확히 하나만 남는다.

attn1과 attn2는 cross-attention이 없어 인코더 출력이 디코더에 도달하지 않는다. 인코더는 forward에서 계속 실행되지만 그 결과는 사용되지 않는다.

## 2. 반드시 `--output-constraint free`를 쓸 것

`permutation`과 `input-only`는 `_valid_token_mask`가 입력 집합으로 로짓을 마스킹한다. 즉 어텐션이 없어도 입력 정보가 출력 제약을 통해 새어 들어간다. 어텐션 기여도를 측정하는 것이 목적이면 `free`를 사용해야 한다.

`--smoke`(n=7, m=3)에서 확인한 차이는 다음과 같다.

| 파일 | `permutation` test exact | `free` test exact | `free` test set |
| --- | ---: | ---: | ---: |
| attn1 | 0.706 | 0.000 | 0.000 |
| attn2 | 0.706 | 0.000 | 0.000 |
| attn3 | 0.882 | 0.824 | 0.882 |

`free`에서 attn1과 attn2가 0인 것이 정상이다. 디코더가 입력을 볼 경로 자체가 없다.

## 3. 사전 확인

```bash
pip install -r requirements.txt
python -c "import torch; print(torch.__version__); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no CUDA')"

python sortformer_attn1.py --smoke
python sortformer_attn2.py --smoke
python sortformer_attn3.py --smoke
```

## 4. 공용 데이터 생성

세 파일과 baseline이 같은 데이터를 쓰도록 한 번만 생성한다.

```bash
python sortdata.py \
  --n 50 \
  --m 5 \
  --modulus 5 \
  --train-count 128 \
  --split-strategy relation-complete \
  --seed 0 \
  --n-test 20000 \
  --enumerate-limit 5000000 \
  --out experiments/data/n50_m5_tc128_rc \
  --preview 5
```

## 5. 학습 (GPU, 풀 옵션)

네 명령 모두 파일 이름과 출력 경로만 다르고 하이퍼파라미터는 동일하다. 비교 대상이므로 `--seed`, `--batch-size`, `--steps`를 반드시 맞춰야 한다.

### attn1 — 인코더 self-attention만

```bash
python sortformer_attn1.py \
  --data experiments/data/n50_m5_tc128_rc \
  --task ascending \
  --output-constraint free \
  --n-embd 128 \
  --n-head 4 \
  --n-enc-layer 2 \
  --n-layer 2 \
  --dropout 0.0 \
  --init-std 0.02 \
  --init-scale 1.0 \
  --steps 200000 \
  --batch-size 8192 \
  --lr 1e-3 \
  --beta1 0.9 \
  --beta2 0.98 \
  --weight-decay 1.0 \
  --grad-clip 1.0 \
  --warmup 100 \
  --lr-schedule constant \
  --label-smoothing 0.0 \
  --device auto \
  --dtype bfloat16 \
  --compile \
  --compile-mode default \
  --seed 42 \
  --eval-every 250 \
  --n-eval 20000 \
  --eval-batch 4096 \
  --log-csv experiments/runs/attn1_asc_free.csv \
  --out-dir experiments/runs/attn1_asc_free \
  --ckpt-every 10000
```

### attn2 — 디코더 self-attention만

```bash
python sortformer_attn2.py \
  --data experiments/data/n50_m5_tc128_rc \
  --task ascending \
  --output-constraint free \
  --n-embd 128 \
  --n-head 4 \
  --n-enc-layer 2 \
  --n-layer 2 \
  --dropout 0.0 \
  --init-std 0.02 \
  --init-scale 1.0 \
  --steps 200000 \
  --batch-size 8192 \
  --lr 1e-3 \
  --beta1 0.9 \
  --beta2 0.98 \
  --weight-decay 1.0 \
  --grad-clip 1.0 \
  --warmup 100 \
  --lr-schedule constant \
  --label-smoothing 0.0 \
  --device auto \
  --dtype bfloat16 \
  --compile \
  --compile-mode default \
  --seed 42 \
  --eval-every 250 \
  --n-eval 20000 \
  --eval-batch 4096 \
  --log-csv experiments/runs/attn2_asc_free.csv \
  --out-dir experiments/runs/attn2_asc_free \
  --ckpt-every 10000
```

### attn3 — 디코더 cross-attention만

```bash
python sortformer_attn3.py \
  --data experiments/data/n50_m5_tc128_rc \
  --task ascending \
  --output-constraint free \
  --n-embd 128 \
  --n-head 4 \
  --n-enc-layer 2 \
  --n-layer 2 \
  --dropout 0.0 \
  --init-std 0.02 \
  --init-scale 1.0 \
  --steps 200000 \
  --batch-size 8192 \
  --lr 1e-3 \
  --beta1 0.9 \
  --beta2 0.98 \
  --weight-decay 1.0 \
  --grad-clip 1.0 \
  --warmup 100 \
  --lr-schedule constant \
  --label-smoothing 0.0 \
  --device auto \
  --dtype bfloat16 \
  --compile \
  --compile-mode default \
  --seed 42 \
  --eval-every 250 \
  --n-eval 20000 \
  --eval-batch 4096 \
  --log-csv experiments/runs/attn3_asc_free.csv \
  --out-dir experiments/runs/attn3_asc_free \
  --ckpt-every 10000
```

### baseline — 어텐션 세 개 전부

```bash
python sortformer.py \
  --data experiments/data/n50_m5_tc128_rc \
  --task ascending \
  --output-constraint free \
  --n-embd 128 \
  --n-head 4 \
  --n-enc-layer 2 \
  --n-layer 2 \
  --dropout 0.0 \
  --init-std 0.02 \
  --init-scale 1.0 \
  --steps 200000 \
  --batch-size 8192 \
  --lr 1e-3 \
  --beta1 0.9 \
  --beta2 0.98 \
  --weight-decay 1.0 \
  --grad-clip 1.0 \
  --warmup 100 \
  --lr-schedule constant \
  --label-smoothing 0.0 \
  --device auto \
  --dtype bfloat16 \
  --compile \
  --compile-mode default \
  --seed 42 \
  --eval-every 250 \
  --n-eval 20000 \
  --eval-batch 4096 \
  --log-csv experiments/runs/base_asc_free.csv \
  --out-dir experiments/runs/base_asc_free \
  --ckpt-every 10000
```

### 네 개 순차 실행

```bash
for f in attn1 attn2 attn3 base; do
  case $f in
    base) script=sortformer.py ;;
    *)    script=sortformer_$f.py ;;
  esac
  python $script \
    --data experiments/data/n50_m5_tc128_rc \
    --task ascending --output-constraint free \
    --n-embd 128 --n-head 4 --n-enc-layer 2 --n-layer 2 \
    --dropout 0.0 --init-std 0.02 --init-scale 1.0 \
    --steps 200000 --batch-size 8192 \
    --lr 1e-3 --beta1 0.9 --beta2 0.98 \
    --weight-decay 1.0 --grad-clip 1.0 \
    --warmup 100 --lr-schedule constant --label-smoothing 0.0 \
    --device auto --dtype bfloat16 --compile --compile-mode default \
    --seed 42 --eval-every 250 --n-eval 20000 --eval-batch 4096 \
    --log-csv experiments/runs/${f}_asc_free.csv \
    --out-dir experiments/runs/${f}_asc_free \
    --ckpt-every 10000
done
```

## 6. 학습 (CPU 로컬 확인용, 풀 옵션)

설정과 경로가 맞는지 몇 분 안에 확인하는 용도다. 실험 결과로 쓰지 않는다.

```bash
python sortdata.py \
  --n 30 --m 4 --modulus 5 \
  --train-count 128 --split-strategy relation-complete \
  --seed 0 --n-test 2000 \
  --out experiments/data/abl_check --preview 0

for n in 1 2 3; do
  python sortformer_attn$n.py \
    --data experiments/data/abl_check \
    --task ascending \
    --output-constraint free \
    --n-embd 128 \
    --n-head 4 \
    --n-enc-layer 2 \
    --n-layer 2 \
    --dropout 0.0 \
    --init-std 0.02 \
    --init-scale 1.0 \
    --steps 300 \
    --batch-size 512 \
    --lr 1e-3 \
    --beta1 0.9 \
    --beta2 0.98 \
    --weight-decay 1.0 \
    --grad-clip 1.0 \
    --warmup 100 \
    --lr-schedule constant \
    --label-smoothing 0.0 \
    --device cpu \
    --dtype float32 \
    --seed 42 \
    --eval-every 150 \
    --n-eval 512 \
    --eval-batch 512 \
    --log-csv experiments/runs/abl_attn$n.csv \
    --out-dir experiments/runs/abl_attn$n \
    --ckpt-every 0
done
```

이 설정에서 실제로 나온 300 step 결과는 다음과 같다. attn3만 학습되고 나머지 둘은 test exact 0이다.

| 파일 | final train exact | final test exact |
| --- | ---: | ---: |
| attn1 | 0.008 | 0.000 |
| attn2 | 0.008 | 0.000 |
| attn3 | 1.000 | 0.617 |

## 7. 데이터 파일 없이 인라인 생성

`--data`를 생략하면 sizing 옵션으로 데이터를 즉석 생성한다. `--data`를 줄 때는 아래 8개 옵션을 쓰지 않는다.

```bash
python sortformer_attn3.py \
  --n 50 \
  --m 5 \
  --modulus 5 \
  --train-count 128 \
  --split-strategy relation-complete \
  --data-seed 0 \
  --n-test 20000 \
  --enumerate-limit 5000000 \
  --task ascending \
  --output-constraint free \
  --n-embd 128 --n-head 4 --n-enc-layer 2 --n-layer 2 \
  --steps 200000 --batch-size 8192 \
  --lr 1e-3 --weight-decay 1.0 --warmup 100 \
  --device auto --dtype bfloat16 --compile \
  --seed 42 \
  --eval-every 250 --n-eval 20000 --eval-batch 4096 \
  --log-csv experiments/runs/attn3_inline.csv \
  --out-dir experiments/runs/attn3_inline
```

`--train-count`와 `--train-percent`는 동시에 쓸 수 없다. 둘 다 생략하면 `--train-count 128`이 적용된다.

## 8. 비교 그래프

```bash
python plot_grokking.py "experiments/runs/*_asc_free.csv" \
  --out experiments/plots/ablation.png \
  --title "Attention Ablation (ascending, free)"
```

x축을 선형으로 보려면 `--linear-x`를 추가한다.

## 9. 체크포인트 재개

```bash
python sortformer_attn3.py \
  --data experiments/data/n50_m5_tc128_rc \
  --task ascending --output-constraint free \
  --n-embd 128 --n-head 4 --n-enc-layer 2 --n-layer 2 \
  --steps 200000 --batch-size 8192 \
  --lr 1e-3 --weight-decay 1.0 --warmup 100 \
  --device auto --dtype bfloat16 --compile \
  --seed 42 --eval-every 250 --n-eval 20000 --eval-batch 4096 \
  --resume experiments/runs/attn3_asc_free/ckpt_00010000.pt \
  --log-csv experiments/runs/attn3_asc_free.csv \
  --out-dir experiments/runs/attn3_asc_free \
  --ckpt-every 10000
```

`--steps`는 추가 step 수가 아니라 최종 step이다. 체크포인트는 `model_config`와 `run_signature`(task, constraint, seed, dataset 해시)가 일치할 때만 재개된다. 파일이 다르면 어텐션 구조가 달라 `model_config`가 불일치하므로, attn1의 체크포인트를 attn2로 재개할 수 없다.

## 10. 옵션 레퍼런스

세 파일과 `sortformer.py`가 완전히 동일하다.

### 데이터 (`--data` 미지정 시에만 사용)

| 옵션 | 기본값 | 내용 |
| --- | ---: | --- |
| `--data DIR` | 없음 | `sortdata.py` 출력 디렉터리 |
| `--n` | 20 | 값 범위 `0..n-1` |
| `--m` | 4 | 입력 집합 크기 |
| `--train-count K` | 128 | train을 정확히 K개 |
| `--train-percent P` | 없음 | 전체의 P%를 train으로 |
| `--split-strategy` | `random` | `random` 또는 `relation-complete` |
| `--modulus` | 3 | `mod` 과제의 modulus |
| `--data-seed` | 0 | split seed |
| `--n-test` | -1 | test 개수, -1은 자동 |
| `--enumerate-limit` | 5000000 | 조합 열거 상한 |

### 모델

| 옵션 | 기본값 | 내용 |
| --- | ---: | --- |
| `--n-embd` | 128 | 임베딩 차원, `--n-head`로 나누어떨어져야 함 |
| `--n-head` | 4 | 어텐션 head 수 |
| `--n-enc-layer` | 2 | 인코더 블록 수, 0 이상 |
| `--n-layer` | 2 | 디코더 블록 수, 1 이상 |
| `--dropout` | 0.0 | `[0, 1)` |
| `--tie` | off | `lm_head`를 `encoder_embedding`과 공유 |
| `--output-constraint` | `permutation` | `permutation`, `input-only`, `free` |
| `--init-std` | 0.02 | 초기화 표준편차 |
| `--init-scale` | 1.0 | 초기화 후 전체 파라미터 배율 |

### 학습

| 옵션 | 기본값 | 내용 |
| --- | ---: | --- |
| `--task` | `ascending` | `ascending`, `mod`, `alternating` |
| `--steps` | 100000 | 최종 step |
| `--batch-size` | 512 | -1은 full batch |
| `--lr` | 1e-3 | 학습률 |
| `--beta1`, `--beta2` | 0.9, 0.98 | AdamW betas |
| `--weight-decay` | 1.0 | AdamW weight decay |
| `--grad-clip` | 1.0 | gradient clipping, 0이면 비활성 |
| `--warmup` | 100 | warmup step |
| `--lr-schedule` | `constant` | `constant`, `cosine`, `linear` |
| `--label-smoothing` | 0.0 | `free`에서만 사용 가능 |
| `--seed` | 42 | 모델/배치 seed |

### 실행 환경

| 옵션 | 기본값 | 내용 |
| --- | ---: | --- |
| `--device` | `auto` | `auto`, `cuda`, `mps`, `cpu` |
| `--dtype` | `bfloat16` | CUDA에서만 적용, 그 외는 float32 |
| `--compile` | off | CUDA에서만 적용 |
| `--compile-mode` | `default` | `torch.compile` mode |

### 평가와 출력

| 옵션 | 기본값 | 내용 |
| --- | ---: | --- |
| `--eval-every` | 250 | 평가 주기 |
| `--n-eval` | 4096 | 평가 subset 크기, -1은 전체 |
| `--eval-batch` | 1024 | 평가 batch |
| `--log-csv PATH` | 없음 | 학습 로그 CSV |
| `--out-dir DIR` | 없음 | `config.json`과 체크포인트 |
| `--ckpt-every K` | 0 | 중간 체크포인트 주기, 0이면 최종만 |
| `--resume PATH` | 없음 | 체크포인트 재개 |
| `--smoke` | off | 작은 CPU 자체 검사 |

## 11. 주의사항

| 상황 | 조치 |
| --- | --- |
| attn1/attn2 test exact가 0 | `free`에서는 정상이다. 두 파일은 디코더가 입력을 볼 수 없다 |
| attn1/attn2가 `permutation`에서 잘 나옴 | 어텐션이 아니라 출력 마스크의 효과다. `free`로 다시 측정한다 |
| 체크포인트 재개 거부 | 파일마다 `model_config`가 다르다. 같은 파일의 체크포인트만 재개된다 |
| CSV schema 불일치 | 네 파일의 schema는 동일하다. 예전 CSV는 분리한다 |
| CUDA 메모리 부족 | `--batch-size`와 `--eval-batch`를 낮춘다 |
| `--label-smoothing` 오류 | `--output-constraint free`와 함께 쓴다 |

`sortformer_attn*.py`는 어텐션 부분을 제외하면 `sortformer.py`와 동일해야 한다. `sortformer.py`를 수정하면 세 파일도 다시 만들어야 하며, 다음으로 확인한다.

```bash
for n in 1 2 3; do
  echo "attn$n: Attention $(grep -c 'Attention(config.n_embd' sortformer_attn$n.py)개, diff $(diff sortformer.py sortformer_attn$n.py | grep -c '^[<>]')줄"
done
```

정상이면 각 파일의 `Attention`이 1개이고, diff는 각각 34줄, 26줄, 29줄이며 전부 `EncoderBlock`, `DecoderBlock`, `SortTransformer.forward`, 독스트링 영역에만 나타난다.
