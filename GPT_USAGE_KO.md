# Causal GPT 정렬 모델 사용법

`sortformer_gpt.py`는 encoder와 cross-attention 없이 하나의 GPT stack만 사용하는
정렬 모델이다. 입력은 고정 길이이며 값은 중복되지 않는다.

## 토큰 구성

길이 `m`의 입력과 target을 다음 순서로 직렬화한다.

```text
[무작위 입력 0 ... m-1] [BOS] [target 0 ... m-1]
```

학습 시 실제 Transformer context는 다음 token 예측을 위해
`[입력 m개][BOS][target 앞부분]`이 된다. 전체 context에 절대 위치 embedding과
causal self-attention을 적용하고, loss는 BOS 이후의 target `m`개 예측에만 준다.
입력 token 예측에는 loss를 주지 않는다.

기존 `sortformer_nocross.py`와 달리 입력끼리도 causal하다. 즉 모든 입력 token을
양방향으로 먼저 인코딩하는 prefix mask가 아니라 일반적인 GPT의 하삼각 mask를
그대로 사용한다.

## 데이터 생성

```bash
python gpt_sortdata.py \
  --n 50 \
  --m 5 \
  --modulus 5 \
  --train-count 128 \
  --split-strategy relation-complete \
  --seed 0 \
  --n-test 20000 \
  --out experiments/data/gpt_n50_m5_tc128_rc \
  --preview 5
```

각 행의 입력은 seed와 조합에 의해 재현 가능한 무작위 순서로 저장된다.
train/test 분리는 순서가 아니라 underlying combination 기준이므로 같은 값 집합의
다른 permutation이 두 split에 나뉘어 들어가지 않는다. `ascending`, `mod`,
`alternating` target과 relation coverage/test strata는 기존 데이터와 동일하다.

```text
12 1 19 6 4 -> asc: 1 4 6 12 19 | mod: ... | alt: ...
```

데이터 생성기 자체 검증은 다음과 같이 실행한다.

```bash
python gpt_sortdata.py --selftest
```

## 학습

```bash
python sortformer_gpt.py \
  --data experiments/data/gpt_n50_m5_tc128_rc \
  --task ascending \
  --output-constraint free \
  --n-embd 128 \
  --n-head 4 \
  --n-layer 4 \
  --steps 200000 \
  --batch-size 8192 \
  --lr 1e-3 \
  --weight-decay 1.0 \
  --device auto \
  --dtype bfloat16 \
  --compile \
  --seed 42 \
  --eval-every 250 \
  --n-eval 20000 \
  --eval-batch 4096 \
  --log-csv experiments/runs/gpt_asc_free.csv \
  --out-dir experiments/runs/gpt_asc_free \
  --ckpt-every 10000
```

`--data` 없이도 기존 모델과 같은 generated-data 옵션을 사용할 수 있다.

```bash
python sortformer_gpt.py \
  --n 50 --m 5 --modulus 5 \
  --train-count 128 --split-strategy relation-complete \
  --data-seed 0 --n-test 20000 \
  --task ascending --output-constraint free --steps 200000
```

## 유지되는 옵션

기존 `sortformer.py`의 다음 동작을 그대로 사용한다.

- `ascending`, `mod`, `alternating`
- `permutation`, `input-only`, `free`
- embedding/head/decoder layer/dropout/init/tie 설정
- AdamW, learning-rate schedule, warmup, label smoothing, gradient clipping
- CPU/CUDA/MPS, dtype, `torch.compile`
- CSV 지표, test strata, 체크포인트 저장과 재개
- batch sampling, 평가 간격과 평가 subset

encoder가 없으므로 `--n-enc-layer` 옵션만 제거했다. `--n-layer`가 GPT block 수다.
attention 자체의 기여를 평가할 때는 입력 집합이 hard mask로 노출되지 않는
`--output-constraint free`를 권장한다.

## 빠른 확인

```bash
python sortformer_gpt.py --smoke --output-constraint free
```

`--smoke`는 설치와 전체 실행 경로를 확인하는 20-step CPU 테스트이며 성능
benchmark가 아니다.
