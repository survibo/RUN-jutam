# 정렬 Grokking 실험 원리

이 문서는 모델과 데이터 split의 설계 이유, identifiability, test strata와 grokking 지표의 해석을 설명한다. 실제 실행 명령과 옵션은 [`USAGE_KO.md`](USAGE_KO.md)를 참고한다.

## 1. 실험 질문

이 실험은 정수 집합을 정해진 규칙으로 정렬하는 자기회귀 모델이 학습 예제를 암기한 뒤, 관측한 관계를 합성해 보지 않은 조합까지 처리하는 시점을 측정한다.

세 과제를 비교한다.

| 과제 | target order |
| --- | --- |
| `ascending` | 값 오름차순 |
| `mod` | `(x % modulus, x)` key 오름차순 |
| `alternating` | 최소, 최대, 두 번째 최소, 두 번째 최대, ... |

핵심 질문은 단순한 test 정확도가 아니라 다음 두 가지다.

1. 학습 데이터가 정답 순서를 결정하기에 충분한 관계 정보를 제공했는가?
2. 모델이 직접 함께 본 값 쌍을 넘어 관계의 전이를 합성했는가?

## 2. 모델이 사전에 아는 것과 모르는 것

각 값 `0..n-1`은 scalar가 아니라 서로 독립적인 categorical embedding으로 표현된다. 모델 입력에는 숫자 거리, 대소 관계, 나머지 연산 결과가 직접 주어지지 않는다.

집합 인코더에는 위치 embedding이 없고 학습 때 각 입력 행을 섞는다. 따라서 입력 나열 순서는 정답을 알아내는 단서가 아니다. decoder는 이전 출력과 출력 위치를 사용해 causal하게 다음 값을 생성하고, cross-attention으로 인코딩된 입력 집합을 참조한다.

이 설계에서 값 사이의 순서 관계는 train 예제로부터 학습해야 한다. 그러므로 train에 없는 관계 때문에 답을 결정할 수 없는 경우와, 필요한 정보는 있지만 모델이 합성하지 못한 경우를 분리해야 한다.

## 3. 관계 그래프와 식별 가능성

한 train 조합의 target 순서는 그 안에 있는 값 쌍들의 directed order를 제공한다. 이 관계들을 모아 directed graph를 만들고 transitive closure를 계산하면 train으로부터 추론 가능한 순서를 얻을 수 있다.

예를 들어 train이 `a < b`와 `b < c`를 제공하면 `a < c`는 직접 함께 관측하지 않았어도 전이적으로 결정된다. 반대로 두 값 사이에 도달 경로가 없다면 해당 관계는 train만으로 결정되지 않는다.

`order_identifiable=true`는 이 closure가 해당 과제의 전체 target order를 유일하게 결정한다는 뜻이다. 식별 불가능한 test 실패를 알고리즘 일반화 실패로 해석하면 안 된다.

## 4. Relation-Complete Split

`relation-complete`는 세 과제에 필요한 전체 순서가 train 관계로 식별되도록 만든다.

1. numeric order `0,1,...,n-1`의 모든 인접 pair를 requirement로 만든다.
2. `(x % modulus, x)` 순서의 모든 인접 pair를 requirement로 만든다.
3. 두 requirement의 합집합을 덮도록 연속 window 후보에서 deterministic greedy basis를 구성한다.
4. basis 뒤에는 seeded random 조합을 채워 요청한 train count를 정확히 맞춘다.
5. train complement에서 별도 deterministic seed로 test를 선택한다.

전체 linear order에서는 모든 인접 관계를 알면 전이 폐쇄로 임의의 두 값 관계를 결정할 수 있다. `ascending`과 `alternating`은 같은 numeric underlying order를 공유하고, `mod`는 별도의 key order를 사용한다.

따라서 relation-complete split에서는 세 과제 모두 `order_identifiable=true`이고 test의 `unresolved`가 0이 된다. 요청한 train count가 최소 basis보다 작으면 실험 조건을 만족시킬 수 없어 생성이 실패한다.

`random` split은 같은 train count에서도 관계 누락이 생길 수 있다. relation-complete와 random을 함께 비교하면 학습 알고리즘의 문제와 데이터 identifiability의 문제를 구분할 수 있다.

## 5. Test Strata

각 test 조합에서 정답을 결정하는 데 필요한 모든 ordered pair를 train 관계와 비교한다.

| stratum | 정의 |
| --- | --- |
| `direct` | 필요한 모든 pair를 train의 어떤 조합에서 직접 함께 관측 |
| `transitive` | 직접 보지 않은 pair가 있지만 모든 관계가 transitive closure로 결정됨 |
| `unresolved` | closure로도 결정할 수 없는 pair가 하나 이상 존재 |

`direct` 정확도에는 직접 관측한 pair의 재사용만으로 풀 수 있는 예제가 포함된다. `transitive` 정확도는 관측 관계를 합성해야 하는 예제를 측정한다. `unresolved` 정확도는 모델이 데이터에 없는 규칙성을 우연히 또는 embedding 구조를 통해 알아낸 정도일 수 있으므로 다른 두 strata와 같은 의미로 해석할 수 없다.

relation-complete에서 핵심 지표는 `test_transitive_exact_acc`다. overall test 정확도만 보면 direct 예제 비율에 따라 일반화 능력이 과대평가될 수 있다.

## 6. 출력 제약의 의미

세 출력 모드는 모델이 해결해야 하는 문제의 범위를 바꾼다.

| 모드 | 구조적으로 보장되는 것 |
| --- | --- |
| `permutation` | 출력은 항상 입력의 순열이고 마지막 위치는 자동 결정 |
| `input-only` | 모든 출력이 입력 집합에 속함 |
| `free` | 보장 없음 |

`permutation`은 순서 관계 학습에 집중하는 유용한 ablation이지만 in-set accuracy와 set accuracy가 구조적으로 1이다. `input-only`도 in-set accuracy가 구조적으로 1이다. `free`에서는 값 선택, 집합 완성, 순서를 모두 모델이 해결해야 하므로 알고리즘 전체를 평가하는 기준으로 사용한다.

## 7. 지표 해석

| 지표 | 해석 |
| --- | --- |
| `loss`, `token_acc` | teacher-forced 다음 token 예측 성능 |
| `gen_in_set_token_acc` | 자유 생성 token이 입력 집합에 속하는 비율 |
| `set_acc` | 생성 전체가 순서를 무시할 때 입력 집합과 같은 비율 |
| `exact_acc` | 생성 전체가 target 순서와 같은 비율 |
| `test_transitive_exact_acc` | 전이 합성이 필요한 test의 완전 정답률 |
| `weight_norm` | 학습 중 parameter L2 norm 변화 |

teacher-forced token accuracy가 높아도 자유 생성에서는 초반 오류가 누적될 수 있다. 따라서 grokking 판정에는 `exact_acc`를 우선 사용하고, 실패 원인을 구분할 때 in-set과 set 지표를 함께 본다.

`grokking gap`은 test exact accuracy가 0.90에 처음 도달한 step을 train exact accuracy가 0.99에 처음 도달한 step으로 나눈 값이다. 값이 크면 train 암기와 test 일반화 사이의 지연이 길다는 뜻이다. 임계값에 도달하지 않은 run은 성공 run으로 집계되지 않는다.

## 8. 권장 비교 설계

1. `n`, `m`, `modulus`, 정확한 train count와 test count를 고정한다.
2. relation-complete와 random을 같은 data seed, model seed로 비교한다.
3. overall test exact와 transitive exact를 따로 확인한다.
4. metadata의 `order_identifiable`과 strata count를 먼저 확인한다.
5. count, weight decay와 최대 step을 바꾸되 유망한 설정은 model seed를 최소 3개 사용한다.

percentage보다 정확한 train count를 primary 축으로 사용하는 편이 조합 공간 크기가 다른 실험을 명확하게 비교할 수 있다. random 결과에서 unresolved 비율이 다르면 overall 정확도를 relation-complete 결과와 직접 같은 의미로 비교해서는 안 된다.

양수 batch size는 복원추출 횟수이므로 값을 키우면 같은 train example의 여러 입력 permutation을 한 optimizer step에서 평균한다. 이는 GPU workload만 바꾸는 것이 아니라 gradient noise와 step당 표본 수를 바꾼다. 따라서 batch size가 다른 run을 동일한 학습 trajectory로 간주하지 않고, 주 비교에서는 batch size도 고정한다.

## 9. Coverage의 역할

`elements`는 모든 값이 train에 나타났는지 확인한다. `pairs`는 임의의 값 쌍이 얼마나 직접 함께 관측됐는지 보여 준다. `adjacent_pairs`는 전체 target order를 연결하는 핵심 관계가 포함됐는지 보여 준다.

전체 pair coverage가 낮아도 adjacent 관계가 완전하면 transitive closure로 전체 order를 식별할 수 있다. 따라서 pair coverage 비율만으로 데이터 충분성을 판단하지 않고 `adjacent_pairs`, `order_identifiable`, `test_strata`를 함께 사용한다.

pair 수가 지나치게 클 때 구현은 부정확한 근삿값을 기록하지 않고 분석을 `skipped` 처리한다. 이 경우에도 adjacent coverage와 identifiability, strata는 별도로 계산된다.
