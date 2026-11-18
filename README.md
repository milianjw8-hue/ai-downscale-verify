# ai-downscale-verify (dsverify)

AI 다운스케일링 지상풍을 **독립 관측(ASOS/AWS)**으로 검증하는 Python 도구입니다.
CorrDiff 같은 생성형 AI 다운스케일링 결과, ERA5 같은 저해상도 재분석, 수치예보 모델 결과를
같은 관측 지점과 같은 시각에서 비교하고, 저해상도 자료 대비 "추가 가치(added value)"가
통계적으로 유의한지 판단합니다.

*English summary below.*

## 왜 필요한가

AI 다운스케일링 모델은 보통 고해상도 **수치모델 출력**을 정답으로 학습하고 검증합니다.
그러면 수치모델의 계통오차까지 함께 학습할 수 있고, 검증 점수도 실제 대기와의 차이를
보여 주지 못합니다. 이 도구는 학습에 쓰지 않은 **지상 관측**을 기준으로 삼아 다음 질문에 답합니다.

- 다운스케일링 결과가 관측 풍속의 크기와 변동을 더 잘 재현하는가? (편차, RMSE, 상관, 표준편차 비)
- 강풍 꼬리(상위 분위수)와 최댓값을 살려내는가? (p90·p95·p99 편차, Q-Q 그림)
- 강풍 사건을 맞히는가? (임계값별 POD, FAR, CSI, 빈도 편차)
- 저해상도 자료보다 나은 정도가 우연이 아닌가? (MSE 기술점수와 이동 블록 부트스트랩 95% 신뢰구간)
- 앙상블이라면 불확실성을 제대로 표현하는가? (CRPS, 퍼짐-오차 비, 순위 히스토그램, 브라이어 점수와 신뢰도 도표)

## 설치

```bash
git clone https://github.com/milianjw8-hue/ai-downscale-verify.git
cd ai-downscale-verify
pip install -e ".[dev]"            # 자료 내려받기까지 쓰려면 ".[dev,download]"
```

conda를 쓰면 `conda env create -f environment.yml` 후 `pip install -e .`를 실행합니다.

## 바로 해 보기 (합성 예제)

실제 자료 없이 전 과정을 확인할 수 있도록 **합성(가짜) 제주 강풍 사례**를 만드는 스크립트를 넣었습니다.

```bash
python scripts/make_synthetic_demo.py          # data/demo/ 생성 (약 1.5 MB)
dsverify run configs/example_jeju_synthetic.yaml
```

결과는 `outputs/jeju_synthetic/`에 저장됩니다.

| 파일 | 내용 |
|---|---|
| `summary.md` | 전체 지점 통합 점수, 강풍 사건 점수, 기술점수 요약 |
| `scores_station.csv`, `scores_all.csv` | 연속 점수(지점별, 통합) |
| `categorical_station.csv`, `categorical_all.csv` | 임계값별 분할표 점수 |
| `skill_vs_reference.csv` | 기준 자료 대비 MSE 기술점수와 신뢰구간 |
| `probabilistic_all.csv` | 모든 모델의 CRPS(결정론 모델은 MAE와 같음), 앙상블의 공정 CRPS·퍼짐-오차 비·순위 평탄도·브라이어 점수 |
| `reliability_table.csv` | 임계값별 신뢰도 도표 자료 |
| `matched.csv` | 관측-모델 짝 자료(재분석용) |
| `figures/` | 지점별 시계열, 산점도, Q-Q 그림, 지점별 점수 막대그림 |

합성 예제의 점수는 도구 동작을 확인하는 용도입니다. 실제 모델 성능을 뜻하지 않습니다.

## 실제 사례에 적용

1. **ERA5 내려받기** (Copernicus CDS 계정과 `~/.cdsapirc` 필요)
   ```bash
   python scripts/download_era5.py --start 2025-04-11 --end 2025-04-12 \
       --area 34.0 125.8 33.0 127.2 --out data/era5/era5_u10v10_20250411-12.nc
   ```
2. **ASOS 시간자료 준비**: 기상자료개방포털(data.kma.go.kr)에서 CSV를 내려받거나,
   공공데이터포털 API로 받습니다(`DATA_GO_KR_KEY` 환경변수 필요).
   ```bash
   python scripts/fetch_asos_hourly.py --stations 184 185 188 189 \
       --start "2025-04-11 00" --end "2025-04-13 23" --out data/obs/asos_20250411-13.json
   ```
3. **지점 정보** `stations.csv` 작성: `station,name,lat,lon,height_m`.
   좌표와 풍속계 높이는 기상자료개방포털 지점정보에서 확인합니다.
4. `configs/template_real_case.yaml`을 복사해 경로와 변수 이름을 고친 뒤 실행합니다.
   ```bash
   dsverify run configs/my_case.yaml
   ```

### 설정에서 주의할 점

- **시간대**: 기상청 관측은 KST, 모델은 UTC입니다. `obs.timezone: Asia/Seoul`로 두면 자동 변환합니다.
- **ERA5 시간 이름**: 새 CDS에서 받은 NetCDF는 시간 좌표가 `valid_time`입니다.
- **격자 형태**: 위경도가 1차원이면 `bilinear`, 2차원(곡선 격자)이면 `idw4`나 `nearest`를 씁니다.
- **영역 밖 지점**: 가장 가까운 격자점까지 거리가 `max_dist_km`를 넘으면 그 지점은 제외합니다.
- **공정 비교**: `common_sample: true`이면 모든 모델이 같은 (지점, 시각) 짝으로만 검증됩니다.
- **앙상블**: 모델 설정에 `member_dim: ensemble`을 넣으면(Earth2Studio 앙상블 출력의 차원 이름) 구성원별로 지점 값을 뽑습니다. 결정론 점수는 구성원 풍속 평균으로, 확률 점수는 전체 구성원으로 계산합니다.
- **풍속계 높이**: 10 m가 아닌 지점은 `height_m`를 채우고 `adjust_to_10m: true`로 로그 법칙 보정을
  할 수 있습니다. 1차 보정이므로 결과 해석에 유의합니다.

## 공개 범위와 자료 정책

- 이 저장소에는 **코드와 합성 예제만** 들어 있습니다.
- 현업 수치모델 출력, 기관 내부 AI 산출물, 비공개 관측자료는 저장소에 넣지 않습니다.
  설정 파일에서 저장소 밖 경로를 가리키도록 합니다(`.gitignore`가 `data/`, `outputs/`, `*.nc`를 제외).
- ERA5는 Copernicus 이용 조건을, 기상청 공개자료는 공공누리 조건을 따릅니다. 결과를 공개할 때 출처를 밝힙니다.

## 방법

지표 정의와 부트스트랩 방법은 [`docs/METHODS.md`](docs/METHODS.md)에 정리했습니다.

## 시험

```bash
pytest
```

GitHub Actions가 푸시마다 시험과 합성 예제 전체 실행을 자동으로 돌립니다.

## 인용

`CITATION.cff`를 참고하세요. 릴리스를 Zenodo와 연동하면 DOI를 받을 수 있습니다.

## 라이선스

MIT. [`LICENSE`](LICENSE) 참고.

---

## English summary

`dsverify` verifies AI-downscaled near-surface wind (e.g. NVIDIA CorrDiff output) against
**independent station observations** (KMA ASOS/AWS), side by side with a coarse reanalysis
(ERA5) and optionally an NWP model. It handles regular and curvilinear grids, KST/UTC
alignment, common-sample comparison, continuous scores (bias, MAE, RMSE, correlation,
standard-deviation ratio, tail-quantile bias, direction error, vector RMSE), strong-wind
contingency scores (POD, FAR, CSI, FBI, ETS) an MSE skill score against a reference
with a moving-block bootstrap confidence interval, and for ensembles (e.g. CorrDiff samples)
the CRPS (plain and fair), spread-skill ratio, rank histograms and Brier scores with
reliability diagrams.

```bash
pip install -e ".[dev]"
python scripts/make_synthetic_demo.py
dsverify run configs/example_jeju_synthetic.yaml
```

The repository contains code and synthetic data only. See `docs/METHODS.md` for definitions.
