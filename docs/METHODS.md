# 검증 방법 (Methods)

기호: 관측 풍속 $o_i$, 모델 풍속 $m_i$, 짝의 수 $n$. 풍속 단위는 m s⁻¹, 풍향은 바람이 불어오는 방향(°, 북=0, 시계방향).

## 1. 자료 짝짓기

1. 관측 시각을 UTC로 바꿉니다(기상청 자료는 KST).
2. 모델 격자의 u, v를 지점 위치로 내삽합니다.
   - 1차원 위경도 격자: 쌍선형 내삽(`bilinear`) 또는 최근접(`nearest`)
   - 2차원 위경도 곡선 격자: 가까운 4개 격자점의 거리 제곱 역수 가중(`idw4`) 또는 최근접
3. 풍속과 풍향은 내삽한 u, v로 계산합니다. 벡터를 내삽한 뒤 크기를 구하므로, 풍향 변화가 큰 곳에서는
   풍속을 직접 내삽한 값보다 약간 작게 나올 수 있습니다.
4. `common_sample: true`이면 모든 모델과 관측이 함께 있는 (지점, 시각) 짝만 남깁니다.

기상청 정시 관측 풍속은 정시 전 10분 평균이고, 모델 값은 그 시각의 순간값 또는 적분 시간 간격 값입니다.
이 대표성 차이는 오차에 포함됩니다.

## 2. 연속 점수

| 지표 | 정의 |
|---|---|
| 편차 (bias) | $\frac{1}{n}\sum (m_i - o_i)$ |
| 평균절대오차 (MAE) | $\frac{1}{n}\sum \lvert m_i - o_i \rvert$ |
| 평균제곱근오차 (RMSE) | $\sqrt{\frac{1}{n}\sum (m_i - o_i)^2}$ |
| 상관계수 (corr) | 피어슨 상관 |
| 표준편차 비 (std_ratio) | $\sigma_m / \sigma_o$. 1보다 작으면 변동을 약하게 모의 |
| 분위수 편차 (pXX_bias) | $Q_m(p) - Q_o(p)$, $p$ = 0.5, 0.9, 0.95, 0.99 |
| 풍향 편차 (dir_bias) | 각도 차 $d_i = ((\theta_m - \theta_o + 180) \bmod 360) - 180$의 원형 평균 |
| 풍향 MAE (dir_mae) | $\frac{1}{n}\sum \lvert d_i \rvert$ |
| 벡터 RMSE | $\sqrt{\frac{1}{n}\sum [(u_m-u_o)^2 + (v_m-v_o)^2]}$ |

풍향 점수는 관측과 모델 풍속이 모두 1 m s⁻¹을 넘는 짝만 씁니다(정온 제외).

## 3. 강풍 사건 점수

사건을 "풍속 ≥ 임계값"으로 정의하고 2×2 분할표를 만듭니다. 적중 $a$, 오경보 $b$, 놓침 $c$, 정부정 $d$.

| 지표 | 정의 | 최적값 |
|---|---|---|
| 탐지확률 (POD) | $a/(a+c)$ | 1 |
| 오경보율 (FAR) | $b/(a+b)$ | 0 |
| 임계성공지수 (CSI) | $a/(a+b+c)$ | 1 |
| 빈도 편차 (FBI) | $(a+b)/(a+c)$ | 1 |
| 공평임계성공지수 (ETS) | $(a-a_r)/(a+b+c-a_r)$, $a_r=(a+b)(a+c)/n$ | 1 |

임계값은 설정 파일의 `thresholds_ms`로 정합니다. 기본 예시는 10, 14, 20 m s⁻¹입니다.
14 m s⁻¹은 육상 강풍주의보 평균풍속 기준을 참고한 값입니다. 정확한 특보 기준은 기상청 고시를 확인하세요.

## 4. 기준 자료 대비 기술점수

$$\mathrm{MSESS} = 1 - \frac{\mathrm{MSE}_{\text{model}}}{\mathrm{MSE}_{\text{reference}}}$$

0보다 크면 기준 자료(예: ERA5)보다 낫고, 1이면 완벽합니다.

시간 자기상관 때문에 단순 부트스트랩은 신뢰구간을 좁게 추정합니다. 그래서 길이 $L$(기본 24시간)의
**이동 블록 부트스트랩**으로 제곱오차 계열을 다시 뽑아 MSESS 분포를 만듭니다.
같은 블록 색인을 모델과 기준 자료에 함께 적용해 두 오차의 상관을 유지합니다.

- `msess_lo95`, `msess_hi95`: 2.5, 97.5 백분위
- `p_not_better`: 부트스트랩 표본 중 MSESS ≤ 0인 비율. 단측 검정의 근사 유의확률로 읽을 수 있습니다.

여러 지점을 합쳐 계산할 때는 (지점, 시각) 순으로 이어 붙이므로 블록이 지점 경계를 넘는 경우가 일부 생깁니다.
사례 수가 적으면 지점별 결과를 함께 보고하는 것을 권합니다.

## 5. 한계

- 지점 관측은 격자 평균과 공간 대표성이 다릅니다. 특히 해안, 산지, 섬 끝처럼 지형 효과가 큰 곳에서 차이가 큽니다.
- 짧은 단일 사례의 점수는 불확실성이 큽니다. 신뢰구간을 함께 보고합니다.
- 로그 법칙 높이 보정은 중립 성층과 균일한 지면을 가정합니다.

## 참고 문헌

- Mardani, M., et al. (2025). Residual corrective diffusion modeling for km-scale atmospheric downscaling. *Communications Earth & Environment*, 6. https://doi.org/10.1038/s43247-025-02042-5
- Wilks, D. S. (2019). *Statistical Methods in the Atmospheric Sciences* (4th ed.). Elsevier.
- Künsch, H. R. (1989). The jackknife and the bootstrap for general stationary observations. *The Annals of Statistics*, 17(3), 1217–1241.
- Hersbach, H., et al. (2020). The ERA5 global reanalysis. *Quarterly Journal of the Royal Meteorological Society*, 146, 1999–2049.
