"""
영화 흥행 예측기
----------------
KOBIS 일별 박스오피스 데이터(kobis_daily.csv)와 영화 정보 데이터(kobis_movies.csv)를
결합해, 영화의 총 관객 수를 예측하는 다중 회귀 모델을 만드는 Streamlit 앱.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DAILY_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/kobis_daily.csv"
MOVIES_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/kobis_movies.csv"

FLOOR = 1000  # 예측 총 관객 수가 이보다 작으면 그래프 바닥에 붙여 표시

st.set_page_config(page_title="영화 흥행 예측기", layout="wide")
st.title("🎬 영화 흥행 예측기")
st.caption("KOBIS 일별 박스오피스 데이터 + 영화 정보 데이터로 총 관객 수를 예측합니다.")


# ---------------------------------------------------------------------------
# 데이터 불러오기
# ---------------------------------------------------------------------------
@st.cache_data
def load_data():
    daily = pd.read_csv(DAILY_URL, encoding="utf-8")
    movies = pd.read_csv(MOVIES_URL, encoding="utf-8")
    return daily, movies


try:
    daily, movies = load_data()
except Exception as e:  # noqa: BLE001
    st.error(f"데이터를 불러오는 중 문제가 발생했습니다: {e}")
    st.stop()

# ---------------------------------------------------------------------------
# 일별 박스오피스 데이터를 영화 단위로 집계
# ---------------------------------------------------------------------------
daily_agg = (
    daily.groupby("영화코드")
    .agg(
        daily_최고스크린수=("스크린수", "max"),
        daily_최고상영횟수=("상영횟수", "max"),
        daily_최고일일관객=("일관객", "max"),
        daily_최고순위=("순위", "min"),
        daily_평균순위=("순위", "mean"),
        daily_상영일수=("날짜", "count"),
    )
    .reset_index()
    .rename(columns={"영화코드": "movieCd"})
)

# ---------------------------------------------------------------------------
# 두 표 병합 — 영화 정보 표의 모든 영화를 그대로 사용 (left merge)
# ---------------------------------------------------------------------------
merged = movies.merge(daily_agg, on="movieCd", how="left")

count_cols = ["daily_최고스크린수", "daily_최고상영횟수", "daily_최고일일관객", "daily_상영일수"]
rank_cols = ["daily_최고순위", "daily_평균순위"]
merged[count_cols] = merged[count_cols].fillna(0)
merged[rank_cols] = merged[rank_cols].fillna(999)  # 일별 데이터에 없으면 순위를 매우 낮게 처리

base_num_cols = ["first_scrn", "first_show", "first_week_audi", "days_in_top10", "peak"]
for col in base_num_cols:
    if col in merged.columns:
        merged[col] = merged[col].fillna(0)

# ---------------------------------------------------------------------------
# 데이터 기준 기간 (일별 데이터의 날짜 범위)
# ---------------------------------------------------------------------------
daily_dates = pd.to_datetime(daily["날짜"].astype(str), format="%Y%m%d")
period_start = daily_dates.min().strftime("%Y-%m-%d")
period_end = daily_dates.max().strftime("%Y-%m-%d")

# ---------------------------------------------------------------------------
# 특성(변수) 후보
# ---------------------------------------------------------------------------
FEATURE_OPTIONS = {
    "peak": "개봉 시 성수기 여부 (1·7·12월 개봉)",
    "first_scrn": "첫 관측일 스크린수",
    "first_show": "첫 관측일 상영횟수",
    "first_week_audi": "첫 주 관객수",
    "days_in_top10": "10위권 유지일수 (영화정보 표 기준)",
    "genre": "장르",
    "nation": "국가",
    "daily_최고스크린수": "최고 스크린수 (일별 데이터)",
    "daily_최고상영횟수": "최고 상영횟수 (일별 데이터)",
    "daily_최고일일관객": "최고 일일 관객수 (일별 데이터)",
    "daily_최고순위": "최고 순위 (일별 데이터, 값이 작을수록 높은 순위)",
    "daily_평균순위": "평균 순위 (일별 데이터)",
    "daily_상영일수": "10위권 등장일수 (일별 데이터 기준)",
}
CATEGORICAL = {"genre", "nation"}
DEFAULT_ON = {"first_scrn", "first_week_audi", "peak"}

st.sidebar.header("예측에 사용할 변수")
st.sidebar.caption(
    "영화코드 순으로 정렬한 뒤, 10편마다 앞의 3편을 시험용으로, "
    "나머지 7편을 학습용으로 나눕니다."
)
selected_features = [
    col
    for col, label in FEATURE_OPTIONS.items()
    if st.sidebar.checkbox(label, value=(col in DEFAULT_ON))
]

if not selected_features:
    st.warning("왼쪽 사이드바에서 예측에 사용할 변수를 하나 이상 선택해 주세요.")
    st.stop()

# ---------------------------------------------------------------------------
# 영화코드 순 정렬 + 학습/시험 분리 (10편마다 앞 3편 = 시험용)
# ---------------------------------------------------------------------------
merged_sorted = merged.sort_values("movieCd").reset_index(drop=True)
merged_sorted["그룹내순번"] = merged_sorted.index % 10
is_test = merged_sorted["그룹내순번"] < 3

train_df = merged_sorted[~is_test].copy()
test_df = merged_sorted[is_test].copy()

# ---------------------------------------------------------------------------
# 특성 테이블 구성 (범주형은 전체 데이터 기준으로 원-핫 인코딩 후 분리)
# ---------------------------------------------------------------------------
feature_df = merged_sorted[selected_features].copy()
cat_features = [c for c in selected_features if c in CATEGORICAL]
if cat_features:
    feature_df = pd.get_dummies(feature_df, columns=cat_features, dummy_na=False)

X_all = feature_df.fillna(0)
y_all = merged_sorted["total_audi"]

X_train, X_test = X_all[~is_test], X_all[is_test]
y_train, y_test = y_all[~is_test], y_all[is_test]

# ---------------------------------------------------------------------------
# 다중 회귀 모델 학습 및 예측
# ---------------------------------------------------------------------------
model = LinearRegression()
model.fit(X_train, y_train)
pred_test = model.predict(X_test)

r2 = r2_score(y_test, pred_test)
mae = mean_absolute_error(y_test, pred_test)
rmse = mean_squared_error(y_test, pred_test) ** 0.5
valid_ratio_mask = y_test != 0
mape = (
    (np.abs((y_test[valid_ratio_mask] - pred_test[valid_ratio_mask.values]) / y_test[valid_ratio_mask])).mean()
    * 100
)

# ---------------------------------------------------------------------------
# 화면 표시 — 학습/시험 편수와 기준 기간
# ---------------------------------------------------------------------------
c1, c2, c3 = st.columns(3)
c1.metric("학습에 사용한 영화 수", f"{len(train_df):,} 편")
c2.metric("점수를 잰 영화 수", f"{len(test_df):,} 편")
c3.metric("데이터 기준 기간", f"{period_start} ~ {period_end}")

st.subheader("예측 성능")
m1, m2, m3 = st.columns(3)
m1.metric("결정계수 (R²)", f"{r2:.3f}")
m2.metric("평균 절대 오차 (MAE)", f"{mae:,.0f} 명")
m3.metric("평균 절대 비율 오차 (MAPE)", f"{mape:.1f} %")
st.caption(f"평균 제곱근 오차(RMSE): {rmse:,.0f} 명")

# ---------------------------------------------------------------------------
# 산점도 — 실제 vs 예측 (로그-로그, 1,000명 미만 예측은 바닥에 고정)
# ---------------------------------------------------------------------------
pred_clamped = np.clip(pred_test, a_min=FLOOR, a_max=None)
n_clamped = int((pred_test < FLOOR).sum())

actual_min = max(y_test.min(), 1)
axis_min = min(actual_min, pred_clamped.min()) * 0.8
axis_max = max(y_test.max(), pred_clamped.max()) * 1.2

hover_names = test_df["movieNm"] if "movieNm" in test_df.columns else [""] * len(test_df)

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=y_test,
        y=pred_clamped,
        mode="markers",
        name="시험용 영화",
        text=hover_names,
        hovertemplate="%{text}<br>실제: %{x:,.0f}명<br>예측: %{y:,.0f}명<extra></extra>",
        marker=dict(size=9, color="royalblue", opacity=0.7),
    )
)
fig.add_trace(
    go.Scatter(
        x=[axis_min, axis_max],
        y=[axis_min, axis_max],
        mode="lines",
        name="예측 = 실제 (기준선)",
        line=dict(dash="dash", color="gray"),
    )
)
fig.update_xaxes(type="log", title="실제 총 관객 수 (명, 로그 스케일)", range=[np.log10(axis_min), np.log10(axis_max)])
fig.update_yaxes(type="log", title="예측 총 관객 수 (명, 로그 스케일)", range=[np.log10(axis_min), np.log10(axis_max)])
fig.update_layout(
    title="시험용 영화: 실제 총 관객 수 vs 예측 총 관객 수",
    height=600,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)

st.plotly_chart(fig, use_container_width=True)
st.caption(
    f"예측 총 관객 수가 {FLOOR:,}명보다 작게 나와 그래프 바닥에 붙여 표시한 영화: {n_clamped}편"
)

# ---------------------------------------------------------------------------
# 참고 — 회귀 계수 및 시험용 데이터 상세
# ---------------------------------------------------------------------------
with st.expander("회귀 계수 보기"):
    coef_df = pd.DataFrame(
        {"변수": X_train.columns, "계수": model.coef_}
    ).sort_values("계수", key=abs, ascending=False)
    st.dataframe(coef_df, use_container_width=True, hide_index=True)
    st.caption(f"절편(intercept): {model.intercept_:,.0f}")

with st.expander("시험용 영화 상세 결과 보기"):
    detail_df = test_df[["movieCd", "movieNm", "total_audi"]].copy()
    detail_df["예측 총 관객 수"] = np.round(pred_test).astype(int)
    detail_df = detail_df.rename(columns={"movieCd": "영화코드", "movieNm": "영화명", "total_audi": "실제 총 관객 수"})
    st.dataframe(detail_df, use_container_width=True, hide_index=True)
