# 🎬 아이템 기반 협업 필터링 영화 추천 시스템

## 📌 프로젝트 개요
이 프로젝트는 **아이템 기반 협업 필터링(Item-Based Collaborative Filtering)** 기법을 활용하여 사용자에게 맞춤형 영화 추천을 제공하는 Streamlit 기반 웹 애플리케이션입니다.

사용자의 영화 평점 데이터를 분석하고, 유사한 선호도를 가진 영화 간의 상관관계를 계산하여 추천 목록을 제공합니다.

---

## 📂 사용 데이터

### 1. `movie_review.csv`
- 사용자(user_id)가 영화(item_id)에 남긴 평점(rating) 정보가 담긴 데이터셋

| user_id | item_id | rating |
|---------|---------|--------|
|   1     |   10    |  4.0   |
|   1     |   20    |  5.0   |

### 2. `movie_title.csv`
- 영화 ID(item_id)와 영화 제목(title)을 매핑한 데이터셋

| item_id | title           |
|---------|------------------|
|   10    | The Matrix       |
|   20    | Inception        |

---

## 🧠 추천 시스템 개념: 아이템 기반 협업 필터링

**아이템 기반 협업 필터링(Item-Based CF)** 은 사용자가 평가한 아이템(영화)과 **유사한 아이템**을 찾아 추천하는 방식입니다.

> 예: 사용자가 '매트릭스'에 높은 평점을 줬다면, '인셉션'처럼 유사한 영화를 추천

이 프로젝트에서는 다음과 같은 방식으로 추천이 이루어집니다:

1. **피봇 테이블**을 통해 유저-아이템 평점 행렬 생성
2. 각 아이템 간 **상관계수(Pearson)** 기반 유사도 계산
3. 사용자가 높게 평가한 영화와 유사한 영화 중, 조건을 만족하는 항목 추천

---

## 🔧 사용 방법

- `movie_review.csv`, `movie_title.csv` 파일을 같은 디렉토리에 위치시킵니다.
- Streamlit 앱을 실행하면 사용자 ID 입력에 따라 개인화된 추천 결과가 표시됩니다.

---

## 🚀 주요 기능

- 유저-아이템 평점 행렬 생성 (`pivot_table` 활용)
- 아이템 간 상관계수 계산 (`corr()` 활용)
- 사용자 기반 평점 필터링 (`rating_threshold`)
- 리뷰 수 필터링 (`min_review_count`)
- 개인화된 추천 결과 출력 (Streamlit UI)

---

## 🧾 추천 로직 요약

1. 사용자가 4점 이상 평가한 영화 선택 (`rating_threshold=4.0`)
2. 각 영화의 유사 영화 탐색 (상관계수 기반)
3. 리뷰 수가 너무 적은 영화 제거 (`min_review_count`)
4. 이미 본 영화는 추천 목록에서 제외
5. 유사도 점수를 누적해 상위 N개의 영화 추천

> 상관계수 행렬에 포함되지 않은 영화에 대해서는 리뷰 수가 많고 평점이 높은 영화를 기본 추천

---

## 🖥️ 개발 환경

- Python 3.10
- Streamlit
- pandas

---

## 📌 예시 화면

- 사용자 ID 입력
- 해당 유저의 영화 평가 목록 표시
- 추천된 영화 리스트 출력

---

## 📬 문의
- 이메일: marurun66@gmail.com
- GitHub: [https://github.com/marurun66](https://github.com/marurun66)
