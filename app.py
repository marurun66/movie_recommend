import streamlit as st
import pandas as pd
st.set_page_config(
    page_icon="🎬",
    page_title="아이템 기반 협업 필터링을 통한 개인화 맞춤형 영화 추천 시스템", 
    layout="wide"
)

def main():

    # 데이터 불러오기 (예시 경로)
    df_review = pd.read_csv("movie_review.csv")
    df_title = pd.read_csv("movie_title.csv")
    df_merged = pd.merge(df_review, df_title, on='item_id', how='left')

    st.title("🎬 아이템 기반 협업 필터링을 통한 개인화 맞춤형 영화 추천 시스템")
    st.subheader("아이템 기반 협업 필터링이란?")
    st.markdown("""아이템 기반 협업 필터링(Item-Based Collaborative Filtering)은 추천 시스템의 한 방식으로,  
                아이템 간의 유사도를 계산하여 사용자가 좋아할 만한 다른 아이템(상품, 영화 등)을 추천하는 방식입니다.  
                """)
    st.header("1️⃣ 데이터 소개")
    st.markdown("""
    - **df_review**: 사용자들이 영화에 남긴 평점 정보
    - **df_title**: 영화 ID에 해당하는 제목 정보""")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ```python
        # df_review 예시
        df_review.head()
        ```""")
    with col2:
        st.dataframe(df_review.head())
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ```python
        # df_title 예시
        df_title.head()
        ```""")
    with col2:
        st.dataframe(df_title.head())

    st.header("2️⃣ 리뷰와 영화 데이터프레임 병합 : df_review + df_title")
    st.markdown("""
    `df_review`와 `df_title`을 **merge(병합)** 하여 `df_merged`를 생성합니다.  
    `'item_id'`기준으로 병합하여 유저 ID, 영화 ID, 평점, 영화 제목을 한 프레임 내에서 확인가능합니다.
    """)
    st.code("""
df_merged = pd.merge(df_review, df_title, on='item_id', how='left')
df_merged.head()
    """, language='python')
    st.dataframe(df_merged.head())

    st.header("3️⃣ 유저-아이템 평점 행렬 생성")
    st.markdown("""
병합한 `df_merged`를 **피봇팅**하여 유저-아이템 평점 행렬을 생성합니다.  
    """)
    st.success("""
**행(index)**: **user_id** // **열(columns)**: **item_id (영화 ID)** // **값(values)**: **rating (유저가 준 평점)**
    """)

    df_matrix_id = df_merged.pivot_table(index='user_id', columns='item_id', values='rating')

    st.code("""
df_matrix_id = df_merged.pivot_table(index='user_id', columns='item_id', values='rating')
    """, language='python')
    st.markdown("이 행렬은 사용자(user)가 각 영화(item_id)에 남긴 평점을 행렬 형태로 변환한 것입니다. NaN은 평가하지 않은 영화입니다.")
    st.dataframe(df_matrix_id)

    st.header("4️⃣ 상관계수(corr)를 이용한 아이템 간 유사도 계산")
    item_corr = df_matrix_id.corr()
    st.code("""
item_corr = df_matrix_id.corr()
""", language='python')
    st.markdown("`corr()`를 사용하면 **아이템 간의 벡터 유사성(Pearson 상관계수)**를 구할 수 있습니다. 이 값은 **영화 간의 선호도 패턴 유사성**을 나타냅니다.")
    st.dataframe(item_corr)
    st.markdown("""
하나의 영화는 자기 자신과의 상관계수 값이 항상 1입니다.  
즉, 상관계수 행렬에서 대각선에 위치한 값들은 모두 1이며, 이는 동일한 영화 간 비교이기 때문입니다.  
일반적으로 1에 가까울수록 유사도가 높다고 해석할 수 있습니다.
""")

    st.header("5️⃣ 사용자 맞춤 추천 알고리즘")
    st.markdown("""
    추천 과정은 다음과 같습니다:

    1. 해당 사용자가 평점한 영화 중 평점이 4점 이상인 이상인 영화들을 추출합니다. `rating_threshold=4.0` 
    2. 해당 영화들과 상관계수가 높은 영화를 모두 모읍니다.
    3. 리뷰 수(`min_review_count`)가 너무 적은 영화는 제외합니다.
    4. 이미 사용자가 본 영화는 제외합니다.
    5. 유사도 점수를 누적하여 정렬한 뒤, 상위 N개의 영화를 추천합니다.
    """)
    def recommend_for_user_corr(user_id, df_review, df_title, item_corr, rating_threshold=4.0, min_review_count=10, top_n=5):
        
        # 해당 사용자가 평점한 영화 중 평점이 4점 이상인 이상인 영화들을 추출 (rating_threshold=4.0)
        user_ratings = df_review[df_review['user_id'] == user_id]

        # 평점이 10개 이상인 영화 ID 목록 추출 (min_review_count=10)
        high_rated = user_ratings[user_ratings['rating'] >= rating_threshold]['item_id'].tolist()

        # 사용자가 이미 본 영화 ID 목록 저장
        seen_movies = user_ratings['item_id'].tolist()

        # 전체 영화별 리뷰 수 계산
        review_counts = df_review.groupby('item_id').size()

        # 추천 후보 영화들의 누적 유사도 점수를 저장할 딕셔너리 만들기
        candidate_scores = {}

        for movie_id in high_rated:
            # 상관계수 행렬에 해당 영화가 없으면 전체 리뷰가 많고 평점이 높은 영화를 추천
            if movie_id not in item_corr:
                popular_movies = (
                    df_review.groupby('item_id')['rating']
                    .mean()
                    .reset_index()
                    .merge(review_counts.rename('review_count'), on='item_id')
                )
                # 리뷰 수 조건 만족 + 이미 본 영화 제외
                popular_movies = popular_movies[
                    (popular_movies['review_count'] >= min_review_count) &
                    (~popular_movies['item_id'].isin(seen_movies))
                ]
                # 평점 기준 정렬 후 점수를 누적
                for _, row in popular_movies.iterrows():
                    candidate_scores[row['item_id']] = candidate_scores.get(row['item_id'], 0) + row['rating']
                continue

            # 해당 영화와 유사한 다른 영화들 (자기 자신, NaN 제외, 유사도 내림차순 정렬)
            similar_movies = item_corr[movie_id].drop(movie_id).dropna().sort_values(ascending=False)

            # 유사 영화를 for문으로 반복해서 추천 후보 영화에 추가하기
            for sim_movie_id, score in similar_movies.items():
                if sim_movie_id in seen_movies:
                    continue # 이미 본 영화는 제외
                if review_counts.get(sim_movie_id, 0) < min_review_count:
                    continue # 리뷰 수가 너무 적은 영화는 제외
                # 유사도 점수를 누적
                candidate_scores[sim_movie_id] = candidate_scores.get(sim_movie_id, 0) + score

        # 누적 유사도 기준으로 정렬 후 상위 top_n개의 영화 ID 추출
        sorted_candidates = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
        top_movie_ids = [movie_id for movie_id, _ in sorted_candidates[:top_n]]

        # 추천된 영화 ID에 해당하는 제목 정보 반환
        return df_title[df_title['item_id'].isin(top_movie_ids)]

    st.code("""
def recommend_for_user_corr(user_id, df_review, df_title, item_corr, rating_threshold=4.0, min_review_count=10, top_n=5):
    
    # 해당 사용자가 평점한 영화 중 평점이 4점 이상인 이상인 영화들을 추출 (rating_threshold=4.0)
    user_ratings = df_review[df_review['user_id'] == user_id]

    # 평점이 10개 이상인 영화 ID 목록 추출 (min_review_count=10)
    high_rated = user_ratings[user_ratings['rating'] >= rating_threshold]['item_id'].tolist()

    # 사용자가 이미 본 영화 ID 목록 저장
    seen_movies = user_ratings['item_id'].tolist()

    # 전체 영화별 리뷰 수 계산
    review_counts = df_review.groupby('item_id').size()

    # 추천 후보 영화들의 누적 유사도 점수를 저장할 딕셔너리 만들기
    candidate_scores = {}

    for movie_id in high_rated:
        # 상관계수 행렬에 해당 영화가 없으면 전체 리뷰가 많고 평점이 높은 영화를 추천
        if movie_id not in item_corr:
            popular_movies = (
                df_review.groupby('item_id')['rating']
                .mean()
                .reset_index()
                .merge(review_counts.rename('review_count'), on='item_id')
            )
            # 리뷰 수 조건 만족 + 이미 본 영화 제외
            popular_movies = popular_movies[
                (popular_movies['review_count'] >= min_review_count) &
                (~popular_movies['item_id'].isin(seen_movies))
            ]
            # 평점 기준 정렬 후 점수를 누적
            for _, row in popular_movies.iterrows():
                candidate_scores[row['item_id']] = candidate_scores.get(row['item_id'], 0) + row['rating']
            continue

        # 해당 영화와 유사한 다른 영화들 (자기 자신 제외)
        similar_movies = item_corr[movie_id].drop(movie_id).dropna().sort_values(ascending=False)

        # 유사 영화를 for문으로 순회하며 추천 후보 영화에 추가하기
        for sim_movie_id, score in similar_movies.items():
            # 이미 사용자가 본 영화는 제외
            if sim_movie_id in seen_movies:
                continue
            # 리뷰 수가 너무 적은 영화는 제외
            if review_counts.get(sim_movie_id, 0) < min_review_count:
                continue
            # 유사도 점수를 누적
            candidate_scores[sim_movie_id] = candidate_scores.get(sim_movie_id, 0) + score

    # 누적 유사도 기준으로 정렬 후 상위 top_n개의 영화 ID 추출
    sorted_candidates = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
    top_movie_ids = [movie_id for movie_id, _ in sorted_candidates[:top_n]]

    # 추천된 영화 ID에 해당하는 제목 정보 반환
    return df_title[df_title['item_id'].isin(top_movie_ids)]
    """, language='python')

    st.header("🍿 아이템 기반 유저별 추천 영화")

    user_input = st.number_input("사용자 ID를 입력하세요:", min_value=1, step=1)

    if user_input in df_review['user_id'].unique():
        st.markdown(f"""
    유저아이디 : {user_input}가 평점을 남긴 영화들 확인
        """)
        st.dataframe(df_merged[df_merged['user_id'] == user_input])
        recommended_movies = recommend_for_user_corr(user_input, df_review, df_title, item_corr, rating_threshold=4.0, min_review_count=1, top_n=5)

        st.markdown("""
    상관계수를 이용한 추천 결과
                    """)
        st.dataframe(recommended_movies)
    else:
        st.warning("해당 사용자 ID는 데이터에 존재하지 않습니다.")



    st.header("6️⃣ 정리")
    st.success("""
    - `merge()`와 `pivot_table`, `corr()`를 활용해 영화 간 유사도를 계산했습니다.
    - 사용자가 좋아한 영화와 유사한 영화 중, 리뷰 수가 충분하고 아직 보지 않은 영화를 추천합니다.
    - 이 방식은 콘텐츠 기반이 아닌, 사용자 행동 기반의 아이템 협업 필터링 방식입니다.
    """)

if __name__ == "__main__":
    main()
