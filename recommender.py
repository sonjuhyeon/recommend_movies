import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix # COO 형식의 행렬을 만들기 위해 사용
from implicit.als import AlternatingLeastSquares # 알고리즘 사용을 위해 import
import pickle # 모델 저장을 위해 import
import sys
import json
import threadpoolctl
threadpoolctl.threadpool_limits(1, "blas") # Python에서 실행되는 과학 계산 라이브러리들이 사용하는 다중 스레드(multithreading)를 제어하기 위한 함수 호출
# threadpool_limits(1, "blas")는 blas 라이브러리가 사용할 수 있는 스레드 수를 1로 제한한다는 의미입니다. 이렇게 하면 병렬 처리 대신 단일 스레드로 작업을 처리하게 됩니다.
# 병렬화된 연산이 오히려 성능을 저하시킬 수 있는 상황에서, 다중 스레드를 제한하여 더 나은 성능을 기대할 수 있습니다.
from scipy.sparse import coo_matrix, csr_matrix # coo_matrix : COO 형식의 행렬을 만들기 위해 사용, csr_matrix : CSR 형식의 행렬을 만들기 위해 사용

saved_model_fname = "model/finalized_model.sav"
data_fname = 'data/ratings.csv'
item_fname = 'data/movie_final.csv'
weight = 10 # 가중치 값 설정 (10으로 설정) : 사용자의 평점을 가중치로 사용하여 모델을 학습

def model_train():
  ratings_df = pd.read_csv(data_fname)
  ratings_df["userId"] = ratings_df["userId"].astype("category")
  ratings_df["movieId"] = ratings_df["movieId"].astype("category")

  # userid와 movieid를 category 형태로 변환
  # 어떤 유저가 어떤 영화에 얼마의 평점을 주었는지 행렬 형태로 표현해 주는 함수
  # 참조 : https://leebaro.tistory.com/entry/scipysparsecoomatrix
  # 참조에서 row에 해당하는 값이 movieId, col에 해당하는 값이 userId, data에 해당하는 값이 rating

  # Create a sparse matrix of all the item/user/counts triples
  rating_matrix = coo_matrix((ratings_df["rating"].astype(np.float32),
                              (ratings_df["movieId"].cat.codes.copy(),
                              ratings_df["userId"].cat.codes.copy(),),))

  # factors : 숫자가 클수록 기존 데이터에 대한 정확도는 높아지지만, 과적합의 위험이 있음. 이 경우 결과는 정확하지만, 새로운 데이터에 대한 예측력은 떨어질 수 있음.
  # 과적합 참조 : https://kimmaadata.tistory.com/31

  # regularization : 과적합을 방지하기 위한 정규화 항. 값이 클수록 정규화가 강해지고, 과적합이 줄어듦. 하지만, 값이 너무 크면 정확도가 떨어질 수 있음.
  # iterations : 반복 횟수. 값이 클수록 정확도는 높아지지만, 시간이 오래 걸림.
  # dtype : 데이터 타입. 기본값은 np.float64
  # float64로 설정하면 정확도가 높아지지만, 메모리 사용량이 늘어남.
  
  als_model = AlternatingLeastSquares(factors=50, regularization=0.01, iterations=50, dtype=np.float64)

  # Convert to CSR matrix
  rating_matrix_csr = rating_matrix.tocsr()

  # ALS 모델 학습
  als_model.fit(weight * rating_matrix_csr)

  pickle.dump(als_model, open(saved_model_fname, 'wb'))
  return als_model



def calulate_item_based(item_id, items):
  loaded_model = pickle.load(open(saved_model_fname, 'rb')) # rb : read binary
  recs = loaded_model.similar_items(itemid=int(item_id), N=11)
  return [str(items[r]) for r in recs[0]]

def item_based_recommendation(item_id):
  ratings_df = pd.read_csv(data_fname)
  ratings_df["userId"] = ratings_df["userId"].astype("category")
  ratings_df["movieId"] = ratings_df["movieId"].astype("category")
  movies_df = pd.read_csv(item_fname)

  # cat.categories : 카테고리의 목록을 반환
  # astype("category")를 사용하여 특정 열을 카테고리형으로 변환할 수 있음
  # enumerate() 함수는 반복 가능한 객체(여기서는 movieId의 카테고리 리스트)에 인덱스를 추가
  # dict() 함수는 키-값 쌍을 딕셔너리로 변환

  # 예시 데이터프레임
  # ratings_df = pd.DataFrame({
  #     'movieId': [100, 200, 100, 300, 200, 100]
  # })
  # ratings_df['movieId'] = ratings_df['movieId'].astype('category')

  # enumerate 사용 예시
  # print(dict(enumerate(ratings_df["movieId"].cat.categories)))

  # 출력 결과
  # {0: 100, 1: 200, 2: 300}

  items = dict(enumerate(ratings_df["movieId"].cat.categories))
  try:
    # cat.categories.get_loc() : 카테고리의 위치를 반환
    # item_id를 int로 변환하여 get_loc() 함수에 전달
    # get_loc() 함수는 해당 아이템의 위치를 반환
    parsed_id = ratings_df["movieId"].cat.categories.get_loc(int(item_id))
    result = calulate_item_based(parsed_id, items)
  except KeyError as e:
    result = []

  # result = ['1', '2', '3', 'item_id']이고 item_id = 2라면, 필터링된 결과는 [1, 3]
  # result 리스트 내의 값과 item_id의 타입이 다를 수 있으므로, 조건문에서 타입을 맞춘 후 비교
  # 즉, int(x) 가 아닌 x로 비교하면 item_id와 같은 아이템이 추천 결과에 포함될 수 있음
  result = [int(x) for x in result if int(x) != int(item_id)]

  # isin(result)는 result 리스트에 포함된 영화 ID들만 선택하는 필터링 조건. 즉, result에 포함된 영화 ID가 movies_df의 movieId 열에 있는지 여부를 확인하고, 일치하는 행만 필터링.
  result_items = movies_df[movies_df["movieId"].isin(result)].to_dict("records")
  return result_items




if __name__ == "__main__":
  command = sys.argv[1]
  if command == 'item-based':
    item_id = sys.argv[2]
    print(json.dumps(item_based_recommendation(item_id)))
  else:
    print("Error: Invalid arguments")
    sys.exit(1)

  model_train() # 데이터를 학습하여 모델을 생성하는 함수 호출. 처음한번 실행 후 주석처리. 이후 데이터가 추가되면 주기적으로 학습