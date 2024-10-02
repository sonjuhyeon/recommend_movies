import pandas as pd
import sys
import json
import re

item_fname = "data/movie_final.csv"
columns = ['id', 'title', 'genres', 'imdb_id', 'tmdb_id', 'imdb_url', 'rating_count', 'rating_avg', 'image_url']

# 랜덤으로 count개의 아이템을 반환
def random_items(count):
  movies_df = pd.read_csv(item_fname)[1:]
  movies_df = movies_df.fillna("") # 공백을 채워준다
  result_items = movies_df.sample(n=count).to_dict("records")
  return result_items

# 최신 영화를 count개 반환
def latest_items(count):
    movies_df = pd.read_csv(item_fname)[1:]
    movies_df = movies_df.fillna("")  # 공백으로 NaN을 채워줌

    def extract_year(title):
        match = re.search(r'\((\d{4})\)', title)
        if match:
            return int(match.group(1))
        return None  # 연도가 없으면 None 반환

    movies_df['year'] = movies_df['title'].apply(extract_year)
    latest_movies_df = movies_df.sort_values(by='year', ascending=False).head(count)
    result_items = latest_movies_df.to_dict("records")
    return result_items

# genre 키워드를 포함하는 영화 count개 반환
def genres_items(genre, count):

  movies_df = pd.read_csv(item_fname, names=columns)
  genres_df = movies_df.fillna("") # 공백으로 NaN을 채워줌
  genres_df = movies_df[movies_df["genres"].str.contains(genre, case=False, na=False)]
  # case = False : 대소문자 구분 안함

  result_items = genres_df.sample(n=count).to_dict("records")
  return result_items

if __name__ == "__main__":
  
  try:
    command = sys.argv[1]

    if command == "random":
      count = int(sys.argv[2])
      print(json.dumps(random_items(count)))
    elif command == "latest":
      count = int(sys.argv[2])
      print(json.dumps(latest_items(count)))
    elif command == "genres":
      genre = sys.argv[2]
      count = int(sys.argv[3])
      print(json.dumps(genres_items(genre, count)))
    else:
      print("Error: Invalid command error") 
      sys.exit(1)

  except ValueError:
    print("Error: Invalid arguments")
    sys.exit(1)