# PROJE: Hybrid Recommender System

import pandas as pd
pd.pandas.set_option('display.max_columns', None)
pd.pandas.set_option('display.width', 300)

# Görev 1: Verinin Hazırlanması

# Adım 1: Movie ve Rating veri setlerini okutunuz.

movie = pd.read_csv("datasets/movie.csv")
rating = pd.read_csv("datasets/rating.csv")
rating.head()

# Adım 2: Rating veri setine filmlerin isimlerini ve türünü movie film setini kullanrak ekleyiniz.

df = movie.merge(rating, how="left", on="movieId")
df.head()
df.info()
df.isnull().sum()

# Adım 3: Herbir film için toplam kaç kişinin oy kullandığını hesaplayınız.Toplam oy kullanılma sayısı 1000'un altında olan filmleri veri setinden çıkarınız.

comment_counts = pd.DataFrame(df["title"].value_counts())
rare_movies = comment_counts[comment_counts["title"] <= 1000].index
common_movies = df[~df["title"].isin(rare_movies)]
common_movies["title"].nunique()

# Adım 4: # index'te userID'lerin sutunlarda film isimlerinin ve değer olarakta ratinglerin bulunduğu
# dataframe için pivot table oluşturunuz.

user_movie_df = common_movies.pivot_table(index="userId", columns="title", values="rating")
user_movie_df.head()

# Adım 5: Yukarıda yapılan tüm işlemleri fonksiyonlaştıralım

def create_user_movie_df(dataframe):
    import pandas as pd
    pd.pandas.set_option('display.max_columns', None)
    pd.pandas.set_option('display.width', 300)
    movie = pd.read_csv("datasets/movie.csv")
    rating = pd.read_csv("datasets/rating.csv")
    dataframe = movie.merge(rating, how="left", on="movieId")
    comment_counts = pd.DataFrame(dataframe["title"].value_counts())
    rare_movies = comment_counts[comment_counts["title"] <= 1000].index
    common_movies = dataframe[~dataframe["title"].isin(rare_movies)]
    user_movie_df = common_movies.pivot_table(index="userId", columns="title", values="rating")
    return user_movie_df

create_user_movie_df(df)

# Görev 2: Öneri Yapılacak Kullanıcının İzlediği Filmlerin Belirlenmesi

# Adım 1: Rastgele bir kullanıcı id'si seçiniz.

#random_user = int(pd.Series(user_movie_df.index).sample(1).values)
random_user = 108170

# Adım 2: Seçilen kullanıcıya ait gözlem birimlerinden oluşan random_user_df adında yeni bir dataframe oluşturunuz.

random_user_df = user_movie_df[user_movie_df.index == random_user]

# Adım 3:Seçilen kullanıcıların oy kullandığı filmleri movies_watched adında bir listeye atayınız.

movies_watched = random_user_df.columns[random_user_df.notna().any()].tolist()
len(movies_watched) #186 tane filme oy vermiş

# Görev 3: Aynı Filmleri İzleyen Diğer Kullanıcıların Verisine ve Id'lerine Erişmek

# Adım 1: Seçilen kullanıcının izlediği fimlere ait sutunları user_movie_df'ten seçiniz ve movies_watched_df adında yeni bir dataframe oluşturuyoruz.

movies_watched_df = user_movie_df[movies_watched]
movies_watched_df.head()

# Adım 2: Herbir kullancının seçili user'in izlediği filmlerin kaçını izlediği bilgisini taşıyan user_movie_count adında yeni bir dataframe oluşturunuz.
# Ve yeni bir df oluşturuyoruz.

user_movie_count = movies_watched_df.T.notna().sum()  #notna() yerine notnull() da yazabilirdik.
user_movie_count.head()
user_movie_count = user_movie_count.reset_index()
user_movie_count.columns = ["userId", "movie_count"]

# Adım 3: Seçilen kullanıcının oy verdiği filmlerin yüzde 60 ve üstünü izleyenleri benzer kullanıcılar olarak görüyoruz.
# Bu kullanıcıların id’lerinden users_same_movies adında bir liste oluşturunuz.

perc = len(movies_watched)*60/100
users_same_movies = user_movie_count[user_movie_count['movie_count'] >= perc]["userId"]
len(users_same_movies)

#############################################
# Görev 4: Öneri Yapılacak Kullanıcı ile En Benzer Kullanıcıların Belirlenmesi
#############################################

# Adım 1: user_same_movies listesi içerisindeki seçili user ile benzerlik gösteren kullanıcıların id’lerinin bulunacağı şekilde movies_watched_df dataframe’ini filtreleyiniz.


user_movies_df = movies_watched_df[movies_watched_df.index.isin(users_same_movies)]
user_movies_df.head()

# Adım 2: Kullanıcıların birbirleri ile olan korelasyonlarının bulunacağı yeni bir corr_df dataframe’i oluşturunuz.

corr_df = user_movies_df.T.corr().unstack().drop_duplicates()
corr_df.head()
corr_df = pd.DataFrame(corr_df, columns=["corr"])
corr_df.index.names = ["user_id_1", "user_id_2"]
corr_df = corr_df.reset_index()

# Adım 3: Seçili kullanıcı ile yüksek korelasyona sahip (0.65’in üzerinde olan) kullanıcıları filtreleyerek top_users adında yeni bir dataframe oluşturunuz.

top_users = corr_df[(corr_df["user_id_1"] == random_user) & (corr_df["corr"] >= 0.65)][["user_id_2", "corr"]].reset_index(drop=True)
top_users.head()
top_users = top_users.sort_values(by="corr", ascending=False)
top_users.rename(columns={"user_id_2": "userId"}, inplace=True)

# Adım 4:  top_users dataframe’ine rating veri seti ile merge ediniz

top_users_ratings = top_users.merge(rating[["userId", "movieId", "rating"]], how="inner")
top_users_ratings.head()
top_users_ratings = top_users_ratings[top_users_ratings["userId"] != random_user]

#############################################
# Görev 5: Weighted Average Recommendation Score'un Hesaplanması ve İlk 5 Filmin Tutulması
#############################################

# Adım 1: Her bir kullanıcının corr ve rating değerlerinin çarpımından oluşan weighted_rating adında yeni bir değişken oluşturunuz.

top_users_ratings["weighted_rating"] = top_users_ratings["corr"]*top_users_ratings["rating"]

# Adım 2: Film id’si ve her bir filme ait tüm kullanıcıların weighted rating’lerinin ortalama değerini içeren recommendation_df adında yeni bir
# dataframe oluşturunuz.

recommedation_df = top_users_ratings.groupby("movieId").agg({"weighted_rating": "mean"})
recommedation_df.head()
recommedation_df = recommedation_df.reset_index()

# Adım 3: recommendation_df içerisinde weighted rating'i 3.5'ten büyük olan filmleri seçiniz ve weighted rating’e göre sıralayınız.
# İlk 5 gözlemi movies_to_be_recommend olarak kaydediniz.

movies_to_be_recommend = recommedation_df[recommedation_df["weighted_rating"] >= 3.5].\
                        sort_values("weighted_rating", ascending=False)
movies_to_be_recommend.head()
# Adım 4:  Tavsiye edilen 5 filmin isimlerini getiriniz.

movie = pd.read_csv("datasets/movie.csv")
movie.head()
movies_to_be_recommend.merge(movie[["movieId","title"]])["title"][:5]

# Görev 6: Item-Based Recommendation

# Adım 1: movie,rating veri setlerini okutunuz.

movie = pd.read_csv("datasets/movie.csv")
rating = pd.read_csv("datasets/rating.csv")

# Adım 2: Öneri yapılacak kullanıcının 5 puan verdiği filmlerden puanı en güncel olan filmin id'sinin alınız.

user = 108170
movie_id = rating[(rating["userId"] == user) & (rating["rating"] == 5.0)].sort_values("timestamp", ascending=False)["movieId"][0:1]

# Adım 3 :User based recommendation bölümünde oluşturulan user_movie_df dataframe’ini seçilen film id’sine göre filtreleyiniz.

movie_df = user_movie_df[movie[movie["movieId"] == movie_id]["title"].values[0]]

# Adım 4: Filtrelenen dataframe’i kullanarak seçili filmle diğer filmlerin korelasyonunu bulunuz ve sıralayınız.

user_movie_df.corrwith(movie_df).sort_values(ascending=False).head(10)

# Son iki adımı uygulayan fonksiyon
def item_based_recommender(movie_name, user_movie_df):
    movie = user_movie_df[movie_name]
    return user_movie_df.corrwith(movie).sort_values(ascending=False).head(10)


# Adım 5: Seçili film’in kendisi haricinde ilk 5 film’I öneri olarak veriniz.
movies_from_item_based = item_based_recommender(movie[movie["movieId"] == movie_id]["title"].values[0], user_movie_df)
# 1'den 6'ya kadar. 0'da filmin kendisi var. Onu dışarda bıraktık.
movies_from_item_based[1:6].index.to_list()





















