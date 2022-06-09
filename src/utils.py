import pandas as pd
import numpy as np


def prefilter_items(data, take_n_popular=5000, item_features=None):
    # Уберём неинтересные для рекомендаций категории (department)
    if item_features is not None:
        department_size = pd.DataFrame(item_features. \
                                       groupby('department')['item_id'].nunique(). \
                                       sort_values(ascending=False)).reset_index()

        department_size.columns = ['department', 'n_items']
        rare_departments = department_size[department_size['n_items'] < 150].department.tolist()
        items_in_rare_departments = item_features[
            item_features['department'].isin(rare_departments)].item_id.unique().tolist()

        data = data[~data['item_id'].isin(items_in_rare_departments)]

    # Уберём слишком дешёвые товары (на них не заработаем). 1 покупка из рассылок стоит 60 руб.
    data['price'] = data['sales_value'] / (np.maximum(data['quantity'], 1))
    data = data[data['price'] > 2]

    # Уберём слишком дорогие товары
    data = data[data['price'] < 50]

    # Возьмём топ по популярности
    popularity = data.groupby('item_id')['quantity'].sum().reset_index()
    popularity.rename(columns={'quantity': 'n_sold'}, inplace=True)

    top = popularity.sort_values('n_sold', ascending=False).head(take_n_popular).item_id.tolist()

    # Заведём фиктивный item_id (если юзер покупал товары из топ-5000, то он "купил" такой товар)
    data.loc[~data['item_id'].isin(top), 'item_id'] = 999999

    return data


def postfilter_items(user_id, recommendations):
    pass


def prepare_matrix(data):
    """Готовит user-item матрицу"""
    user_item_matrix = pd.pivot_table(data,
                                      index='user_id', 
                                      columns='item_id',
                                      values='quantity', # можно пробовать другие варианты
                                      aggfunc='mean',
                                      fill_value=0
                                      )

    user_item_matrix = user_item_matrix.astype(float) # необходимый тип матрицы для implicit

    return user_item_matrix


def prepare_dicts(user_item_matrix):
    """Подготавливает вспомогательные словари"""
        
    userids = user_item_matrix.index.values
    itemids = user_item_matrix.columns.values

    matrix_userids = np.arange(len(userids))
    matrix_itemids = np.arange(len(itemids))

    id_to_itemid = dict(zip(matrix_itemids, itemids))
    id_to_userid = dict(zip(matrix_userids, userids))

    itemid_to_id = dict(zip(itemids, matrix_itemids))
    userid_to_id = dict(zip(userids, matrix_userids))
        
    return id_to_itemid, id_to_userid, itemid_to_id, userid_to_id


def fit_own_recommender(user_item_matrix):
    """Обучает модель, которая рекомендует товары, среди товаров, купленных юзером"""
    
    own_recommender = ItemItemRecommender(K=1, num_threads=4)
    own_recommender.fit(csr_matrix(user_item_matrix).T.tocsr())
        
    return own_recommender


def fit(user_item_matrix, n_factors=20, regularization=0.001, iterations=15, num_threads=4, random_state=41):
    """Обучает ALS"""
        
    model = AlternatingLeastSquares(factors=n_factors, 
                                    regularization=regularization,
                                    iterations=iterations,  
                                    num_threads=num_threads,
                                    random_state=random_state
                                    )
    model.fit(csr_matrix(user_item_matrix).T.tocsr())
        
    return model


def get_similar_item(model, itemid_to_id, id_to_itemid, x):
    _id = itemid_to_id[x]
    recs = model.similar_items(_id, N=2)
    top_rec = recs[1][0]

    return id_to_itemid[top_rec]


def get_own_recommendations(own, userid, user_item_matrix, N):
    res = model.recommend(userid=self.userid_to_id[user],
                          user_items=csr_matrix(self.user_item_matrix).tocsr(),
                          N=N,
                          filter_already_liked_items=False,
                          filter_items=[self.itemid_to_id[999_999]],
                          recalculate_user=True
                         )

    res = [self.id_to_itemid[rec[0]] for rec in res]

    return res


def get_similar_items_recommendation(self, user, N=5):
    """Рекомендуем товары, похожие на топ-N купленных юзером товаров"""

    top_purchases = data.groupby(['user_id', 'item_id'])['quantity'].count().reset_index()
    top_purchases.sort_values('quantity', ascending=False, inplace=True)
    top_purchases = top_purchases[top_purchases['item_id'] != 999_999]

    top_users_purchases = top_purchases[top_purchases['user_id'] == user].head(N)
    res = top_users_purchases['item_id'].apply(lambda x: get_similar_item(model, 
                                                                          itemid_to_id=itemid_to_id, 
                                                                          id_to_itemid=id_to_itemid, 
                                                                          x=x)).tolist()

    assert len(res) == N, 'Количество рекомендаций != {}'.format(N)
    return res


def get_similar_users_recommendation(self, user, N=5):
    """Рекомендуем топ-N товаров, среди купленных похожими юзерами"""

    res = []

    similar_users = model.similar_users(userid_to_id[userid], N=N+1)
    similar_users = [rec[0] for rec in similar_users]
    similar_users = similar_users[1:]

    for user in similar_users:
        userid = self.id_to_userid[user]
        res.extend(self.get_own_recommendations(userid, N=1))

    assert len(res) == N, 'Количество рекомендаций != {}'.format(N)
    return res
