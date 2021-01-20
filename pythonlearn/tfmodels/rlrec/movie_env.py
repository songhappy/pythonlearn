from gym import spaces
from gym import Env
import numpy as np
import sys
import random
import pandas
from scipy import spatial
from bigdl.dataset import movielens
from zoo.examples.textclassification.news20 import get_glove
import string
from zoo.models.recommendation import NeuralCF

class MovieEnv(Env):
    def __init__(self, config):
        super(MovieEnv, self).__init__()
        [self.users, self.movies, self.ratings, self._um_relation] = self._get_data()
        self.info = {}
        self.user_num = max(self.users.keys())
        self.movie_num = max((self.movies.keys()))
        self.rating_num = len(self.ratings)
        self.observation_space = spaces.Box(-3.4028234663852886e+38, 3.4028234663852886e+38, (40,))
        self.action_space = spaces.Discrete(self.movie_num)
        self._step = 0
        self._done = False
        uid = random.randint(1, self.user_num)
        mids = [random.randint(1, self.movie_num) for _ in range(10)]
        self._uid = uid
        self._um_relation = {uid: mids}

    def step(self, action):
        assert self._done is not True, ("cannot call step() once episode finished (call reset insead)")
        self._step += 1
        reward = self._get_reward(action)
        obs = self._get_obs(action)
        self._curr_obs = obs
        # check if end
        done = True if self._step >= 30 else False
        self.info["step"] = self._step
        # get some info
        return obs, reward, done, self.info

    def _get_reward(self, action):
        uid = self._uid
        mid = action
        rate = self.ratings[(uid, mid)] if (uid, mid) in self.ratings.keys() else 0
        rate = rate if mid not in self._um_relation[uid] else 0
        m_vecs = self._get_mvecs(self._um_relation[uid])

        m_action = np.array([0.0 for _ in range(20)]) if mid not in self.movies.keys() \
            else self.movies[mid]

        similarity = 1 - spatial.distance.cosine(m_action, m_vecs) if mid in self.movies.keys() \
            else 1
        reward = rate + (1 - similarity)

        #print("rate, similarity, reward", rate, similarity, reward)
        return reward

    def _get_obs(self, action):
        uid = self._uid
        mid = action
        mids = self._um_relation[uid]
        mids.pop(0)
        mids.append(mid)
        self._um_relation[uid] = mids
        obs = self.users[uid]
        m_vecs = self._get_mvecs(mids)
        obs = np.concatenate([obs, m_vecs])
        return obs

    def reset(self):
        self._step = 0
        uid = random.randint(1, self.user_num)
        mids = [random.randint(1, self.movie_num) for _ in range(10)]
        self._uid = uid
        self._um_relation = {uid: mids}
        obs = self.users[uid]
        m_vecs = self._get_mvecs(mids)
        obs = np.concatenate([obs, m_vecs])
        return obs

    def render(self, mode='human'):
        pass

    def _get_data(self):
        glove_data = get_glove(base_dir="./data/glove.6B", dim=50)
        movielens_data = movielens.get_id_ratings("./data/movielens/")
        # users_dict = self._get_users()
        # movie_dict = self._get_movies(glove_data)

        users_dict, movie_dict = self._get_embeddings()

        # users_dict = {i:np.array([13]) for i in range(len(users_dict))}
        # movie_dict = {i:np.array([4]) for i in range(len(movie_dict))}
        df = pandas.DataFrame(movielens_data, columns=['uid','mid','rate']) \
            .groupby('uid')['mid'].apply(list).reset_index(name='mids')
        um_relation = {}
        for index, row in df.iterrows():
            um_relation[row['uid']] = row['mids']
        ratings_data = {(movielens_data[i][0],movielens_data[i][1]): movielens_data[i][2] for i in range(len(movielens_data))}
        return [users_dict, movie_dict, ratings_data, um_relation]

    def _get_movies1(self, embedding_dict, movie_file="./data/movielens/ml-1m/movies.dat"):
        movie_dict = {}
        with open(movie_file, encoding='latin-1') as movie_f:
            for line in movie_f:
                values = line.split("::")
                key = values[0]
                table = str.maketrans('', '', string.punctuation)
                words = values[2].replace('\'s', '').replace('-', '|').strip().lower().split("|")
                words = [w.translate(table) for w in words]
                vector = [0.0 for i in range(50)]
                count = 0
                for word in words:
                    if word in embedding_dict.keys():
                        embedding_vector = embedding_dict[word]
                        count = count + 1
                        for i in range(0, 50):
                            vector[i] = vector[i] + embedding_vector[i]
                if count == 0:
                    sys.exit("please check the movie title")
                vector = [item / count for item in vector]
                movie_dict[int(key)] = np.array(vector)
            movie_f.close()
        return movie_dict

    def _get_users1(self, user_file ="./data/movielens/ml-1m/users.dat"):
        user_dict = {}
        with open(user_file) as user_f:
            for line in user_f:
                values = line.split("::")
                vector = []
                vector.append(0.0) if values[1] == "M" else vector.append(1.0)
                vector.append(float(values[2])/100)
                vector.append(float(values[3])/100)
                user_dict[int(values[0])] = np.array(vector)
            user_f.close()
        return user_dict

    def _get_mvecs(self, mids):
        m_vecs = []
        for mid in mids:
            m_vec = np.array([0.0 for _ in range(20)]) if mid not in self.movies.keys() \
                else self.movies[mid]
            m_vecs.append(m_vec)
        m_vecs = sum(m_vecs)
        return m_vecs

    def _get_embeddings(self):
        ncf = NeuralCF(user_count=6040,
               item_count=3952,
               class_num=5,
               hidden_layers=[20, 10],
               include_mf = False)
        loaded = ncf.load_model("/Users/guoqiong/intelWork/git/learn/pythonlearn/pythonlearn/zoomodels/save_model/movie_ncf.zoomodel")
        user_embed = loaded.get_weights()[0]
        print(user_embed.shape)
        item_embed = loaded.get_weights()[1]
        print(item_embed.shape)
        user_dict = {}
        for i in range(1,6041):
            user_dict[i] = user_embed[i][:]

        item_dict = {}
        for i in range(1, 3953):
            item_dict[i] = item_embed[i][:]
        return(user_dict, item_dict)
