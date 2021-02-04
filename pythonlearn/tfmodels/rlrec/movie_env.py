from gym import spaces
from gym import Env
import random
import pandas
from scipy import spatial
from bigdl.dataset import movielens
from collections import deque
from pythonlearn.tfmodels.rlrec.utils import *
from zoo.examples.textclassification.news20 import get_glove

class EnvConfig(object):
    def __init__(self, values= None):
        self._values = {
            'user_max': 6040,
            'movie_max': 3952,
            'rate_dim': 5,
            'glove_dim': 50,
            'ncf_embed': True,
            'user_dim': 20, # if not ncf_embed, user_dim =30 from one hot
            'movie_dim': 20,
            'ncf_model_path': "/Users/guoqiong/intelWork/git/learn/pythonlearn/pythonlearn/zoomodels/save_model/movie_ncf.zoomodel",
            'episode_length': 30,
            'history_length' :10
        }
        if values:
            self._values.update(values)

class MovieEnv(Env):
    def __init__(self, config = None):
        super(MovieEnv, self).__init__()
        self._config = config
        [self.users, self.movies, self.ratings, self._user_movies_env] = self._get_data()
        self.info = {}
        obs_dim = self._config["user_dim"] + self._config["movie_dim"]
        self.observation_space = spaces.Box(-3.4028234663852886e+38, 3.4028234663852886e+38, (obs_dim,))
        self.action_space = spaces.Discrete(self._config["movie_max"])
        userid = random.randint(1, self._config["user_max"] + 1)
        hist_moviesids = deque([random.randint(1, self._config["movie_max"] + 1) \
                      for _ in range(self._config["history_length"])])
        self._userid = userid
        self._user_movies_hist = {userid: hist_moviesids}
        self._steps = 0
        self._done = False

    def step(self, action):
        assert self._done is not True, ("cannot call step() once episode finished)")
        self._steps += 1
        reward = self._get_reward(action)
        obs = self._get_obs(action)
        self._curr_obs = obs
        # check if end
        done = True if self._steps >= self._config["episode_length"] else False
        self.info["step"] = self._steps
        # get some info
        return obs, reward, done, self.info

    def _get_reward(self, action):
        userid = self._userid
        movieid = action
        rate = self.ratings[(userid, movieid)] if (userid, movieid) in self.ratings.keys() else 0
        rate = rate if movieid not in self.user_movies_hist[userid] else 0
        movie_embed_hist = self._average_movie_embed(self.user_movies_hist[userid])

        movie_embed_action = np.array([0.0 for _ in range(self._config["movie_dim"])]) \
            if movieid not in self.movies.keys() else self.movies[movieid]

        similarity = 1 - spatial.distance.cosine(movie_embed_action, movie_embed_hist) \
            if movieid in self.movies.keys() else 1
        reward = rate + (1 - similarity)

        return reward

    def _get_obs(self, action):
        userid = self._userid
        movieid = action
        movieids = self._user_movie_hist[userid]
        movieids.popleft()
        movieids.append(movieid)
        self._user_movie_hist[userid] = movieids
        user_embed = self.users[userid]
        movie_embed = self._average_movie_embed(movieids)
        obs = np.concatenate([user_embed, movie_embed])
        return obs

    def reset(self):
        self._steps = 0
        userid = random.randint(1, self.user_max + 1)
        movieids = [random.randint(1, self.movie_max + 1) \
                    for _ in range(self._config["history_length"])]
        self._userid = userid
        self._user_movie_hist = {userid: movieids}
        obs = self.users[userid]
        m_vecs = self._average_movie_embed(movieids)
        obs = np.concatenate([obs, m_vecs])
        return obs

    def render(self, mode='human'):
        pass

    def _get_data(self):
        movielens_data = movielens.get_id_ratings("./data/movielens/")

        if (self._config["ncf_embed"]):
            users_dict, movie_dict = self._get_embed_ncf()
        else:
            # users_dict = encode_user_embed(user_file ="./data/movielens/ml-1m/users.dat")
            movie_dict = self._get_movies_embed_glove(movie_file="./data/movielens/ml-1m/movies.dat",
                                                     embed_dim = self._config['glove_dim'])

        df = pandas.DataFrame(movielens_data, columns=['uid','mid','rate']) \
            .groupby('uid')['mid'].apply(list).reset_index(name='mids')
        user_hist_movies = {}
        for index, row in df.iterrows():
            user_hist_movies[row['uid']] = row['mids']
        ratings_data = {(movielens_data[i][0],movielens_data[i][1]): \
                            movielens_data[i][2] for i in range(len(movielens_data))}
        return [users_dict, movie_dict, ratings_data, user_hist_movies]

    def _average_movie_embed(self, mids):
        mean_embed = []
        for mid in mids:
            movie_embed = np.array([0.0 for _ in range(self._config['movie_dim'])]) \
                if mid not in self.movies.keys()  else self.movies[mid]
            mean_embed.append(movie_embed)
        mean_embed = sum(mean_embed)/self._config['history_length']
        return mean_embed

    def _get_embed_ncf(self):
        from zoo.models.recommendation import NeuralCF
        ncf = NeuralCF(user_count=self._config["user_max"],
               item_count=self._config["movie_max"],
               class_num=self._config["rate_dim"],
               hidden_layers=[20, 10],
               include_mf = False)
        loaded = ncf.load_model(self._config["ncf_model_path"])
        user_embed = loaded.get_weights()[0]
        item_embed = loaded.get_weights()[1]
        user_dict = {}
        for i in range(1,self._config["user_max"] + 1):
            user_dict[i] = user_embed[i][:]

        item_dict = {}
        for i in range(1, self._config["movie_max"] + 1):
            item_dict[i] = item_embed[i][:]
        return(user_dict, item_dict)

    def _get_movies_embed_glove(movie_file="./data/movielens/ml-1m/movies.dat", embed_dim = 50):
        assert()
        glove_dict = get_glove(base_dir="./data/glove.6B", dim=embed_dim)
        movie_dict = {}
        with open(movie_file, encoding='latin-1') as movie_f:
            for line in movie_f:
                values = line.split("::")
                key = values[0]
                words = custom_text_to_sequence(values[2])
                movie_embed = [0.0 for i in range(embed_dim)]
                count = 0
                for word in words:
                    if word in glove_dict.keys():
                        word_embed = glove_dict[word]
                        count = count + 1
                        for i in range(0, embed_dim):
                            movie_embed[i] = movie_embed[i] + word_embed[i]
                if count == 0:
                    sys.exit("please check the movie title")
                movie_embed = [item / count for item in movie_embed]
                movie_dict[int(key)] = np.array(movie_embed)
        return movie_dict
