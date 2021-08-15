import sys
from src.recommend import Recommend

from setup import set_env

args = set_env()
model = Recommend()


if __name__ == "__main__":
    user_id = args.user_id
    eval_k = args.eval_k

    result = model.recommend(user_id, eval_k)
    sys.exit(result)
