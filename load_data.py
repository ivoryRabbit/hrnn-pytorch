import glob
import subprocess
import zipfile
import pandas as pd


def load_data(save_file_name: str) -> None:
    """load Movie Lens data"""
    file_name = "ml-1m.zip"
    download_url = f"https://files.grouplens.org/datasets/movielens/{file_name}"

    if glob.glob(save_file_name):
        return

    subprocess.check_call(f"curl {download_url} -O", shell=True)
    file_zip = zipfile.ZipFile(file_name)
    file_zip.extractall()

    data = "ml-1M/ratings.dat"
    col_names = ["user_id", "item_id", "rating", "timestamp"]
    ratings = pd.read_csv(data, delimiter="::", names=col_names, engine="python")
    ratings.to_csv(save_file_name, index=False)

    subprocess.check_call(f"rm {file_name}", shell=True)
    subprocess.check_call(f"rm -r ml-1M", shell=True)


if __name__ == "__main__":
    load_data(save_file_name="data/ml-1m.csv")
