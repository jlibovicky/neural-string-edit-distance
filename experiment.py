import datetime
import logging
import os
import subprocess

from typing import List

import yaml

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
#logging.getLogger().addHandler(logging.StreamHandler(sys.stderr))


def save_git_info(git_commit_file: str, git_diff_file: str,
                  branch: str = "HEAD") -> None:
    repo_dir = os.path.dirname(os.path.realpath(__file__))

    with open(git_commit_file, "wb") as file:
        subprocess.run(["git", "log", "-1", "--format=%H", branch],
                       cwd=repo_dir, stdout=file, check=False)

    with open(git_diff_file, "wb") as file:
        subprocess.run(
            ["git", "--no-pager", "diff", "--color=always", branch],
            cwd=repo_dir, stdout=file, check=False)


def experiment_logging(experiment_id: str, args) -> str:
    experiment_dir = os.path.join("experiments", experiment_id)
    os.mkdir(experiment_dir)

    save_git_info(
        os.path.join(experiment_dir, "git_commit"),
        os.path.join(experiment_dir, "git_diff"))

    with open(os.path.join(experiment_dir, "args"), "w") as file:
        print(yaml.dump(args.__dict__), file=file)

    log_handler = logging.FileHandler(
        os.path.join(experiment_dir, "train.log"))
    log_handler.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
    logging.getLogger().addHandler(log_handler)

    return experiment_dir


def get_timestamp() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")


def save_vocab(itos_list: List[str], path: str) -> None:
    with open(path, "w") as file:
        for item in itos_list:
            print(item, file=file)
