import dvc.api
import os
from dvc.repo import Repo as DvcRepo
import matplotlib.pyplot as plt
import numpy as np
from git.repo import Repo as GitRepo


def update_package_list():
    os.system("conda env export > environment.yml")

# %%

results_dir = r"results/parameter_estimation/example_column/column_transport"

if not os.path.exists(results_dir):
    os.makedirs(results_dir, exist_ok=True)

dvc_repo = DvcRepo(".")
git_repo = GitRepo(".")
index = git_repo.index

with dvc.api.open(
        r"data\chromatograms\example_column\non-interacting_tracer\non_pore_penetrating_tracer.csv",
        repo=r'https://github.com/ronald-jaepel/DVC-data-test'
) as f:
    # print(f.readlines())
    data = np.loadtxt(f, delimiter=",")
    fig, ax = plt.subplots(1)
    ax.plot(data[:, 0], data[:, 1])
    plt.savefig(os.path.join(results_dir, "input.png"))
    dvc_repo.add(results_dir)
    update_package_list()
    git_repo.git.add(".")
    commit_return = git_repo.git.commit("-m", "Add results file")
    print(commit_return)

# %%

url = r'https://github.com/ronald-jaepel/DVC-data-test'
out = r"raw_data\non-interacting_tracer\non_pore_penetrating_tracer.csv"
path = r"data\chromatograms\example_column\non-interacting_tracer\non_pore_penetrating_tracer.csv"

dvc_repo.imp(url=url, path=path, out=out)
# %%

url = r'https://github.com/ronald-jaepel/DVC-data-test'
out = r"raw_data"
path = r"data\chromatograms\example_column\non-interacting_tracer"

dvc_repo.imp(url=url, path=path, out=out)
