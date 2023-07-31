# %%
import dvc.api
import os
# from dvc.repo import add as dvc_add
from dvc.repo import Repo as DvcRepo
import matplotlib.pyplot as plt
import numpy as np
from git.repo import Repo as GitRepo

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
    git_repo.git.add(".")
    git_repo.git.commit()

# %%
