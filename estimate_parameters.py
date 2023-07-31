# ---
# jupyter:
#   jupytext:
#     formats: md:myst,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import dvc.api
import os
from dvc.repo import Repo as DvcRepo
import matplotlib.pyplot as plt
import numpy as np
from git.repo import Repo as GitRepo


# %%
def update_package_list():
    os.system("conda env export > environment.yml")


def commit(message="bump"):
    update_package_list()
    git_repo.git.add(".")
    commit_return = git_repo.git.commit("-m", message)
    print("\n" + commit_return)


# %%

repo_path = r"C:\Users\ronal\PycharmProjects\DVC-code-example"
os.chdir(repo_path)

# %%
results_dir = r"results/parameter_estimation/example_column/column_transport"

if not os.path.exists(results_dir):
    os.makedirs(results_dir, exist_ok=True)

dvc_repo = DvcRepo(".")
git_repo = GitRepo(".")
# index = git_repo.index

# %% [markdown]

# ## Use data from existing raw_data repository
# The information on what data is available sits in the github repo
# the large datafiles themselves are located in some file hosting service
# it can be a folder on your hard-drive or a google drive or one of many other hosting services.

# %%
if not os.path.exists(os.path.join(results_dir, "input.png")):
    with dvc.api.open(
            r"data\chromatograms\example_column\non-interacting_tracer\non_pore_penetrating_tracer.csv",
            repo=r'https://github.com/ronald-jaepel/DVC-data-test'
    ) as f:
        # print(f.readlines())
        data = np.loadtxt(f, delimiter=",")
        fig, ax = plt.subplots(1)
        ax.plot(data[:, 0], data[:, 1])
        plt.savefig(os.path.join(results_dir, "input.png"))
        dvc_repo.add(os.path.join(results_dir, "input.png"))

# %% [markdown]

# It is also possible to import data into the current repo. Using the DVC import functionality
# the imported data is combined with the source, URL and git commit hash.

# %%
# Example for importing an individual file
# url = r'https://github.com/ronald-jaepel/DVC-data-test'
# out = r"raw_data\non-interacting_tracer\non_pore_penetrating_tracer.csv"
# path = r"data\chromatograms\example_column\non-interacting_tracer\non_pore_penetrating_tracer.csv"
#
# dvc_repo.imp(url=url, path=path, out=out)

# %%

# Example for importing an entire folder
os.makedirs("imports", exist_ok=True)

url = r'https://github.com/ronald-jaepel/DVC-data-test'
out = r"imports\experimental_data"
path = r"data\chromatograms\example_column\non-interacting_tracer"

if not os.path.exists(out):
    dvc_repo.imp(url=url, path=path, out=out)

# %% [markdown]
# (fit_column_transport)=
# # Fit Column Transport Parameters
# When characterizing a chromatographic system column, some parameters of the model can be directly measured (e.g. column length, diameter) or are provided by the resin manufacturer (e.g. particle radius).
# However, other transport parameters like axial dispersion and porosities need to be determined through experiments.
#
# One approach to determine these parameters is the inverse method.
# By adjusting the values of the parameters in the simulation model and comparing the resulting behavior to the experimental data, the optimal parameter values that match the observed behavior can be found.
#
# ## Fit Bed Porosity and Axial Dispersion
# The bed porosity is an important parameter that describes the fraction of the column volume that is available for fluid flow.
# Axial dispersion is a transport parameter that describes the spreading of a solute or tracer as it moves through a chromatographic system column along the length of the column.
# It occurs due to differences in velocity between different layers of the fluid, leading to radial mixing of the fluid and the solutes.
#
# To determine the value of these parameters, an experiment is conducted using a non-pore-penetrating tracer.
# The tracer is injected into the column and its concentration at the column outlet is measured and compared to the concentration predicted by simulation results.

# %%
import numpy as np

data = np.loadtxt('imports/experimental_data/non_pore_penetrating_tracer.csv', delimiter=',')

time_experiment = data[:, 0]
c_experiment = data[:, 1]

from CADETProcess.reference import ReferenceIO


tracer_peak = ReferenceIO(
    'Tracer Peak', time_experiment, c_experiment
)

if __name__ == '__main__':
    _ = tracer_peak.plot()

# %% [markdown]
# ### Reference Model
# To accurately model a chromatographic system, it is crucial to establish a reference model that closely resembles the real system.
# The reference model allows for the determination of parameter influences and optimization of their values.
# Arbitrary values can be set for the unknown parameters such as `bed_porosity` and `axial_dispersion`, since they will be optimized.
# In order to model that the non-penetrating tracer does in fact not enter the pore, the `film_diffusion` needs to be set to $0$.
# Moreover, `particle_porosity` will be determined using a separate experiment.

# %%
from CADETProcess.processModel import ComponentSystem

component_system = ComponentSystem(['Non-penetrating Tracer'])

# %%
from CADETProcess.processModel import Inlet, Outlet, LumpedRateModelWithPores

feed = Inlet(component_system, name='feed')
feed.c = [0.0005]

eluent = Inlet(component_system, name='eluent')
eluent.c = [0]

column = LumpedRateModelWithPores(component_system, name='column')

column.length = 0.1
column.diameter = 0.0077
column.particle_radius = 34e-6

column.axial_dispersion = 1e-8
column.bed_porosity = 0.3
column.particle_porosity = 0.8
column.film_diffusion = [0]

outlet = Outlet(component_system, name='outlet')

# %%
from CADETProcess.processModel import FlowSheet

flow_sheet = FlowSheet(component_system)

flow_sheet.add_unit(feed)
flow_sheet.add_unit(eluent)
flow_sheet.add_unit(column)
flow_sheet.add_unit(outlet)

flow_sheet.add_connection(feed, column)
flow_sheet.add_connection(eluent, column)
flow_sheet.add_connection(column, outlet)

# %%
from CADETProcess.processModel import Process

Q_ml_min = 0.5  # ml/min
Q_m3_s = Q_ml_min / (60 * 1e6)
V_tracer = 50e-9  # m3

process = Process(flow_sheet, 'Tracer')
process.cycle_time = 15 * 60

process.add_event(
    'feed_on',
    'flow_sheet.feed.flow_rate',
    Q_m3_s, 0
)
process.add_event(
    'feed_off',
    'flow_sheet.feed.flow_rate',
    0,
    V_tracer / Q_m3_s
)

process.add_event(
    'feed_water_on',
    'flow_sheet.eluent.flow_rate',
    Q_m3_s,
    V_tracer / Q_m3_s
)

process.add_event(
    'eluent_off',
    'flow_sheet.eluent.flow_rate',
    0,
    process.cycle_time
)

# %% [markdown]
# ### Simulator

# %%
from CADETProcess.simulator import Cadet

simulator = Cadet()

if __name__ == '__main__':
    simulation_results = simulator.simulate(process)
    _ = simulation_results.solution.outlet.inlet.plot()

# %% [markdown]
# ### Comparator

# %%
from CADETProcess.comparison import Comparator

comparator = Comparator()
comparator.add_reference(tracer_peak)
comparator.add_difference_metric(
    'NRMSE', tracer_peak, 'outlet.outlet',
)

if __name__ == '__main__':
    comparator.plot_comparison(simulation_results)

# %% [markdown]
# ### Optimization Problem

# %%
from CADETProcess.optimization import OptimizationProblem

optimization_problem = OptimizationProblem('bed_porosity_axial_dispersion')

optimization_problem.add_evaluation_object(process)

optimization_problem.add_variable(
    name='bed_porosity', parameter_path='flow_sheet.column.bed_porosity',
    lb=0.1, ub=0.6,
    transform='auto'
)

optimization_problem.add_variable(
    name='axial_dispersion', parameter_path='flow_sheet.column.axial_dispersion',
    lb=1e-10, ub=0.1,
    transform='auto'
)

optimization_problem.add_evaluator(simulator)

optimization_problem.add_objective(
    comparator,
    n_objectives=comparator.n_metrics,
    requires=[simulator]
)


def callback(simulation_results, individual, evaluation_object, callbacks_dir='./'):
    comparator.plot_comparison(
        simulation_results,
        file_name=f'{callbacks_dir}/{individual.id}_{evaluation_object}_comparison.png',
        show=False
    )


optimization_problem.add_callback(callback, requires=[simulator], )

# %% [markdown]
# ### Optimizer

# %%
from CADETProcess.optimization import U_NSGA3

optimizer = U_NSGA3()
optimizer.n_cores = -3
optimizer.n_max_gen = 3

# %%

results_dir = r"results\parameter_estimation\example_column\column_transport"
# os.chdir(results_dir)
results_folder = "results_bed_porosity_axial_dispersion"

# %%

if __name__ == '__main__':
    optimization_results = optimizer.optimize(
        optimization_problem,
        use_checkpoint=False,
        results_directory=results_dir
    )

    dvc_repo.add(os.path.join(repo_path, results_dir, "checkpoint.h5"))
    dvc_repo.add(os.path.join(repo_path, results_dir,  "results_all.csv"))
    dvc_repo.add(os.path.join(repo_path, results_dir,  "results_last.csv"))
    dvc_repo.add(os.path.join(repo_path, results_dir,  "figures"))
    dvc_repo.add(os.path.join(repo_path, results_dir,  "callbacks"))

    print(git_repo.git.status())
# ```
