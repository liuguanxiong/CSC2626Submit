from carla.driving_benchmark.experiment_suites import CoRL2017
from carla.driving_benchmark.metrics import Metrics

experiment = CoRL2017("Town01")
metrics_object = Metrics(experiment.metrics_parameters, experiment.dynamic_tasks)
summary = metrics_object.compute("_benchmarks_results/test_CoRL2017_Town01")
print(summary)