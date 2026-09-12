# Simulation Framework
This repository contains a framework for studying the stochastic gradient descent (SGD) solver on various objective functions. 
It contains the source code implementing the measurements of a bachelor thesis in mathematics, investigating how the SGD method behaves in comparison to the GD method, depending on the noise structure and intensity for different types of objectives. 
While not actively maintained, it can be used to reproduce the results of the thesis or to build on top of it by creating a fork or copying certain parts. 

The `stochplot` folder is a copy of [stochplot](https://github.com/m-kaletta/stochplot). 
The source is included to ensure reproducibility of the thesis results. 

For the usage of the components, see the unittests in `tests\` and the scripts in `visualizations\` and `measurements\`. 

### The Different Objectives
There are four different objective types available with different levels of convexity. 
Ranging from strong convexity through strict convexity, non-strict convexity, to non-convexity.

### The Different Solver Types

The GD method starts at a parameter value, knows the gradient perfectly and then updates the parameters in that exact direction. 
The SGD method is similar, but uses a gradient estimation in each step. 
Typically the estimation arises by selecting a data subsample and computing the gradient on that subsample. 

However, on the parameter domains studied in this thesis, the noise structure arising from subsampling is complex. 
For the strongly convex objective the noise structure arising from the sampling process in a specific linear regression problem was identified. 
Based on the identified noise structure three stochastic solvers were implemented:

1. **PseudoSGD** - The simplest model is a GD method with additive spherical Gaussian noise. The noise is independent of the current parameter value and the solver can be applied to any objective. 
2. **StochasticGradientDescent** - The most complex model is an exact simulation of the SGD method for learning a linear regression model with normally distributed zero-mean inputs and normally distributed outputs using the L2-loss. The noise structure is then state-dependent, meaning that it changes with the parameter value.
3. **ApproximateSGD** - A middle ground, using additive Gaussian noise with a covariance that changes with the parameter value as it would in the exact simulation. 

So all solvers have noise structures with the same first moment (mean). 
StochasticGradientDescent and ApproximateSGD additionally share the same second moment (covariance) but differ in higher moments.

## Installation
1) Clone or download the repository. 
2) Create a virtual environment and install the dependencies (listed in `requirements.txt`)

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

The code was developed and tested with Python 3.13. 


## Reproducing Results
All visualizations and results can be reproduced by running scripts in the `visualizations/` and `measurements/` folder:

```bash
# Example: Create plots of objectives
python visualizations/plot_objectives.py

# Example: Analyze double-well objective behavior
python measurements/eval_double_well.py
```

Results are cached in `.csv` files and plots are saved as `.pdf` files, both saved in their respective script directories. 
The cached files are identified uniquely (apart from hash-collisions) for the configuration of the measurement. 
For usage examples of the core components see `tests/`. 

## Project Structure
The main folder contains the core components:
- `objective.py` - Objective function implementations.
- `solver.py` - Gradient descent solver implementations.
- `helper.py` - Utility functions for matrix construction.
  
and the subfolders:
- `tests/` - unittests for the core components.
- `stochplot/` - A library for stochastic process visualizations, vendored from [stochplot](https://github.com/m-kaletta/stochplot) at release [v0.1.0-beta](https://github.com/m-kaletta/stochplot/releases/tag/v0.1.0-beta), commit 65ec36c. 
- `visualizations/` - Exploratory visualization scripts.
- `measurements/` - Evaluation scripts and configuration management.

More detailed explanations are given below. All plotting scripts create `.pdf` files in their respective directories. 

### Core Components

**`objective.py`** - Objective functions with a mutual interface for value and gradient.
- `StronglyConvex` - Quadratic objective with strong convexity.
- `StrictlyConvex` - Strictly convex (but not strongly convex) objective based on quartic terms.
- `Plateau` - A flat area with flexibly defined edges via another objective.
- `DoubleWell` - Non-convex objective with two minima. 

**`solver.py`** - Gradient descent solver implementations
- `SolveConfiguration` - Encapsulates step sizes and initial conditions for solvers.
- `Solver` - Base class for all solvers, providing a shared interface and implementing the solving process.
- `TrueGradientDescent` - Deterministic gradient descent implementation.
- `GDAnalytical` - Analytical solution for parameter evolution under the GD method on the strongly convex objective. 
- `StochasticSolver` - Base class for all stochastic solvers.
- `PseudoSGD` - GD with additive spherical Gaussian noise imitating SGD.
- `StochasticGradientDescent` - Exact simulation of the SGD method for linear regression with normally distributed zero-mean inputs and normally distributed scalar outputs.
- `ApproximateSGD` - Uses additive Gaussian noise with covariance derived from exact SGD simulation.

  
**`helper.py`** - Utility functions for matrix construction.

### visualizations/
- `plot_objectives.py` - Illustrations of objective functions.
- `plot_solver.py` - Evolution of solver trajectories over iterations.
- `plot_sgd_hist.py` - Comparison of gradient noise across the three stochastic solver types.
- `plot_solver_dist.py` - Shows the evolution of the distribution of the pseudo SGD trajectories.

### measurements/ - modules

**`configuration_manager.py`** - Scenario setup and factories
- `ObjectiveFactory` - Creates standard objective instances.
- `SolverFactory` - Instantiates solver collections.
- `ScenarioCollection` - Bundles objectives and solvers for cohesive evaluation. 
  Contains static methods for the scenario setups used in the different scripts. 

**`measurements/evaluation.py`** - Evaluation framework
- `Evaluator` - Runs Monte Carlo evaluations with caching.
- `EvaluationCase` - Single objective-solver pair evaluation
- `Metrics` - Base class that aggregates metrics via the `@Metrics.metric` decorator. 
  Currently has limited practical value as only one metrics class is actually used. 
  But it allows easy creation of different metric objects and would make any future extension easy. 
- `EvalMetrics` - Creates single metric with several quantities used in the evaluations. 


**`measurements/eval_plotting.py`** - Shared plotting utilities
- `plot_distance_to_min_strip` - one-dimensional scatter plot (strip plot) with boxplot overlay of solver performance. 
- `df_preprocessing` - Processes evaluation results, including standard deviation (std) value matching for comparable solvers and extracting noise std-values from solver names. 
  Provides support for results with an incompatible data structure for the analysis. 
  
### measurements/ - scripts
- `eval_solver.py` - Compares solver performance across objectives and noise levels. Quantifies the performance by the distance of the found parameter to the optimal parameter at the end of solver runs. Creates one-dimensional scatter plots with boxplot overlays. 
- `eval_double_well.py` - Analyzes pseudo-SGD convergence on plateau and double-well objectives across noise levels and dimensions. Visualizes convergence to the optimum via scatter plots with boxplot overlays and histograms. Further reports the probability of converging to the optimal well.


## License
The source code in this project is licensed under the MIT License. 
See the [LICENSE](LICENSE) file for details.
