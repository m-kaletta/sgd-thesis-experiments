from abc import ABC, abstractmethod

import numpy as np
from scipy.stats import gaussian_kde


class EnsembleInterface(ABC):
    """Abstract interface for ensemble-like objects."""

    @abstractmethod
    def __getitem__(self, idx) -> np.ndarray: 
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    @property
    @abstractmethod
    def shape(self) -> tuple:
        pass

    @property
    @abstractmethod
    def num_steps(self) -> int: 
        pass

    @property
    @abstractmethod
    def time(self) -> np.ndarray: 
        pass

    @property
    def time_len(self) -> float: 
        return self.time[-1] - self.time[0]
    
    @property
    def time_increment(self) -> float:
        return self.time_len / self.num_steps

    @property
    @abstractmethod
    def num_processes(self) -> int: 
        pass

    @property
    @abstractmethod
    def processes(self) -> np.ndarray: 
        pass

    @processes.setter
    @abstractmethod
    def processes(self, proc) -> None:
        pass

    @property
    @abstractmethod
    def increments(self) -> np.ndarray: 
        pass

    @increments.setter
    @abstractmethod
    def increments(self, incr) -> None:
        pass


class Ensemble(EnsembleInterface):
    """User-facing Ensemble object.

    Parameters
    ----------
    time_len : float
        Total simulated time length.
    processes : ndarray (optional if increments are provided)
        2D array of full sample paths (num_processes x num_steps).
    increments : ndarray (optional if processes are provided)
        2D array of increments (num_processes x (num_steps-1)).
    initial_value : float, optional
        Initial value for processes if `increments` is provided.

    Exactly one of `processes` or `increments` must be provided.
    Processes and increments are lazy-initialized by each other when needed.
    """

    def __init__(self, time_len, processes=None, increments=None, initial_value=0.0):
        # increments, respectively processes are only created once actually needed
        if (processes is None) == (increments is None):
            raise ValueError("Exactly one of 'processes' or 'increments' must be provided.")
        if processes is not None:
            self.processes = processes
        elif increments is not None:
            self._initial_value = initial_value
            self.increments = increments
        self._time = np.linspace(0, time_len, self.num_steps)

    def __getitem__(self, idx):
        return self.processes[idx]
    
    def __len__(self):
        return self._num_processes
    
    @property
    def time(self):
        return self._time

    @property
    def shape(self):
        return (self._num_processes, self._num_steps)
    
    @property
    def num_steps(self):
        return self._num_steps
    
    @property
    def num_processes(self):
        return self._num_processes

    @property
    def processes(self):
        if self._processes is None:
            self._processes = np.zeros(self.shape)
            self._processes[:, 0] = self._initial_value
            self._processes[:,1:] = np.cumsum(self._increments, axis=1)
        return self._processes

    @processes.setter
    def processes(self, proc):
        assert proc.ndim == 2
        self._num_processes = proc.shape[0]
        self._num_steps = proc.shape[1]
        self._processes = proc
        self._increments = None

    @property
    def increments(self):
        if self._increments is None:
            self._increments = np.diff(self._processes, axis=1)
        return self._increments

    @increments.setter
    def increments(self, incr):
        assert incr.ndim == 2
        self._increments = incr
        self._processes = None
        self._num_processes = incr.shape[0]
        self._num_steps = incr.shape[1] + 1


class EnsembleHistogram:
    """Histogram-based density estimator over ensemble values.

    Produces bin edges, bin centers and a density matrix of shape
    (bin_number, num_steps). Intended for internal use by visualizer.
    """

    def __init__(self, ensemble, rv_range, std):
        self._bin_number = int(np.sqrt(ensemble.num_processes) * 0.5)
        self._bin_edges = np.linspace(rv_range[0], rv_range[1], self._bin_number + 1)
        self._std = std
        self._density = np.zeros(shape=(self._bin_number, ensemble.num_steps))
        for time_step in range(ensemble.num_steps):
            counts, _ = np.histogram(ensemble[:, time_step], bins=self._bin_edges, density=True)
            self._density[:, time_step] = counts

    def __len__(self):
        return self._bin_number
    
    @property
    def density(self):
        return self._density
    
    @property
    def norm_density(self):
        return self._std * self._density

    @property
    def bin_number(self):
        return self._bin_number
    
    @property 
    def bin_edges(self):
        return self._bin_edges

    @property
    def bin_centers(self):
        return 0.5 * (self._bin_edges[:-1] + self._bin_edges[1:])
        
    @property
    def rv_array(self):
        return self.bin_centers


class EnsembleKDE:
    """KDE-based density estimator over ensemble values.

    Evaluates Gaussian KDE per time step on a fixed random variable (rv) grid.
    """

    
    def __init__(self, ensemble, rv_range, std, resolution=100):
        self._ensemble = ensemble
        self._std = std
        self._density = np.zeros(shape=(resolution, self._ensemble.num_steps))
        self._rv = np.linspace(rv_range[0], rv_range[1], resolution)
        for time_step in range(1, self._ensemble.num_steps):  # at step 1, all we have is the single initial value so the kde would raise an error
            kde_at_step = gaussian_kde(ensemble[:, time_step], bw_method='silverman')
            self._density[:, time_step] = kde_at_step(self._rv)

    @property
    def density(self):
        return self._density
    
    @property
    def norm_density(self):
        return self._std * self._density

    @property
    def rv_array(self):
        return self._rv


class EnsembleDistribution(EnsembleInterface):
    """Lazy wrapper that exposes ensemble statistics and density estimators.

    Provides `mean`, `std`, and accessors `histogram` and `kde` which are
    computed on first access.
    """

    # Lazy initialization: calculate properties once needed
    def __init__(self, ensemble, rv_range):
        self._ensemble = ensemble
        self._rv_range = rv_range
        self._std = None
        self._mean = None
        self._histogram = None
        self._kde = None

    # Ensemble Interface
    def __getitem__(self, idx): 
        return self._ensemble[idx]

    def __len__(self):
        return len(self._ensemble)

    @property
    def shape(self):
        return self._ensemble.shape

    @property
    def num_steps(self): 
        return self._ensemble.num_steps

    @property
    def num_processes(self): 
        return self._ensemble.num_processes
    
    @property
    def time(self):
        return self._ensemble.time

    @property
    def processes(self): 
        return self._ensemble.processes

    @processes.setter
    def processes(self, proc):
        self._ensemble.processes = proc

    @property
    def increments(self): 
        return self._ensemble.increments

    @increments.setter
    def increments(self, incr):
        self._ensemble.increments = incr

    # Distribution specific Interface
    @property
    def rv_range(self):
        return self._rv_range

    @property
    def mean(self):
        if self._mean is None:
            self._mean = np.mean(self._ensemble.processes, axis=0)
        return self._mean

    @property
    def std(self):
        if self._std is None:
            self._std = np.std(self._ensemble.processes, axis=0)
        return self._std

    @property
    def histogram(self):
        if self._histogram is None:
            self._histogram = EnsembleHistogram(self._ensemble, self._rv_range, self.std)
        return self._histogram

    @property 
    def kde(self):
        if self._kde is None:
            self._kde = EnsembleKDE(self._ensemble, self._rv_range, self.std)
        return self._kde

    def get_density_obj(self, method):
        if method not in ['kde', 'kde norm', 'hist', 'hist norm']:
            raise ValueError(f'Unsupported method for density gradient: {method}')
        if method.startswith('hist'):
            return self.histogram
        elif method.startswith('kde'):
            return self.kde
    
    def get_density(self, method):
        density_obj = self.get_density_obj(method)
        if method.endswith('norm'):
            return density_obj.norm_density
        return density_obj.density
