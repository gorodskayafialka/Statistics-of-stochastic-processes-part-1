from pymle.models import OrnsteinUhlenbeck
from pymle.sim.Simulator1D import Simulator1D
from pymle.TransitionDensity import ExactDensity
from pymle.fit.AnalyticalMLE import AnalyticalMLE
import numpy as np

# ===========================
# Set the true model (OU) params, to simulate the process
# ===========================
model = OrnsteinUhlenbeck()  

kappa = 3  # rate of mean reversion
mu = 1  # long term level of process
sigma = 2  # volatility

model.params = np.array([kappa, mu, sigma])

# ===========================
# Simulate a sample path (we will fit to this path)
# ===========================
S0 = 10  # initial value of process
T = 1000  # number of days of the sample
freq = 1  # observations per day
dt = 1. / freq
seed = 123  # random seed: set to None to get new results each time

simulator = Simulator1D(S0=S0, M=T * freq, dt=dt, model=model).set_seed(seed=seed)
sample = simulator.sim_path()

# ===========================
# Fit maximum Likelihood estimators
# ===========================
# Set the parameter bounds for fitting  (kappa, mu, sigma)
param_bounds = [(0, 6), (0, 5), (0, 5)]

# Choose some initial guess for params fit
guess = np.array([0.5, 0.1, 0.4])

# Fit using Exact MLE
exact_est = AnalyticalMLE(sample, param_bounds, dt, density=ExactDensity(model)).estimate_params(guess)

print(f'\nExact MLE: {exact_est}')