from pymle.models import OrnsteinUhlenbeck
from pymle.sim.Simulator1D import Simulator1D
from pymle.TransitionDensity import ExactDensity, EulerDensity
from pymle.fit.AnalyticalMLE import AnalyticalMLE
import numpy as np

# ===========================
# Set the true model (OU) params, to simulate the process
# ===========================
model = OrnsteinUhlenbeck()  

kappa = 0  # rate of mean reversion
mu = 3  # long term level of process
sigma = 2  # volatility

model.params = np.array([kappa, mu, sigma])

# ===========================
# Simulate a sample path (we will fit to this path)
# ===========================
S0 = 10  # initial value of process
T = 1000  # num years of the sample
freq = 1  # observations per year
dt = 1. / freq
seed = 123  # random seed: set to None to get new results each time

simulator = Simulator1D(S0=S0, M=T * freq, dt=dt, model=model).set_seed(seed=seed)
X = simulator.sim_path()



# ===========================
# Fit maximum Likelihood estimators
# ===========================
# Set the parameter bounds for fitting  (kappa, mu, sigma)
param_bounds = [(0, 5), (0, 5), (0, 5)]

# Choose some initial guess for params fit
guess = np.array([0.5, 0.1, 0.4])

# Fit using Exact MLE
exact_est = AnalyticalMLE(X, param_bounds, dt, density=ExactDensity(model)).estimate_params(guess)

# Fit using Euler MLE
euler_est = AnalyticalMLE(X, param_bounds, dt, density=EulerDensity(model)).estimate_params(guess)


print(f'\nExact MLE: {exact_est} \n')

print(f'\nEuler MLE: {euler_est} ')

sum1 = 0
sum2 = 0
sum3 = 0

N = len(X)

for i in range(1, N):
  sum1 += X[i][0]*X[i-1][0]
  sum2 += X[i-1][0] ** 2

theta2 = -1/dt*np.log(sum1/sum2)

for i in range(1, N):
  sum3+=(X[i][0] - X[i-1][0]*np.exp(-1*dt*theta2))**2

theta3 = np.sqrt(2*theta2/(N*(1-np.exp(-2*dt*theta2))) * sum3)

theta_ = [0, theta2, theta3]

print()
print(f'Analitical: {theta_} ')
