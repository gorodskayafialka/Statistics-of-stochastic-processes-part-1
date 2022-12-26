from pymle.models import CKLS
from pymle.sim.Simulator1D import Simulator1D
from pymle.TransitionDensity import ExactDensity, EulerDensity
from pymle.fit.AnalyticalMLE import AnalyticalMLE
import numpy as np
import scipy.stats as st

# ===========================
# Set the true model (OU) params, to simulate the process
# ===========================
model = CKLS()  
model.params = np.array([1, 2, 0.5, 0.3])

S0 = 2  # initial value of process
T = 2  # num years of the sample
freq = 1000  # observations per year
dt = 1. / freq
seed = None#123  # random seed: set to None to get new results each time

simulator = Simulator1D(S0=S0, M=T * freq, dt=dt, model=model).set_seed(seed=seed)
sample = simulator.sim_path()

param_bounds = [(0, 5), (0, 5), (0, 5), (0, 5)]

guess = np.array([0.5, 0.1, 0.4, 0.1])

theta1 = []
theta2 = []
theta3 = []
theta4 = []

for i in range (0, 100):
    print(i)
    simulator = Simulator1D(S0=S0, M=T * freq, dt=dt, model=model).set_seed(seed=seed)
    sample = simulator.sim_path()
    euler_est = AnalyticalMLE(sample, param_bounds, dt, density=EulerDensity(model)).estimate_params(guess)
    theta1.append(euler_est.params[0])
    theta2.append(euler_est.params[1])
    theta3.append(euler_est.params[2])
    theta4.append(euler_est.params[3])

print("th1: ")
print(st.t.interval(alpha=0.95, df=len(theta1)-1, loc=np.mean(theta1), scale=st.sem(theta1)) )
print("th2: ")
print(st.t.interval(alpha=0.95, df=len(theta2)-1, loc=np.mean(theta2), scale=st.sem(theta2)) )
print("th3: ")
print(st.t.interval(alpha=0.95, df=len(theta3)-1, loc=np.mean(theta3), scale=st.sem(theta3)) )
print("th4: ")
print(st.t.interval(alpha=0.95, df=len(theta4)-1, loc=np.mean(theta4), scale=st.sem(theta4)) )

print(f'\nEuler MLE: {euler_est} ')





