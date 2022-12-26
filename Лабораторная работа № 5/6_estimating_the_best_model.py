import pandas as pd
import matplotlib.pyplot as plt
from pymle.models.CIR import CIR
from pymle.models.ModifiedCIR import ModifiedCIR
from pymle.models.BrownianMotion import BrownianMotion 
from pymle.models.CEV import CEV
from pymle.models.CKLS import CKLS
from pymle.models.FellerRoot import FellerRoot
from pymle.models.Hyperbolic import Hyperbolic
from pymle.models.Hyperbolic2 import Hyperbolic2
from pymle.models.IGBM import IGBM
from pymle.models.Jacobi import Jacobi
from pymle.models.LinearSDE1 import LinearSDE1
from pymle.models.LinearSDE2 import LinearSDE2
from pymle.models.Logistic import Logistic
from pymle.models.NonLinearSDE import NonLinearSDE
from pymle.models.Pearson import Pearson
from pymle.models.PeralVerhulst import PeralVerhulst
from pymle.models.RadialOU import RadialOU
from pymle.models.Threehalf import Threehalf
from pymle.models.OrnsteinUhlenbeck import OrnsteinUhlenbeck
from pymle.TransitionDensity import *
from pymle.sim.Simulator1D import Simulator1D
from pymle.fit.AnalyticalMLE import AnalyticalMLE
import seaborn as sns
import matplotlib.dates as mdates
import datetime

sns.set_style('whitegrid')

method_name = "modified CIR"
model = ModifiedCIR()
guess = np.asarray([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
param_bounds = [(-5, 5), (-5, 5), (-5, 5), (-5, 5), (-5, 5), (-5, 5), (-5, 5), (-5, 5), (-5, 5), (-5, 5)]

df = pd.read_csv("./DGS10.csv")
df.columns = ['Date', 'DGS10']
df = df.loc[df['DGS10'] != '.']
df['DGS10'] = df['DGS10'].astype('float')


skip = 20
#dt = skip / 252.
#sample = df['DGS10'].values[:-1:skip]
dt = 1
sample = np.array(df['DGS10'].values[:])

# Fit using Kessler MLE
print("\nStarting KESSLER------------------\n")
kessler_est = AnalyticalMLE(sample=sample, param_bounds=param_bounds, dt=dt,
                            density=KesslerDensity(model)).estimate_params(guess)
print(f'\nKessler MLE: {kessler_est} \n')

# Fit using Shoji-Ozaki MLE
print("\nStarting SHOJI-OZAKI------------------\n")
shojioz_est = AnalyticalMLE(sample=sample, param_bounds=param_bounds, dt=dt,
                            density=ShojiOzakiDensity(model)).estimate_params(guess)
print(f'\nShoji-Ozaki MLE: {shojioz_est}')

# Fit using Euler
print("\nStarting EULER------------------\n")
euler_est = AnalyticalMLE(sample=sample, param_bounds=param_bounds, dt=dt,
                          density=EulerDensity(model)).estimate_params(guess)

print(f'\nEuler MLE: {euler_est}')

min_AIC = min(kessler_est.aic, shojioz_est.aic, euler_est.aic)
if min_AIC == kessler_est.aic:
    name = "Kessler Estimation"
elif min_AIC == shojioz_est.aic:
    name = "Shoji-Ozaki Estimation"
elif min_AIC == euler_est.aic:
    name = "Euler Estimation"
print('\n'+name + " is the most optimal estimation for CIR")

#///////////////////////////////////////////////////

models = [CIR(), OrnsteinUhlenbeck(), BrownianMotion(), CEV(), CKLS(), FellerRoot(), Hyperbolic(), Hyperbolic2(), IGBM(), Jacobi(), LinearSDE1(), LinearSDE2(), Logistic(), NonLinearSDE(), Pearson(), PeralVerhulst(), RadialOU(), Threehalf()]
name_ =['CIR', 'OU', 'BM', 'CEV', 'CKLS', 'FellerRoot', 'Hyperbolic', 'Hyperbolic2', 'IGBM', 'Jacobi', 'LinearSDE1', 'LinearSDE2', 'Logistic','NonLinearSDE', 'Pearson', 'PeralVerhulst', 'RadialOU', 'Threehalf']
counter = 0

for model in models:
    print("\nStarting KESSLER------------------\n")
    kessler_est_1 = AnalyticalMLE(sample=sample, param_bounds=param_bounds, dt=dt,
                            density=KesslerDensity(model)).estimate_params(guess)
    print(f'\nKessler MLE: {kessler_est} \n')

    # Fit using Shoji-Ozaki MLE
    print("\nStarting SHOJI-OZAKI------------------\n")
    shojioz_est_1 = AnalyticalMLE(sample=sample, param_bounds=param_bounds, dt=dt,
                            density=ShojiOzakiDensity(model)).estimate_params(guess)
    print(f'\nShoji-Ozaki MLE: {shojioz_est}')

    # Fit using Euler
    print("\nStarting EULER------------------\n")
    euler_est_1 = AnalyticalMLE(sample=sample, param_bounds=param_bounds, dt=dt,
                          density=EulerDensity(model)).estimate_params(guess)

    print(f'\nEuler MLE: {euler_est}')

    # Fit using Kessler MLE
    print("\nStarting KESSLER------------------\n")
    kessler_est_1 = AnalyticalMLE(sample=sample, param_bounds=param_bounds, dt=dt,
                            density=KesslerDensity(model)).estimate_params(guess)
    print(f'\nKessler MLE: {kessler_est} \n')

    # Fit using Shoji-Ozaki MLE
    print("\nStarting SHOJI-OZAKI------------------\n")
    shojioz_est_1 = AnalyticalMLE(sample=sample, param_bounds=param_bounds, dt=dt,
                            density=ShojiOzakiDensity(model)).estimate_params(guess)
    print(f'\nShoji-Ozaki MLE: {shojioz_est}')

    min_AIC1 = min(kessler_est_1.aic, shojioz_est_1.aic, euler_est_1.aic)
    if min_AIC1 == kessler_est_1.aic:
        name1 = "Kessler Estimation"
    elif min_AIC1 == shojioz_est_1.aic:
        name1 = "Shoji-Ozaki Estimation"
    elif min_AIC1 == euler_est_1.aic:
        name1 = "Euler Estimation"
    print('\n'+name1 + " is the most optimal estimation for OU")

    if (min_AIC > min_AIC1):
        min_AIC = min_AIC1
        name = name1
        method_name = name_[counter]

    counter+=1



print("The most optimal model is " + method_name + " " + name)


do_plot = True

model = NonLinearSDE()
euler_est_1 = AnalyticalMLE(sample=sample, param_bounds=param_bounds, dt=dt,
                          density=EulerDensity(model)).estimate_params(guess)
seed = 20
model.params = np.array(euler_est_1.params)
simulator = Simulator1D(S0=2.49, M=1249, dt=dt, model=model).set_seed(seed=seed)
model_sample = simulator.sim_path()

if do_plot:
    fig, ax = plt.subplots()

    df['Date'] = [datetime.datetime.strptime(d, "%Y-%m-%d").date() for d in df['Date']]

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    ax.plot(df['Date'].values, df['DGS10'].values)
    ax.plot(df['Date'].values, model_sample)
    plt.xlabel('Date')
    plt.ylabel('DGS10')
    fig.autofmt_xdate()
    plt.legend (['real data','best model'], loc = 'best')

    plt.show()





