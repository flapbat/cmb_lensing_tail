from pysr import PySRRegressor, TemplateExpressionSpec

import matplotlib.pyplot as plt
import numpy as np

import camb

plt.rcParams['font.family'] = 'stixgeneral'

# Load Data

pars     = np.load('../CL_data/parameter_all.npy')  # [H0, ombh2, omch2 ] x 100
lensed   = np.load('../CL_data/lensed_all.npy')     # [C_2, ..., C_5000] x 100 (lensed)
unlensed = np.load('../CL_data/unlensed_all.npy')     # [C_2, ..., C_5000] x 100 (unlensed)

past_ells = 1000
n_ells = 4998 - past_ells

# Truncate to ignore first 1000 l's
y_pysr = lensed[:, past_ells:]/unlensed[:, past_ells:]  #lensing

def moving_average(x):
    val = np.convolve(x, np.ones(500), 'valid') / 500
    return val

# Smoothing the Lensing Tail for Training
do_smoothing = True

if do_smoothing:
    y_pysr[:, 249:-250] = np.apply_along_axis(moving_average, axis = 1, arr = y_pysr[:, :])

# Reformatting data
y_pysr = y_pysr.reshape(-1)
# y_pysr : [par1_c502, par1_c503, ..., par1_c5000, par2_c502, ..., par299_c502, ..., par299_c5000]

# Reformatting data
X_ells = np.array([ell for ell in range(past_ells + 2, 5000)])
pars_pysr = pars[:]
pars_pysr[1] = [pars_pysr[1][i]/0.0224 - 1 for i in range(len(pars_pysr[1]))]
pars_pysr[2] = [pars_pysr[2][i]/0.12 - 1 for i in range(len(pars_pysr[2]))]
pars_pysr[3] = [pars_pysr[3][i]/3.043 - 1 for i in range(len(pars_pysr[3]))]
pars_pysr[4] = [pars_pysr[4][i]/0.965 - 1 for i in range(len(pars_pysr[4]))]


X_pysr = np.zeros((y_pysr.shape[0], 5 + 1))  #for the three/five cosmo parameters plus ells

# Reformatting data
for i in range(100):
    X_pysr[n_ells*i:n_ells*(i+1), :5] = np.tile(pars_pysr[i], n_ells).reshape(n_ells, -1)
    X_pysr[n_ells*i:n_ells*(i+1), -1] = X_ells   #final column is ells

# Template Function

template = TemplateExpressionSpec(
    expressions = ["g"],
    variable_names = ["H0", "x1", "x2", "x5", "x6", "ell"],  #H0, ombh2, omch2, a_s, n_s
    combine = """
        
        beta1 =  Float32(0.6838)  * (1 + Float32(0.6785)  * x1 + Float32(5.1423) * x1 ^2 +Float32(-0.0394) * x2 + Float32(1.1529) * x1 * x2 + Float32(0.00295) * x5 + Float32(0.0772) * x5^2 + Float32(0.02447) * x2 * x5 + Float32(0.21970498) * x1 * x5)
        
        beta2 = Float32(2803.1562) * (1 + Float32(-0.2020) * x1 + Float32(-0.1305) * x2 + Float32(-0.03670) * x1 ^ 2 + Float32(0.1632) * x2 ^2 + Float32(0.01521) * x1 * x2 + Float32(0.01288)  * x5 + Float32(-0.1299) * x5^2 + Float32(-0.0278) * x2 * x5)
        
        beta3 = Float32(3752.098) * (1 + Float32(0.7636) * x1 + Float32(-0.4728) * x2  + Float32(0.3576) * x2^2  + Float32(0.5453) * x1^2 + Float32(0.67459829) * x1 * x2 + Float32(-0.1017) * x5 + Float32(0.1915)  * x5^2 + Float32(1.1921) * x2 * x5 + Float32(0.72633366) * x1 * x5 + Float32(0.01193 ) * x6 + Float32(0.1114) * x6 ^2 + Float32(6.00738696) * x1 * x2 * x5 * x6)
        
        beta4 = Float32(407.009) * (1 + Float32(1.4784) * x1 + Float32(0.2269) * x1 ^2 + Float32(-0.066) * x2 + Float32(-0.1976) *x2^2 + Float32(-2.74281344) * x1 * x2 + + Float32(43.89663989) * x1 ^2 * x2 ^2 + Float32(0.5307) * x5 + Float32(0.1483) * x5^2 + Float32(1.0168) * x2 * x5 +  Float32(5.56638597) * x1 * x5 + Float32(1.3423) * x6 + Float32(1.3534) * x6 ^2 + Float32(2.46017622) * x1 * x6+ Float32(13.78584921) * x1 ^2 * x6 ^2 + Float32(21.98351577) * x1 * x2 * x5 * x6)
        
        alpha = Float32(2.5867) *(1 + Float32(-1.1877) * x1 + Float32(0.700)  * x2 + Float32(0.2784) * x1 ^2  + Float32(0.3064) * x2 ^2 + Float32(-1.01633878) * x1 * x2 + Float32(-0.65334445) * x1 ^2 * x2 ^2 + Float32(1.4088)  * x5 + Float32(0.7482)   * x5^2  + Float32(1.006) *  x5 * x2 + Float32(-1.10231154) * x1 * x5+ Float32(-1.52840887) * x1 ^2 * x5^2 + Float32(0.4324)  * x6 + Float32(+0.2114) * x6 ^2 + Float32(0.3) * x5 * x6 + Float32(0.4246) * x6 * x2  + Float32(0.0382146) * x6 * x1  + Float32(3.07542713)* x1 * x2 * x5 * x6) 
                
        sigma = (1 + exp(-(ell-beta3)/beta4))^(-1)

        poly = ((beta1)*(ell/beta2)^alpha - 1)
        
        1 + poly*sigma 
    """  
)

# PySR Model

# PySR Model

model = PySRRegressor(
    niterations = 100,
    binary_operators = ["+", "-", "*", "pow"],  #allowed operations
    constraints = {'pow': (4, 1), "*": (4, 4)},   #enforces maximum complexities on arguments of operators 
    batching = True, 
    batch_size = 10000, 
    maxsize = 30,
    populations = 20,
    expression_spec = template,
    complexity_of_variables = 3, #global complexity of variables
    procs = 4
)

# Train

model.fit(X_pysr, y_pysr)