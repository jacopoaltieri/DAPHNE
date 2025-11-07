import torch


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
DATA_PATH = r"C:\Users\jacop\Desktop\PhD\DATG\DAPHNE\data\Filo_JPG"
TEMP_PATH = r"C:\Users\jacop\Desktop\PhD\DATG\DAPHNE\data\Temperature_JPG"

# Physics parameters
# Hue to temperature conversion parameters (Logistic curve)
HUE_NORM = 223.0
HUE_AVG = 32.86
HUE_SIGMA = 4.0
HUE_SHIFT = 0.0

GLOBAL_MIN_TEMP = 31.5  # °C
GLOBAL_MAX_TEMP = 34.0  # °C
GLOBAL_PLATE_TEMP = 36.0  # °C temperature of the heating plate under the phantom

KAPPA = 0.00025  # thermal conductivity (W/mmK) (DragonSkin 10 NV)
ALPHA = 0.203  # thermal diffusivity (mm^2/s) (DragonSkin 10 NV)
Q_PRIME = 0.01  # heat source intensity (W/mm) ITS RANDOM FOR NOW, WE NEED TO MEASURE IT!!!
SIGMA = 1  # spatial standard deviation of the source (mm)
RHO = 1.07e-3 # density (g/mm^3) (DragonSkin 10 NV)
C = 1.15 # specific heat capacity (J/gK) (DragonSkin 10 NV)

# Phantom dimensions
X = 30  # mm
Y = 30  # mm
Z = 10  # mm
T = 40 # seconds

X_SOURCE_POS =15  # x coordinate of the source (mm)
Z_S_TRAIN = [1, 3, 5, 7]  # mm, depth of the known sources DUMMY FOR NOW!!!


## Training parameters
BATCH_SIZE = 120_000
LR = 0.001  # learning rate
LAMBDA_PHYSICS = 1  # weight for the physics loss
LAMBDA_CONDITIONS = 1  # weight for the initial and boundary conditions loss
N_EPOCHS = 100
N_COLLOCATION = 10_000  # number of collocation points for the physics loss
N_INITIAL = 1000  # number of initial condition points
N_BC = 1000  # number of boundary condition points

if __name__ == "__main__":
    pass