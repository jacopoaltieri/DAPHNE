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

KAPPA = 0.00025  # thermal conductivity (W/mmK) (DragonSkin 10 NV)
ALPHA = 203  # thermal diffusivity (mm^2/s) (DragonSkin 10 NV)
Q_PRIME = 0.01  # heat source intensity (W/mm) ITS RANDOM FOR NOW, WE NEED TO MEASURE IT!!!
SIGMA = 1.0  # spatial standard deviation of the source (mm)

# Phantom dimensions
X = 300  # mm
Y = 300  # mm
Z = 90  # mm
T = 20 # seconds

X_SOURCE_POS =150  # x coordinate of the source (mm)
Z_S_TRAIN = [10, 30, 50, 70]  # mm, depth of the known sources DUMMY FOR NOW!!!


## Training parameters
LAMBDA_PHY = 0.1  # weight for the physics loss
N_EPOCHS = 2000
N_COLLOCATION = 1000  # number of collocation points for the physics loss

if __name__ == "__main__":
    pass