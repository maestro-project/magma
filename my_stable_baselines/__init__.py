from my_stable_baselines.a2c import A2C
from my_stable_baselines.acer import ACER
from my_stable_baselines.acktr import ACKTR
from my_stable_baselines.deepq import DQN
from my_stable_baselines.her import HER
from my_stable_baselines.ppo2 import PPO2
from my_stable_baselines.td3 import TD3
from my_stable_baselines.sac import SAC

# Load mpi4py-dependent algorithms only if mpi is installed.
try:
    import mpi4py
except ImportError:
    mpi4py = None

if mpi4py is not None:
    from my_stable_baselines.ddpg import DDPG
    from my_stable_baselines.gail import GAIL
    from my_stable_baselines.ppo1 import PPO1
    from my_stable_baselines.trpo_mpi import TRPO
del mpi4py

__version__ = "2.10.0"
