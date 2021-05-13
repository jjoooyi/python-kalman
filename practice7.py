import os
import math

import torch
import pyro
import pyro.distributions as dist
from pyro.infer.autoguide import AutoDelta
from pyro.optim import Adam
from pyro.infer import SVI, Trace_ELBO, config_enumerate
from pyro.contrib.tracking.extended_kalman_filter import EKFState
from pyro.contrib.tracking.distributions import EKFDistribution
from pyro.contrib.tracking.dynamic_models import NcvContinuous
from pyro.contrib.tracking.measurements import PositionMeasurement

smoke_test = ('CI' in os.environ)
assert pyro.__version__.startswith('1.6.0')
dt = 1e-2
num_frames = 10
dim = 4

# Continuous model
ncv = NcvContinuous(dim, 2.0)

# Truth trajectory
xs_truth = torch.zeros(num_frames, dim)
# initial direction
theta0_truth = 0.0
# initial state
with torch.no_grad():
    xs_truth[0, :] = torch.tensor(
        [0.0, 0.0,  math.cos(theta0_truth), math.sin(theta0_truth)])
    for frame_num in range(1, num_frames):
        # sample independent process noise
        dx = pyro.sample('process_noise_{}'.format(
            frame_num), ncv.process_noise_dist(dt))
        xs_truth[frame_num, :] = ncv(xs_truth[frame_num-1, :], dt=dt) + dx

# Measurements
measurements = []
mean = torch.zeros(2)
# no correlations
cov = 1e-5 * torch.eye(2)
with torch.no_grad():
    # sample independent measurement noise
    dzs = pyro.sample('dzs', dist.MultivariateNormal(
        mean, cov).expand((num_frames,)))
    # compute measurement means
    zs = xs_truth[:, :2] + dzs


def model(data):
    # a HalfNormal can be used here as well
    R = pyro.sample('pv_cov', dist.HalfCauchy(2e-6)) * torch.eye(4)
    Q = pyro.sample('measurement_cov', dist.HalfCauchy(1e-6)) * torch.eye(2)
    # observe the measurements
    pyro.sample('track_{}'.format(i), EKFDistribution(xs_truth[0], R, ncv,
                                                      Q, time_steps=num_frames), obs=data)


guide = AutoDelta(model)  # MAP estimation


optim = pyro.optim.Adam({'lr': 2e-2})
svi = SVI(model, guide, optim, loss=Trace_ELBO(retain_graph=True))

pyro.set_rng_seed(0)
pyro.clear_param_store()

for i in range(250 if not smoke_test else 2):
    loss = svi.step(zs)
    if not i % 10:
        print('loss: ', loss)

# retrieve states for visualization
R = guide()['pv_cov'] * torch.eye(4)
Q = guide()['measurement_cov'] * torch.eye(2)
ekf_dist = EKFDistribution(xs_truth[0], R, ncv, Q, time_steps=num_frames)
states = ekf_dist.filter_states(zs)
