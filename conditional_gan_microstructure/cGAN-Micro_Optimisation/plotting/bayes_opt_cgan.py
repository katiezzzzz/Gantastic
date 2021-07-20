from bayes_opt import BayesianOptimization as BO
from slicecgan import *
from taufactor import Solver as TauNet
from stat_analysis.metric_calcs import *
import pybamm
import sys, os
import pickle
from matplotlib.lines import Line2D
from scipy.interpolate import UnivariateSpline

from plotting.bayes_gp_plot import *
import numpy as np
import colorcet as cc
import matplotlib.pyplot as plt
from matplotlib import gridspec

Project_name = 'wt_psd_comp_rep0'
Project_dir = 'trained_generators/NMC_Alej/'

## Data Processing
image_type = 'threephase' # threephase, twophase or colour


isotropic = True
Training = 0 # Run with False to show an image during training
Project_path = mkdr(Project_name, Project_dir, Training)

##Create Networks
n_lbls = 3
netD, netG = slicecgan_rc_nets(Project_path, Training, n_lbls * 2)

netG = netG()
netG.load_state_dict(torch.load(Project_path + '_Gen.pt'))
netG.eval()

netG.cuda()


# optimisation params
lf = 8
nz = 32
batch_size = 1
noise = torch.randn(batch_size, nz, lf, lf, lf).cuda()
sims = []
voltages = []
times = []
def generator(label):
    """
    generators image given the current label
    :param lbl: label for G
    :return: OHE image
    """
    y = torch.zeros((batch_size, n_lbls * 2, lf, lf, lf)).cuda()
    for ch, lbl in enumerate(label):
        y[:, ch] = lbl
        y[:, ch + n_lbls] = 1-lbl
    out = post_proc(netG(noise, y), image_type)
    return out

def opt_func(am_wt, comp):
    """
    calculates current loss
    :param lbl: label for G
    :return: the value of the function to be optimised at position 'lbl'
    """
    label = [am_wt, 0, comp]
    with torch.no_grad():
        imgs = generator(label)
    ac = 0
    for l in range(batch_size):
        img = imgs[l]
        am_wt = volfrac(img, 0)
        porosity = volfrac(img, 1)
        # active_SA = surface_area(img, 0, 1)
        img[img != 1] = 0
        Deff = TauNet(img).solve()
        ac += accessible_capacity(Deff, porosity, am_wt, current=current)/batch_size
    return ac

def accessible_capacity(neg_diff, neg_porosity, neg_am_frac, current=10):
    print(neg_am_frac)
    model = pybamm.lithium_ion.DFN()
    parameter_values = pybamm.ParameterValues(chemistry=pybamm.parameter_sets.Chen2020)
    parameter_values["Current function [A]"] = current
    parameter_values["Negative electrode diffusivity [m2.s-1]"] *= neg_diff
    parameter_values["Negative electrode porosity"] = neg_porosity
    parameter_values["Negative electrode active material volume fraction"] = neg_am_frac
    sim = pybamm.Simulation(model, parameter_values=parameter_values)
    sim.solve([0, tm])
    sims.append(sim)
    voltages.append(sim.solution['Terminal voltage [V]'].entries)
    times.append(sim.solution['Time [s]'].entries)
    return np.sum(sim.solution['Terminal voltage [V]'].entries) * current

def normalize_mass_loading(vf, targ_ml=0.018, p_nmc=4.65, l=265, res = 0.00002):
    mass_per_voxel = res * p_nmc * vf #res**3 * p_nmc * vf / res**2 is the expected mass per cm**2
    return targ_ml / mass_per_voxel

# label bounds
opts = []
p_bounds = {'am_wt': (0, 1),
            'comp': (0, 1)}

# for current in np.arange(8, 9):
#     print(current)
#     optimizer = BO(
#         f=opt_func,
#         pbounds=p_bounds,
#         verbose=2,
#         random_state=1
#     )
#
#     optimizer.maximize(
#         init_points=3,
#         n_iter=0,
#         kappa=5
#     )
#     opts.append(optimizer)
optimizer = BO(
            f=opt_func,
            pbounds=p_bounds,
            verbose=2,
            random_state=1
        )
xs, targets, utilities, vars = [], [], [], []
current=5
ip = 5
k = 5
tm=7200
optimizer.maximize(init_points=ip, n_iter=0, kappa=k)
reps = 5
for i in [1] * reps:
    optimizer.maximize(init_points=0, n_iter=i, kappa=k)
    x, target, utility, var = plot_gp_2D(optimizer, k)
    xs.append(x)
    targets.append(target)
    utilities.append(utility)
    vars.append(var)
# vert
plt.close('all')
st = 3
fin = 5
n = fin-st
fig, axs = plt.subplots(3, n+1, figsize = (3, 3.5), gridspec_kw={"width_ratios":[1*(n)].append(0.05)})
fig.subplots_adjust(wspace=0.01)

# fig.set_size_inches(12, 10)
cmap = [cc.cm.bgy, cc.cm.CET_L8, cc.cm.fire]
sub_xs, sub_targets, sub_utilities, sub_vars = xs[st:fin], targets[st:fin], utilities[st:fin], vars[st:fin]

tmax, tmin = np.array(sub_targets).max(), np.array(sub_targets).min()
umax, umin = np.array(sub_utilities).max(), np.array(sub_utilities).min()
vmax, vmin = np.array(sub_vars).max(), np.array(sub_vars).min()

for i, (x, targ, ut, var) in enumerate(zip(sub_xs, sub_targets, sub_utilities, sub_vars)):
    axs[0, i].imshow(targ, cmap=cmap[0], extent=([0, 20, 85, 95]), vmin=tmin, vmax= tmax)
    axs[0, i].scatter((x[:ip+i+st, 0])*20, 94.5-x[:ip+i+st, 1]*9.5,
                      marker = 'o',s=10, linewidths=1, color=(0,0,0))
    axs[1, i].imshow(ut, cmap=cmap[1], extent=([0, 20, 85, 95]), vmin=umin, vmax= umax)
    x, y = np.where(ut==ut.max())
    print(x, y)
    axs[1, i].scatter((y/100)*20, 95-(x/100)*10, marker = 'o' if i==0 else 'o',
                      s=10, linewidths=1, color=(0,0,0))
    axs[0, i+1].scatter((y/100)*20, 95-(x/100)*10, marker = 'o',
                        s=10, linewidths=1, color=(0,0,0))
    im = axs[2, i].imshow(var, cmap=cmap[2], extent=([0, 20, 85, 95]), vmin=vmin, vmax= vmax)

for ax in axs.ravel():
    ax.set_aspect(2)
    ax.set_xticks([])
    ax.set_yticks([])

for j, lims in enumerate([[tmin, tmax], [umin, umax], [vmin, vmax]]):
    sm = plt.cm.ScalarMappable(cmap=cmap[j], norm=plt.Normalize(vmin=lims[0], vmax=lims[1]))
    sm.set_array([])
    fig.colorbar(sm, cax=axs[j,-1])
    axs[j, -1].set_aspect(18)

#
# cmap = plt.cm.get_cmap('viridis', len(opts))
# fig, axs = plt.subplots(4, 5)

# for i, (ax, opt) in enumerate(zip(axs.ravel(), opts)):
#     targets = [res['target'] for res in opt.res]
#     max_res = np.max(targets)
#     min_res = np.min(targets)
#
#     x = [res['params']['comp'] for res in opt.res]
#     y = [res['params']['psd'] for res in opt.res]
#     c = [res['target'] for res in opt.res]
#
#     ax.scatter(x, y, c=c, cmap='viridis')
#     max = np.argmax(c)
#     # ax.scatter(x[max], y[max], color=(0, 0, 0), s=1.5)
#     if i ==0:
#         ax.set_xlabel('comp')
#         ax.set_ylabel('psd')
#     ax.set_title('current {}A'.format(i*1))
#     sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin=min_res, vmax=max_res))
#     sm.set_array([])
#     fig.colorbar(sm, ax=ax, orientation='vertical', pad=0.2)

# save

# data = {}
# for i, opt in enumerate(opts):
#     data[str(i)] = opt.res
#
# with open('optimisation_runs/optimisers1.pickle', 'b') as handle:
#     pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

plt.figure()
volt = voltages[:9]
time = times[:9]
n = len(volt)
colors = [(150/255, 150/255, 150/255), (0,0,0,)]
for i, (v, t) in enumerate(zip(volt, time)):
    # v = v[1:-1]
    # t = t[1:-1]
    vl = len(v)
    c = colors[0] if i < n-1 else colors[1]
    # x = np.linspace(0,tm,vl)
    s = UnivariateSpline(t, v, s=0.001)
    xspline = np.linspace(0,t.max(),vl*5)
    yspline = s(xspline)
    print(i, vl, c)
    plt.plot(xspline, yspline, '-', c=c, linewidth= 1if i < n-1 else 2.5)
    plt.scatter(t, v,s=16, marker= '' if i < n-1 else 'o', c=[c]*vl, cmap="viridis")
    plt.xticks([0, 3600])

    plt.xlabel('Time (s)')
    plt.ylabel('Voltage (V)')
lines = [Line2D([0], [0], color=(0,0,0,), linewidth=lw, marker= marker)  for marker, lw, c in zip(['', 'o'], [1, 2.5], colors)]
labels = [ 'Previous iterations', 'Current iteration']
plt.legend(lines, labels)