import numpy as np
import timm
import torch

from einops import rearrange

from the_well.data import WellDataset
from the_well.benchmark.models import UNetConvNext
from the_well.benchmark.metrics import VRMSE

from funcs import get_stats, rot, flip, well_base_path

device = "cuda"
verbose = False

datasets = ["acoustic_scattering_maze", "active_matter", "gray_scott_reaction_diffusion", "rayleigh_benard", "shear_flow", "turbulent_radiative_layer_2D"]

dataset_name = datasets[2]

print(dataset_name)
print(torch.cuda.get_device_name(torch.cuda.current_device()))

data = WellDataset(
    well_base_path = well_base_path,
    well_dataset_name = dataset_name,
    well_split_name = "test",
    n_steps_input = 4,
    n_steps_output = 0,
    use_normalization = False,
)

F = data.metadata.n_fields
T = data.metadata.n_steps_per_trajectory[0] - (4+0) + 1
N = np.sum(data.metadata.n_trajectories_per_file)
res = data.metadata.spatial_resolution

model = UNetConvNext.from_pretrained("polymathic-ai/UNetConvNext-" + dataset_name).to(device)
model.eval()

mu, sigma = get_stats(data.metadata)

mu    = torch.tensor(mu, device=device)
sigma = torch.tensor(sigma, device=device)


def preprocess(x):
    return (x - mu) / sigma

def postprocess(x):
    return sigma * x + mu

def average(run, start, mode="none", ax=0, n=4):
    
    if mode == "d4" or mode == "z4":
        assert res[0] == res[1]
    if mode == "s1" or mode == "d1":
        assert ax in [0,1]
    if mode == "t2" or mode == "s1":
        assert isinstance(n, int) and n > 0
    
    def forward(xin):
        a = rearrange(xin, "Ti Lx Ly F -> 1 (Ti F) Lx Ly")
        b = model(a)
        c = rearrange(b, "1 (To F) Lx Ly -> To Lx Ly F", F=F)
        return c

    match mode:
        case "none":
            norm = 1.
        case "d4":
            norm = 8.
        case "d2" | "z4":
            norm = 4.
        case "d1":
            norm = 2.
        case "t2" | "s1":
            norm = 1.*n
    
    t = run*T + start
    
    item = data[t]
    x = item["input_fields"].to(device)
    x = preprocess(x)
    
    z = torch.zeros([T-start, res[0], res[1], F], dtype=torch.float32, device=device)
    z[:4] = x
    
    with torch.no_grad():
        for i in range(T-start-4):
            
            match mode:
                case "none":
                    xin = z[i:i+4]
                    xout = forward(xin)
                    z[i+4] = xout[0]
                    
                case "d4":
                    for k in range(4):
                        
                        xin = rot(z[i:i+4], data.metadata, k)
                        xout = forward(xin)
                        z[i+4] = z[i+4] + rot(xout, data.metadata, -k)[0]
                        
                        xin = flip(rot(z[i:i+4], data.metadata, k), data.metadata)
                        xout = forward(xin)
                        z[i+4] = z[i+4] + rot(flip(xout, data.metadata), data.metadata, -k)[0]
                    
                case "z4":
                    for k in range(4):
                        xin = rot(z[i:i+4], data.metadata, k)
                        xout = forward(xin)
                        z[i+4] = z[i+4] + rot(xout, data.metadata, -k)[0]
                    
                case "d2":
                    xin = z[i:i+4]
                    xout = forward(xin)
                    z[i+4] = z[i+4] + xout[0]
            
                    xin = flip(z[i:i+4], data.metadata, 0)
                    xout = forward(xin)
                    z[i+4] = z[i+4] + flip(xout, data.metadata, 0)[0]

                    xin = flip(z[i:i+4], data.metadata, 1)
                    xout = forward(xin)
                    z[i+4] = z[i+4] + flip(xout, data.metadata, 1)[0]

                    xin = flip(flip(z[i:i+4], data.metadata, 0), data.metadata, 1)
                    xout = forward(xin)
                    z[i+4] = z[i+4] + flip(flip(xout, data.metadata, 1), data.metadata, 0)[0]
                    
                case "d1":
                    xin = z[i:i+4]
                    xout = forward(xin)
                    z[i+4] = z[i+4] + xout[0]
            
                    xin = flip(z[i:i+4], data.metadata, ax)
                    xout = forward(xin)
                    z[i+4] = z[i+4] + flip(xout, data.metadata, ax)[0]
                    
                case "t2":
                    shifts = np.random.randint([0, 0], res, [n,2]).tolist()
                    
                    for k in range(n):
                        xin = torch.roll(z[i:i+4], shifts[k], [1,2])
                        xout = forward(xin)
                        z[i+4] = z[i+4] + torch.roll(xout, [-x for x in shifts[k][::-1]], [2,1])[0]
                    
                case "s1":
                    shifts = np.random.randint(0, res[ax], n).tolist()
                    
                    for k in range(n):
                        xin = torch.roll(z[i:i+4], shifts[k], ax+1)
                        xout = forward(xin)
                        z[i+4] = z[i+4] + torch.roll(xout, -shifts[k], ax+1)[0]
                    
            z[i+4] = z[i+4]/norm
            
        z = postprocess(z)
    
    return z


# gray_scott_reaction_diffusion 
args = [
    ( ["none"], {} ),
    ( ["d4"], {} ),
    ( ["z4"], {} ),
    ( ["d2"], {} ),
    ( ["d1"], {'ax': 0} ),
    #( ["d1"], {'ax': 1} ),
    ( ["t2"], {'n': 16} ),
    ( ["t2"], {'n': 8} ),
    ( ["t2"], {'n': 4} ),
    ( ["t2"], {'n': 2} ),
    ( ["t2"], {'n': 1} ),
    #( ["s1"], {'n': 16} ),
    #( ["s1"], {'n': 8} ),
    ( ["s1"], {'n': 4} ),
    ( ["s1"], {'n': 2} ),
    ( ["s1"], {'n': 1} )
]

ts = [10, 50] #[0, 10, 50, 100]

for start in ts:

    errs = torch.zeros([len(args), N, T-start, F], dtype=torch.float32, device=device)
    
    for run in range(N):
    
        z = torch.zeros([len(args), T-start, res[0], res[1], F], dtype=torch.float32, device=device)
        
        for i, (pos_args, kw_args) in enumerate(args):
            z[i] = average(run, start, *pos_args, **kw_args)
    
        for t in range(T-start):
            dat = data[run*T+start+t]['input_fields'][0].to(device)
    
            for j, _ in enumerate(args):
                errs[j, run, t] = VRMSE.eval(z[j, t], dat, meta=data.metadata)
    
    torch.save(errs, "./the_well_errors/" + dataset_name + "_" + str(start) + ".pt")

print(dataset_name, "done")