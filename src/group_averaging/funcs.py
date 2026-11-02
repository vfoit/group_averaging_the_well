import torch
import yaml

well_base_path = '~/the_well/datasets/'

def flatten(l):

  out = []
  for item in l:
    if isinstance(item, (list, tuple)):
      out.extend(flatten(item))
    else:
      out.append(item)
  return out


def get_stats(metadata):

    with open(well_base_path + metadata.dataset_name + '/stats.yaml', 'r') as f:
      stats = yaml.safe_load(f)

    match metadata.dataset_name:
        case "acoustic_scattering_maze":
            field_names = ['pressure', 'velocity', 'density', 'speed_of_sound']
        case "active_matter":
            field_names = ['concentration', 'velocity', 'D', 'E']
        case "gray_scott_reaction_diffusion":
            field_names = ['A', 'B']
        case "rayleigh_benard":
            field_names = ['buoyancy', 'pressure', 'velocity']
        case "shear_flow":
            field_names = ['tracer', 'pressure', 'velocity']
        case "turbulent_radiative_layer_2D":
            field_names = ['density', 'pressure', 'velocity']
        case _:
            raise ValueError("Dataset_name not recognized")
    
    mu    = [stats['mean'][x] for x in field_names]
    mu    = flatten(mu)
    sigma = [stats['std'][x] for x in field_names]
    sigma = flatten(sigma)

    return mu, sigma


def rot(x, metadata, k=1):
    '''
    returns the rotated and transformed scalar, vector, and tensors
    '''
    
    assert isinstance(k, int), "k must be integer"
    k = k % 4
    
    match metadata.dataset_name:
        case "acoustic_scattering_maze": # scalar, vector
            order = [0, 2, 1]
            negatives = [1]
            
        case "active_matter": # scalar, vector, tensor, tensor
            order = [0, 2, 1, 6, 5, 4, 3, 10, 9, 8, 7]
            negatives = [1, 4, 5, 8, 9]
            
        case "gray_scott_reaction_diffusion": # scalar, scalar
            order = [0, 1]
            negatives = []
            
        case _:
            raise ValueError("dataset_name can't be rotated")

    m = x.detach().clone()
    order = torch.tensor(order, device=x.device)
    
    for i in range(k):
        m = torch.index_select(m, 3, order)
        m[..., negatives] = -m[..., negatives]
    
    m = torch.rot90(m, k, [1, 2])
    
    return m


def flip(x, metadata, ax=0): # 0 is x-axis, 1 is y-axis
    '''
    returns the inverted and transformed scalar, vector, and tensors
    '''
    
    assert ax in [0, 1], "ax must be 0 (x-axis) or 1 (y-axis)"

    match metadata.dataset_name:
        case "acoustic_scattering_maze": # scalar, vector
            negatives = [ax+1]
             
        case "active_matter": # scalar, vector, tensor, tensor
            negatives = [ax+1, 4, 5, 8, 9]

        case "gray_scott_reaction_diffusion": # scalar, scalar
            negatives = []
            
        case "rayleigh_benard": # scalar, scalar, vector
            #ax = 0
            negatives = [ax+2]
            
        case "shear_flow": 
            negatives = [ax+2]
            
        case  "turbulent_radiative_layer_2D":
            #ax = 0
            negatives = [ax+2]
            
        case _:
            raise ValueError("dataset_name not recognized")
    
    m = torch.flip(x, dims=[ax+1]) # makes a copy of input’s data
    m[..., negatives] = -m[..., negatives]

    return m


def create_trafos(num_vars):
    order = []
    negatives = []
    
    j = 0
    for i in num_vars:
        match i:
            case 0:
                pass
            case 1:
                order.append(j)
                j += 1
            case 2:
                order.extend([j+1, j])
                negatives.append(j)
                j += 2
            case 4:
                order.extend([j+3, j+2, j+1, j])
                negatives.extend([j+1, j+2])
                j += 4
            case _:
                raise ValueError("Unknown quantity")

    return order, negatives

