from the_well.utils.download import well_download

base_path = "/PATH/the_well/"

datasets = ['acoustic_scattering_maze', 'active_matter', 'gray_scott_reaction_diffusion', 'rayleigh_benard', 'shear_flow', 'turbulent_radiative_layer_2D']

for dataset in datasets:
    print(dataset)
    well_download(base_path=base_path, dataset=dataset, split="test")

print("done")


# or run
# uv run the-well-download --base-path '~/Projects/the_well' --dataset gray_scott_reaction_diffusion --split test
