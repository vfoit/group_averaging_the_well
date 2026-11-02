# Group Averaging for Physics Applications:
## Accuracy Improvements at Zero Training Cost

This code can be used to replicate the data and figures of paper [2511.09573](https://arxiv.org/abs/2511.09573).

To download [The Well](https://polymathic-ai.org/the_well/) data, run
```BASH
uv run the_well_download.py
```
or use for each required data set:
```BASH
uv run the-well-download --base-path '~/Projects/the_well' --dataset gray_scott_reaction_diffusion --split test
```
Only the `test` split portion is necessary for evluation.

This code works with the following 2D Cartesian [datasets](https://polymathic-ai.org/the_well/datasets_overview/) and [pre-trained models](https://huggingface.co/collections/polymathic-ai/the-well-benchmark-models):
```PYTHON
    datasets = ["acoustic_scattering_maze",
                "active_matter",
                "gray_scott_reaction_diffusion",
                "rayleigh_benard",
                "shear_flow",
                "turbulent_radiative_layer_2D"]
```

Set `well_base_path` of `funcs.py` to the dataset folder.

Run the notebook `the_well.ipynb` to use the datasets interactively. Run `errs_gray_scott.py` to evaluate the entire test set of the `gray_scott_reaction_diffusion` dataset.
