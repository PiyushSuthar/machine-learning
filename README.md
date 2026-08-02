This repo contains the code for ml projects and experiments.

Most of the code is just from Andrej Karpathy's "Neural Networks: From zero to hero" course, but I have added my own experiments and modifications to the code.
> I'm using the cpu version of pytorch. But if you have a gpu, make sure to comment out the required configuration in the pyproject.toml file to use the gpu version of pytorch.

### Usage
Install the required dependencies using uv.
> Make sure to change the torch package in `pyproject.toml` to the gpu version if you have a gpu.
```bash
uv sync
```

The file paths in the code are relative to the root of the repo. So make sure to run the code from the root of the repo.

I have included the 10m model weights in the repo, you can infer from the model using the following command:
```bash
uv run python -m transformer.infer
```

You can read the file to see the arguments.