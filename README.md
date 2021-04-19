# Installation
To run the project, you'll need to install the necessary packages. Those packages are detailed in `environment.yml` at the root of the project. I used conda to set up my environment, so I'd suggest doing the same at least until everybody gets it up and running. If you're going to use conda, run the following command:
`conda env create --file environment.yml`. I'm _nearly_ positive this will make a conda environment called `660-project` with the necessary packages.

# Running the default
The project is structured so that you have to run the command `python run.py` to begin federated training. In the `run` script, you specify a couple important things:
- The config file to use for training (e.g., `@hydra.main(config_path="./config/config.yaml", strict=True)`).
- The type of client models we'll be training (e.g., `model = MLP(**cfg.model.args)`)
- The federated training scheme (e.g., `scheme = FedAvg(...)`)

# Client/Server
The project is organized so that we can treat clients and aggregation servers as modular components.

## Client
The client component is in `models/client.py`. The main functionality that a client does is its local updates. Every communication round, a selected number of clients perform `E`  epochs. Depending on the details of the experiment, `E` may be fixed across all clients or it may be set randomly. The client component simply loops over `E` and performs the appropriate local updates.

## Server
The server component is in `models/server.py`. The main functionality of the server component is to aggregate the local models and perform the validation on the new global model. The aggregation scheme is defined by the training scheme (e.g., `FedAvg` aggregates the client models in a different way than `FedNova`). The validation functionality just evaluates the new global model on the test set.

# Federated Schemes
In the `run` script, you need to specify which scheme you want to run. The two available options are `FedAvg` and `FedNova`. The python scripts for these two options are in `federated_schemes/`. These scripts are responsible for two important things:
- Creating the datasets for each client.
- Orchestrating the training and validation.

## Creating Datasets
Each client will have a data shard. There isn't any overlap between the samples typically. The distribution of the dataset as well as its size are defined in the `config` directory and described later in this README. Currently, the project is set up to use the MNIST dataset. This is not a dataset that's used in the FedNova paper, but it was convenient to use here. Perhaps we can show results for it as an additional experiment?

## Orchestration
While the client and server components have their respective training or validation schemes, I figured it'd be best to have scripts for each scheme in case there are things that we'll need to change to make each one work.

# Logging
The repository this is based off of used Tensorboard, so that's what's used here. I think it might be best to switch over to Weights and Biases (wandb.ai) so everybody would have access to all of the logs, but that's low priority. To run Tensorboard, you'll need to make sure you have the project's conda environment activated. Then, it should just be as simple as running `tensorboard --logdir output/`

# Config
The config documents in `config/` are set up to house all of the hyperparameters for an experiment.

## Models
For debugging purposes, I've been using an MLP as the client model. All of the hyperparameters for that model are defined in `config/model/mlp.yaml`. As the project grows, we'll need to include the hyperparameters for the models defined in the FedNova paper.

## Data Heterogeneity
FedNova attempts to normalize out computation heterogeneity introduced at the clients in the federated learning setting. There are parameters that control the degree of heterogeneity in `config/config.yaml`. The key `client_heterogeneity` contains all of these parameters.

- `iid`: Determines if each client is given an IID sample of data or not. In the MNIST case, this amounts to each client being given a representative sample of the digits 0-9, when `iid=True`. If `iid=False`, then each client may be given samples from just one or two classes.
- `should_use_heterogeneous_data`: If true, this means that each client is given a random amount of data. The data sizes follow a power law, similar to the FedNova paper. The distribution isn't explicitly defined in the paper (from what I've seen!), so I had to pick the `a` parameter of the power law distribution. 
- `should_use_heterogeneous_E`: If true, each client will select its number of local updates by sampling from the uniform distribution defined on `U(E_min, E_max)`, where `E_min` and `E_max` are defined in the config file. If `should_use_heterogeneous_E` is false, then each client will do `E` local updates, where `E` is defined in the config file as well.