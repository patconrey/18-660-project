import os
import logging
import pickle

log = logging.getLogger(__name__)

import torch
from torch.utils.tensorboard import SummaryWriter
import hydra
from omegaconf import DictConfig
from torch.optim import *

from models.mlp import MLP
from models.lr import LR
from utils import *
from utils import seed_everything

from federated_schemes.federated import FederatedScheme


#@hydra.main(config_path="./config/config.yaml", strict=True)
@hydra.main(config_path="./config/config_lr.yaml", strict=True)
def main(cfg: DictConfig):
    os.chdir(cfg.root)
    seed_everything(cfg.seed)
    log.info("\n" + cfg.pretty())
    if cfg.model['type'] == 'mlp':
        model = MLP(**cfg.model.args)
    elif cfg.model['type'] == 'lr':
        if cfg.dataset == 'mnist':
            cfg.model.args['n_features'] = 784
        model = LR(**cfg.model.args)
    elif cfg.model['type'] == 'vgg':
        raise NotImplementedError()
    else:
        raise Exception("Unrecognized model argument")

    writer = SummaryWriter(log_dir=os.path.join(cfg.savedir, "tf"))

    scheme = FederatedScheme(model=model,
                        optimizer=SGD,
                        optimizer_args=cfg.optim.args,
                        federated_type=cfg.fed.type,
                        num_clients=cfg.K,
                        batchsize=cfg.B,
                        fraction=cfg.C,
                        iid=cfg.client_heterogeneity.iid,
                        dataset=cfg.dataset,
                        device=cfg.device,
                        should_use_heterogeneous_E=cfg.client_heterogeneity.should_use_heterogeneous_E,
                        local_epoch=cfg.client_heterogeneity.E,
                        local_epoch_min=cfg.client_heterogeneity.E_min,
                        local_epoch_max=cfg.client_heterogeneity.E_max,
                        should_use_heterogeneous_data=cfg.client_heterogeneity.should_use_heterogeneous_data,
                        writer=writer)

    scheme.fit(cfg.n_round)

    with open(os.path.join(cfg.savedir, "result.pkl"), "wb") as f:
        pickle.dump(scheme.result, f)


if __name__ == "__main__":
    main()
