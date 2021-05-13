import os
import logging
import pickle

log = logging.getLogger(__name__)

import torch
from torch.utils.tensorboard import SummaryWriter
import hydra
from omegaconf import DictConfig
from torch.optim import *

from models.lr import LR
from models.vgg import VGG_32x32
from utils import *
from utils import seed_everything

from federated_schemes.federated import FederatedScheme


#@hydra.main(config_path="./config/config.yaml", strict=True)
# @hydra.main(config_path="./config/config_lr.yaml", strict=True)
@hydra.main(config_path="./config/config_vgg.yaml", strict=True)
def main(cfg: DictConfig):
    os.chdir(cfg.root)
    seed_everything(cfg.seed)
    log.info("\n" + cfg.pretty())
    if cfg.model['type'] == 'lr':
        if cfg.dataset == 'mnist':
            cfg.model.args['n_features'] = 784
        model = LR(**cfg.model.args)
    elif cfg.model['type'] == 'vgg':
        model = VGG_32x32()
    else:
        raise Exception("Unrecognized model argument")

    writer = SummaryWriter(log_dir=os.path.join(cfg.savedir, "tf"))
    experiment_id = get_experiment_id_from_cfg(cfg)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    scheme = FederatedScheme(model=model,
                        optimizer=SGD,
                        optimizer_args=cfg.optim.args,
                        federated_type=cfg.fed.type,
                        num_clients=cfg.K,
                        batchsize=cfg.B,
                        fraction=cfg.C,
                        iid=cfg.client_heterogeneity.iid,
                        dataset=cfg.dataset,
                        device=device,
                        should_use_heterogeneous_E=cfg.client_heterogeneity.should_use_heterogeneous_E,
                        local_epoch=cfg.client_heterogeneity.E,
                        local_epoch_min=cfg.client_heterogeneity.E_min,
                        local_epoch_max=cfg.client_heterogeneity.E_max,
                        should_use_heterogeneous_data=cfg.client_heterogeneity.should_use_heterogeneous_data,
                        writer=writer,
                        cfg=cfg)

    scheme.fit(cfg.n_round)

    


if __name__ == "__main__":
    main()
