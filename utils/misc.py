from models.classifier import Classifier
from utils.dataloader import LAYERS
from fairseq.checkpoint_utils import load_model_ensemble_and_task
import os, torch


def load_model(cfg, device):
    model, _, _ = load_model_ensemble_and_task(
        [os.path.join(cfg.base_ckpt_path)],
    )
    model = model[0]
    model = model.to(device)
    layers = LAYERS
    return model, layers


def get_optimizer(cfg, params_dict):
    params = list(params_dict["classifier"])
    if cfg.mode == "finetune":
        params += list(params_dict["model"])
    if cfg.optim_type == "adam":
        optimizer = torch.optim.Adam(
            params=params,
            lr=cfg.lr,
            weight_decay=cfg.weight_decay, 
            betas=(cfg.beta1, cfg.beta2) 
        )
    else:
        raise ValueError("Provided optimizer not supported")
    return optimizer


def load_from_checkpoint(cfg, device, checkpoint_path=None):
    model, layers = load_model(cfg, device)
    classifier = Classifier(mode=cfg.mode, n_layers=12)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model"])
    classifier.load_state_dict(checkpoint["classifier"])
    classifier = classifier.to(device)
    optimizer = get_optimizer(
        cfg, 
        {
            "classifier": classifier.parameters(), 
            "model": model.parameters()
        }
    )
    return model, layers, classifier, optimizer, checkpoint["metrics"]


def save_checkpoint(model, classifier, optimizer, metrics, epoch):
    path = os.path.join(os.getcwd(), "model.ckpt")
    torch.save(
        {
            "model": model.state_dict(),
            "classifier": classifier.state_dict(),
            "optimizer": optimizer.state_dict(),
            "metrics": metrics,
            "epoch": epoch,
        },
        path,
    )
    return path