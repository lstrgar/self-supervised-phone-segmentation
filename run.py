from experiment.train_test import train_test
from models.classifier import Classifier
from utils.misc import load_model
from utils.dataloader import get_dloaders
import hydra, os, logging, random, numpy as np, torch

@hydra.main(config_path='./config', config_name='conf')
def main(cfg):

    logger = logging.getLogger(__name__)

    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    g = torch.Generator()
    g.manual_seed(cfg.seed)
    torch.use_deterministic_algorithms(True)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    torch.autograd.set_detect_anomaly(True)

    device = torch.device('cuda')

    logger.info("Model base checkpoint is {}".format(cfg.base_ckpt_path))

    logger.info("Instantiating model...")
    model, layers = load_model(cfg, device)
    classifier = Classifier(mode=cfg.mode, n_layers=12)

    model = model.to(device)
    classifier = classifier.to(device)

    trainloader, valloader, testloader = get_dloaders(
        cfg=cfg, layers=layers, logger=logger, g=g)
    logger.info("Dataset is {}".format(cfg.data))

    train_test(cfg, model, classifier, trainloader, valloader, testloader, logger)

if __name__ == "__main__":
    main()