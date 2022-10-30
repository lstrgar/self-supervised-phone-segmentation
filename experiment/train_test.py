import torch
from torch.nn import BCEWithLogitsLoss
from utils.eval import PrecisionRecallMetric
from utils.dataloader import construct_mask
from models.classifier import get_features
from utils.misc import load_from_checkpoint, save_checkpoint, get_optimizer

def train_test(cfg, model, classifier, trainloader, valloader, testloader, logger):
    device = model.parameters().__next__().device
    logger.info("TRAINING MODEL")
    ckpt, _ = train(model, classifier, trainloader, valloader, cfg, logger, device)
    logger.info("Training complete. Loading best model from checkpoint: {}".format(ckpt))
    model, _, classifier, _, metrics = load_from_checkpoint(cfg, device, ckpt)
    logger.info("Best model's VALIDATION METRICS:")
    for k, v in metrics.items():
        logger.info(f"{k}:")
        for m, s in v.items():
            logger.info(f"\t{m+':':<10} {s:>4.4f}")
    logger.info("Testing best model")
    test(model, classifier, testloader, cfg, logger, device)



def train(model, classifier, trainloader, valloader, cfg, logger, device):
    loss_fn = BCEWithLogitsLoss(
        reduction="none", 
        pos_weight=torch.tensor([cfg.pos_weight]).to(device)
    )

    params_dict = {
        "classifier": classifier.parameters(),
    }
    if cfg.mode == "finetune":
        logger.info("Fine-tuning encoder layers")
        params_dict["model"] = model.parameters()
    else:
        logger.info("Training readout (classifier) weights ONLY")

    optimizer = get_optimizer(cfg, params_dict)

    global_step = 0
    best_rval = 0
    best_model = None

    for e in range(cfg.epochs):
        running_loss = 0.0
        for i, samp in enumerate(trainloader):
            if cfg.mode == "finetune":
                model.train()
            else:
                model.eval()
            classifier.train()
            wavs, _, labels, _, lengths, _ = samp
            mask = construct_mask(lengths, device).float()
            wavs = wavs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            results = model.extract_features(wavs, padding_mask=None)
            features = get_features(results, cfg.mode)
            logits = classifier(features).squeeze()
            if len(logits.shape) == 1:
                logits = logits.unsqueeze(0)
            bce_loss = (loss_fn(logits, labels) * mask).sum() / mask.sum()
            loss = bce_loss
            running_loss += loss.item()
            loss.backward()
            optimizer.step()

            if global_step % cfg.print_interval == cfg.print_interval - 1:
                logger.info("Epoch: {}/{} | Batch: {}/{} | Loss: {:.4f}".format(
                    e+1, cfg.epochs, i+1, len(trainloader), running_loss/cfg.print_interval,
                ))
                running_loss = 0.0

            if cfg.val_interval and global_step % cfg.val_interval == cfg.val_interval - 1:
                logger.info("MODEL VALIDATION: Epoch: {}/{} | Batch: {}/{}".format(e+1, cfg.epochs, i+1, len(trainloader)))
                harsh_metrics_val, lenient_metrics_val = test(model, classifier, valloader, cfg, logger, device)
                if harsh_metrics_val["rval"] > best_rval:
                    best_rval = harsh_metrics_val["rval"]
                    logger.info("New best (harsh) validation rval: {:.4f}".format(best_rval))
                    metrics = {"harsh": harsh_metrics_val, "lenient": lenient_metrics_val}
                    checkpoint_path = save_checkpoint(model, classifier, optimizer, metrics, e+1)
                    best_model = checkpoint_path
                    logger.info("Checkpoint saved to: {}".format(checkpoint_path))
            
            global_step += 1
    
    return best_model, best_rval


def test(model, classifier, dataloader, cfg, logger, device):

    model.eval()
    classifier.eval()
    metric_tracker_harsh = PrecisionRecallMetric(tolerance=cfg.label_dist_threshold, mode="harsh")
    metric_tracker_lenient = PrecisionRecallMetric(tolerance=cfg.label_dist_threshold, mode="lenient")
    sigmoid = torch.nn.Sigmoid()
    logger.info("Evaluating model on {} samples".format(len(dataloader.dataset)))

    for samp in dataloader:
        wavs, segs, labels, _, lengths, _ = samp
        segs = [[*segs[i][0]] + [s[1] for s in segs[i][1:]] for i in range(len(segs))]
        wavs = wavs.to(device)
        labels = labels.to(device)
        results = model.extract_features(wavs, padding_mask=None)
        features = get_features(results, cfg.mode)
        preds = classifier(features).squeeze()
        preds = sigmoid(preds)
        preds = preds > 0.5
        preds = [
            torch.where(preds[i, :lengths[i]] == 1)[0].tolist() for i in range(preds.size(0))
        ]
        metric_tracker_harsh.update(segs, preds)
        metric_tracker_lenient.update(segs, preds)

    logger.info("Computing metrics with distance threshold of {} frames".format(cfg.label_dist_threshold))
    
    tracker_metrics_harsh = metric_tracker_harsh.get_stats()
    tracker_metrics_lenient = metric_tracker_lenient.get_stats()

    logger.info(f"{'SCORES:':<15} {'Lenient':>10} {'Harsh':>10}")
    for k in tracker_metrics_harsh.keys():
        logger.info("{:<15} {:>10.4f} {:>10.4f}".format(k+":", tracker_metrics_lenient[k], tracker_metrics_harsh[k]))

    return tracker_metrics_harsh, tracker_metrics_lenient