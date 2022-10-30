import torch.nn as nn
import torch

class Classifier(nn.Module):
    def __init__(
        self, 
        mode="finetune",
        n_layers=12
    ):
        super(Classifier, self).__init__()
        self.mode = mode

        if self.mode == "readout":
            self.n_weights = n_layers
            self.weight = nn.parameter.Parameter(torch.ones(self.n_weights, 1, 1, 1) / self.n_weights)
            self.layerwise_convolutions = nn.ModuleList([
                nn.Sequential(
                    nn.Conv1d(768, 768, kernel_size=9, padding=4, stride=1),
                    nn.ReLU(),
                ) for _ in range(self.n_weights)
            ])
            self.network = nn.Sequential(
                nn.Conv1d(768, 512, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv1d(512, 256, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv1d(256, 128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv1d(128, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv1d(64, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
            )
            self.out = nn.Linear(32, 1)
        elif self.mode == "finetune":
            self.out = nn.Linear(768, 1)

    
    def forward(self, x):
        if self.mode == "readout":
            layers = []
            for i in range(x.size(0)):
                layers.append(self.layerwise_convolutions[i](x[i, :, :, :].permute(0, 2, 1)).permute(0, 2, 1))
            x = torch.stack(layers, dim=0)
            x = torch.mul(x, self.weight).sum(0)
            x = x.permute(0, 2, 1)
            x = self.network(x)
            x = x.permute(0, 2, 1)
        
        out = self.out(x)
        return out


def get_features(results, mode):
    if mode == "finetune":
        return results["x"]
    elif mode == "readout":
        zeros = torch.zeros_like(results["x"])
        results = [r for r in results["layer_results"]]
        features = [r[0].permute(1, 0, 2) if r[0] is not None else zeros.clone() for r in results]
        features = torch.stack(features, dim=0).squeeze(0)
        return features