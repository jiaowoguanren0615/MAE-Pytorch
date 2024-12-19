from models import mae_vit_base_patch16, mae_vit_huge_patch14, mae_vit_large_patch16
import torch
import torch.nn as nn

class MaskAutoEncoderClassify(nn.Module):
    def __init__(self, mae_model, num_classes, mae_pretrained_path = None, **kwargs):
        super(MaskAutoEncoderClassify, self).__init__()
        self.mae_model = eval(mae_model + '()')
        self.mae_pretrained_path = mae_pretrained_path

        self.num_classes = num_classes

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.classify = nn.Linear(self.mae_model.embed_dim, num_classes)

    def load_mae_vit_weights(self):
        vit_ckpt = torch.load(self.mae_pretrained_path, weights_only=True)

        mae_model_params = self.mae_model.state_dict()
        vit_params = vit_ckpt['model']

        keys_to_load = [k for k in vit_params.keys() if 'patch_embed' in k or 'blocks' in k or 'norm' in k]

        for key in keys_to_load:
            if key in mae_model_params:
                mae_model_params[key] = vit_params[key]
        for name, param in self.mae_model.named_parameters():
            if 'patch_embed' in name or 'blocks' in name or 'norm' in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
        self.mae_model.load_state_dict(mae_model_params)
        return self.mae_model

    def forward(self, x):
        mae_model = self.load_mae_vit_weights()
        mae_loss, pred, mask = mae_model(x)

        # from [B, L, C] to [B, C, 1]
        pred = self.avgpool(pred.transpose(1, 2))
        pred = pred.flatten(1)
        out = self.classify(pred)
        return out, mae_loss


# if __name__ == '__main__':
#     model = MaskAutoEncoderClassify('mae_vit_base_patch16', 5)
#     X = torch.randn(1, 3, 224, 224)
#     out, mae_loss = model(X)
#     print(out.shape)  ## torch.Size([1, 5])