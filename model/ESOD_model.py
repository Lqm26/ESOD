# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from thop import profile
import torch.nn.functional as F
from Defor_transformer import DeformableTransformer
from Mamba import ModelArgs, Mamba
import torchvision.ops as ops


def autopad(k, p=None, d=1):  
    # kernel, padding, dilation
    if d > 1:
        # actual kernel-size
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        # auto-pad
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p

class DeformableCNN(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.stride = stride 
        self.deform_conv = ops.DeformConv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.offset_conv = nn.Conv2d(in_channels, 18, kernel_size=3, stride=stride, padding=1) 
        self.mask_conv = nn.Conv2d(in_channels, 9, kernel_size=3, stride=stride, padding=1) 

    def forward(self, x):
        offset = self.offset_conv(x) 
        mask = torch.sigmoid(self.mask_conv(x)) 
        return self.deform_conv(x, offset, mask)

class SiLU(nn.Module):  
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)
    
class C2f(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c      = int(c2 * e) 
        self.cv1    = Conv(c1, 2 * self.c, 1, 1)
        self.cv2    = Conv((2 + n) * self.c, c2, 1)
        self.m      = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
    
class Conv(nn.Module):
    default_act = SiLU() 
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv   = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn     = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.act    = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))

class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))   
    
class SPPF(nn.Module):
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_          = c1 // 2
        self.cv1    = Conv(c1, c_, 1, 1)
        self.cv2    = Conv(c_ * 4, c2, 1, 1)
        self.m      = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class SFAB(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(input_dim, output_dim, 3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(output_dim)
        
        self.Bottleneck = Bottleneck(output_dim, output_dim, True, g=1, k=((3, 3), (3, 3)), e=0.5)
        
        self.conv2 = nn.Conv2d(input_dim, output_dim, 3, stride=2, padding=2, dilation=2, bias=False)
        self.bn2 = nn.BatchNorm2d(output_dim)
        
        self.conv3 = DeformableCNN(input_dim, output_dim, stride=2)
        self.bn3 = nn.BatchNorm2d(output_dim)
        
        self.conv4 = Conv(3 * output_dim, output_dim, 1)
        
        self.relu = nn.ReLU(inplace=True)
        

    def forward(self, x):
        x1 = self.bn1(self.conv1(x))
        x1 = self.relu(x1)
        x1 = self.Bottleneck(x1)
        
        x2 = self.bn2(self.conv2(x))
        x2 = self.relu(x2)
        
        x3 = self.bn3(self.conv3(x))
        x3 = self.relu(x3)
        
        out = self.conv4(torch.cat([x1, x2, x3], dim=1))
        
        return out

class FPN(nn.Module):
    def __init__(self, base_channels, base_depth):
        super().__init__()

        self.upsample   = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv_for_upsample1    = C2f(int(base_channels * 16) + base_channels * 8, base_channels * 8, base_depth, shortcut=False)
        self.conv_for_upsample2    = C2f(base_channels * 8 + base_channels * 4, base_channels * 4, base_depth, shortcut=False)
        
        self.down_sample1 = Conv(base_channels * 4, base_channels * 4, 3, 2)
        self.conv_for_downsample1  = C2f(base_channels * 8 + base_channels * 4, base_channels * 8, base_depth, shortcut=False)
        self.down_sample2           = Conv(base_channels * 8, base_channels * 8, 3, 2)
        self.conv_for_downsample2  = C2f(int(base_channels * 16) + base_channels * 8, int(base_channels * 16), base_depth, shortcut=False)
    
    def forward(self, feat1, feat2, feat3):

        P5_upsample = self.upsample(feat3)
        P4 = torch.cat([P5_upsample, feat2], 1)
        P4 = self.conv_for_upsample1(P4)

        P4_upsample = self.upsample(P4)
        P3 = torch.cat([P4_upsample, feat1], 1)
        P3 = self.conv_for_upsample2(P3)

        P3_downsample = self.down_sample1(P3)
        P4 = torch.cat([P3_downsample, P4], 1)
        P4 = self.conv_for_downsample1(P4)

        P4_downsample = self.down_sample2(P4)
        P5 = torch.cat([P4_downsample, feat3], 1)
        P5 = self.conv_for_downsample2(P5)
        
        return P3, P4, P5
    

class LDTR(nn.Module):
    def __init__(self, c3, c4, c5, n_layer=1, vocab_size=5000):
        super().__init__()
        
        dims = [c3, c4, c5]
        
        self.mamba_layers = nn.ModuleList([
            Mamba(ModelArgs(d_model=dim, n_layer=n_layer, vocab_size=vocab_size))
            for dim in dims
        ])

    def _process_scale(self, x, layer, state):

        B, C, H, W = x.shape
        
        # (B, C, H, W) -> (B, H*W, C)
        x_flat = x.flatten(2).transpose(1, 2)

        if state is None:
            out = layer(x_flat, x_flat)
        else:
            state_flat = state.flatten(2).transpose(1, 2)
            out = layer(x_flat, state_flat)
            
        return out.transpose(1, 2).view(B, C, H, W)

    def forward(self, s3, s4, s5, pre_state = None):
        
        inputs = [s3, s4, s5]
        outputs = []

        for i, (x, layer) in enumerate(zip(inputs, self.mamba_layers)):
            current_state = pre_state[i] if pre_state is not None else None
            out = self._process_scale(x, layer, current_state)
            outputs.append(out)

        return tuple(outputs)



class FDDNet(nn.Module):
    def __init__(self, num_class, num_queries = 300):
        super().__init__()
        
        channels = [10, 14, 28, 56, 112, 224]
        self.num_queries = num_queries

        c1, c2, c3, c4, c5, c_neck = channels
        
        self.layer1 = SFAB(c1, c2) 
        self.layer2 = SFAB(c2, c3)
        self.layer3 = SFAB(c3, c4)
        self.layer4 = SFAB(c4, c5)
        self.neck = SFAB(c5, c_neck)

        self.SPPF = SPPF(c_neck, c_neck, k=5)
        
        base_channels = c2
        base_depth = 1
        ch = [base_channels * 4, base_channels * 8, base_channels * 16]

        self.LDTR = LDTR(c4, c5, c_neck)

        self.FPN = FPN(base_channels, base_depth)
        
        self.input_proj = nn.ModuleList(nn.Sequential(nn.Conv2d(x, c_neck, 1, bias=False), nn.BatchNorm2d(c_neck)) for x in ch)
        self.enc_score_head = nn.Linear(c_neck, num_class)

        self.query_pos_head = MLP(4, c_neck, c_neck, num_layers=2)
        self.enc_bbox_head = MLP(c_neck, c_neck, 4, num_layers=2)

        self.transformer = DeformableTransformer(dim = c_neck)

        self.bbox_head = MLP(c_neck, c_neck, 4*16, num_layers=2)
        self.cls_head = MLP(c_neck, c_neck, num_class, num_layers=2)

        # 初始化权重
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def _get_encoder_input(self, x):
        """
        Processes and returns encoder inputs by getting projection features from input and concatenating them.

        Args:
            x (List[torch.Tensor]): List of feature maps from the backbone.

        Returns:
            (tuple): Tuple containing processed features and their shapes.
        """
        x = [self.input_proj[i](feat) for i, feat in enumerate(x)]

        feats = []
        shapes = []
        for feat in x:
            h, w = feat.shape[2:]
            # [b, c, h, w] -> [b, h*w, c]
            feats.append(feat.flatten(2).permute(0, 2, 1))
            shapes.append([h, w])

        feats = torch.cat(feats, 1)
        return feats, shapes
    
    def _generate_anchors(self, shapes, strides, grid_size=0.05,  dtype=torch.float32, device="cpu", eps=1e-2):
        """
        Generates anchor bounding boxes for given shapes with specific grid size and validates them.

        Args:
            shapes (list): List of feature map shapes.
            grid_size (float, optional): Base size of grid cells. Default is 0.05.
            dtype (torch.dtype, optional): Data type for tensors. Default is torch.float32.
            device (str, optional): Device to create tensors on. Default is "cpu".
            eps (float, optional): Small value for numerical stability. Default is 1e-2.

        Returns:
            (tuple): Tuple containing anchors and valid mask tensors.
        """
        anchors = []
        for i, (h, w) in enumerate(shapes):
            stride = strides[i]
            sy = torch.arange(end=h, dtype=dtype, device=device)
            sx = torch.arange(end=w, dtype=dtype, device=device)
            grid_y, grid_x = torch.meshgrid(sy, sx, indexing="ij")
            grid_xy = torch.stack([grid_x, grid_y], -1)  # (h, w, 2)

            valid_WH = torch.tensor([w, h], dtype=dtype, device=device)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / valid_WH  # (1, h, w, 2)
            wh = torch.ones_like(grid_xy, dtype=dtype, device=device) * grid_size * (2.0**i)
            anchors.append(torch.cat([grid_xy, wh], -1).view(-1, h * w, 4))  # (1, h*w, 4)
            
        anchors = torch.cat(anchors, 1)

        return anchors
    
    
    def _get_decoder_input(self, features, shapes, dn_embed=None, dn_bbox=None):
        """
        Generates and prepares the input required for the decoder from the provided features and shapes.

        Args:
            feats (torch.Tensor): Processed features from encoder.
            shapes (list): List of feature map shapes.
            dn_embed (torch.Tensor, optional): Denoising embeddings. Default is None.
            dn_bbox (torch.Tensor, optional): Denoising bounding boxes. Default is None.

        Returns:
            (tuple): Tuple containing embeddings, reference bounding boxes, encoded bounding boxes, and scores.
        """
        bs = features.shape[0]
        anchors = self._generate_anchors(shapes, [8, 16, 32], dtype=features.dtype, device=features.device)

        enc_outputs_scores = self.enc_score_head(features)  # (bs, h*w, nc)

        topk_ind = torch.topk(enc_outputs_scores.max(-1).values, self.num_queries, dim=1).indices
        batch_ind = torch.arange(end=bs, dtype=topk_ind.dtype).unsqueeze(-1).repeat(1, self.num_queries).view(-1)

        top_k_features = features[batch_ind, topk_ind.view(-1)].view(bs, self.num_queries, -1)
        top_k_anchors = anchors[:, topk_ind.view(-1)].view(bs, self.num_queries, -1)
        
        
        topk_ind_v = torch.topk(enc_outputs_scores.max(-1).values, 1600, dim=1).indices.view(-1)
        batch_ind_v = torch.arange(end=bs, dtype=topk_ind_v.dtype).unsqueeze(-1).repeat(1, 1600).view(-1)

        top_k_features_v = features[batch_ind_v, topk_ind_v].view(bs, 1600, -1)

        return top_k_features, top_k_anchors, top_k_features_v, topk_ind
        
        
    def forward(self, x, pre_state=None):
        # backbone
        x = self.layer1(x)
        x = self.layer2(x)
        x3 = self.layer3(x)
        x4 = self.layer4(x3)
        x5 = self.neck(x4)
        x5 = self.SPPF(x5)
        
        if pre_state is None:
            s3, s4, s5 = self.LDTR(x3, x4, x5)
        else:
            s3, s4, s5 = self.LDTR(x3, x4, x5, pre_state)
        
        x3, x4, x5 = self.FPN(s3, s4, s5)
        
        feats = [x3, x4, x5]
        feats, shapes = self._get_encoder_input(feats) 
        top_k_features, top_k_anchors, top_k_features_v, topk_ind = self._get_decoder_input(feats, shapes)
        
        pos_embedding = self.query_pos_head(self.enc_bbox_head(top_k_features) + top_k_anchors)
        
        output = self.transformer(top_k_features, top_k_features_v, pos_embedding)
        _, _, num = feats.shape
        Preds = feats.scatter_(1, topk_ind.unsqueeze(-1).expand(-1, -1, num), output)
        
        cls_pred = self.cls_head(Preds)
        bbox_pred = self.bbox_head(Preds)
        Preds = torch.cat((bbox_pred, cls_pred), dim = -1) # 64 + 11
        
        return Preds, [s3, s4, s5]



if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_shape = (1, 10, 640, 640)

    model = FDDNet(num_class = 11).to(device)
    model.eval()

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    total_params = count_parameters(model)
    print(f"Total trainable parameters: {total_params/1e6:.2f}M")

    dummy_input = torch.randn(*input_shape).to(device)
    flops, _ = profile(model, inputs=(dummy_input,))
    print(f"FLOPs: {flops/1e9:.2f}G")





