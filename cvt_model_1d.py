from functools import partial
from itertools import repeat
# from torch._six import container_abcs
import collections.abc as container_abcs
import logging
import os
from collections import OrderedDict

import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange

from timm.models.layers import DropPath, trunc_normal_

from scipy import ndimage


# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, container_abcs.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class Mlp(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self,
                 dim_in,
                 dim_out,
                 num_heads,
                 qkv_bias=False,
                 attn_drop=0.,
                 proj_drop=0.,
                 method='dw_bn',
                 kernel_size=3,
                 stride_kv=1,
                 stride_q=1,
                 padding_kv=1,
                 padding_q=1,
                 with_cls_token=True,
                 **kwargs
                 ):
        super().__init__()
        self.stride_kv = stride_kv
        self.stride_q = stride_q
        self.dim = dim_out
        self.num_heads = num_heads
        # head_dim = self.qkv_dim // num_heads
        self.scale = dim_out ** -0.5
        self.with_cls_token = with_cls_token

        self.conv_proj_q = self._build_projection(
            dim_in, dim_out, kernel_size, padding_q,
            stride_q, 'linear' if method == 'avg' else method
        )
        self.conv_proj_k = self._build_projection(
            dim_in, dim_out, kernel_size, padding_kv,
            stride_kv, method
        )
        self.conv_proj_v = self._build_projection(
            dim_in, dim_out, kernel_size, padding_kv,
            stride_kv, method
        )

        self.proj_q = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.proj_k = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.proj_v = nn.Linear(dim_in, dim_out, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim_out, dim_out)
        self.proj_drop = nn.Dropout(proj_drop)

    def _build_projection(self,
                          dim_in,
                          dim_out,
                          kernel_size,
                          padding,
                          stride,
                          method):
        if method == 'dw_bn':
            proj = nn.Sequential(OrderedDict([
                ('conv', nn.Conv1d(
                    dim_in,
                    dim_in,
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=stride,
                    bias=False,
                    groups=dim_in
                )),
                ('bn', nn.BatchNorm1d(dim_in)),
                ('rearrage', Rearrange('b c f -> b f c')),
            ]))
        elif method == 'avg':
            proj = nn.Sequential(OrderedDict([
                ('avg', nn.AvgPool1d(
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=stride,
                    ceil_mode=True
                )),
                ('rearrage', Rearrange('b c f -> b f c')),
            ]))
        elif method == 'linear':
            proj = None
        else:
            raise ValueError('Unknown method ({})'.format(method))

        return proj

    def forward_conv(self, x, f):
        if self.with_cls_token:
            cls_token, x = torch.split(x, [1, f], 1)

        x = rearrange(x, 'b f c -> b c f')

        if self.conv_proj_q is not None:
            q = self.conv_proj_q(x)
        else:
            q = rearrange(x, 'b c f -> b f c')

        if self.conv_proj_k is not None:
            k = self.conv_proj_k(x)
        else:
            k = rearrange(x, 'b c f -> b f c')

        if self.conv_proj_v is not None:
            v = self.conv_proj_v(x)
        else:
            v = rearrange(x, 'b c f -> b f c')

        if self.with_cls_token:
            q = torch.cat((cls_token, q), dim=1)
            k = torch.cat((cls_token, k), dim=1)
            v = torch.cat((cls_token, v), dim=1)

        return q, k, v

    def forward(self, x, f):
        if (
            self.conv_proj_q is not None
            or self.conv_proj_k is not None
            or self.conv_proj_v is not None
        ):
            q, k, v = self.forward_conv(x, f)
        
        q = rearrange(self.proj_q(q), 'b t (h d) -> b h t d', h=self.num_heads)
        k = rearrange(self.proj_k(k), 'b t (h d) -> b h t d', h=self.num_heads)
        v = rearrange(self.proj_v(v), 'b t (h d) -> b h t d', h=self.num_heads)
        
        
        attn_score = torch.einsum('bhlk,bhtk->bhlt', [q, k]) * self.scale
        attn = F.softmax(attn_score, dim=-1)
        attn = self.attn_drop(attn)

        x = torch.einsum('bhlt,bhtv->bhlv', [attn, v])
        x = rearrange(x, 'b h t d -> b t (h d)')
        
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

    @staticmethod
    def compute_macs(module, input, output):
        # T: num_token
        # S: num_token
        input = input[0]
        flops = 0

        _, T, C = input.shape
        H = W = int(np.sqrt(T-1)) if module.with_cls_token else int(np.sqrt(T))

        H_Q = H / module.stride_q
        W_Q = H / module.stride_q
        T_Q = H_Q * W_Q + 1 if module.with_cls_token else H_Q * W_Q

        H_KV = H / module.stride_kv
        W_KV = W / module.stride_kv
        T_KV = H_KV * W_KV + 1 if module.with_cls_token else H_KV * W_KV

        # C = module.dim
        # S = T
        # Scaled-dot-product macs
        # [B x T x C] x [B x C x T] --> [B x T x S]
        # multiplication-addition is counted as 1 because operations can be fused
        flops += T_Q * T_KV * module.dim
        # [B x T x S] x [B x S x C] --> [B x T x C]
        flops += T_Q * module.dim * T_KV

        if (
            hasattr(module, 'conv_proj_q')
            and hasattr(module.conv_proj_q, 'conv')
        ):
            params = sum(
                [
                    p.numel()
                    for p in module.conv_proj_q.conv.parameters()
                ]
            )
            flops += params * H_Q * W_Q

        if (
            hasattr(module, 'conv_proj_k')
            and hasattr(module.conv_proj_k, 'conv')
        ):
            params = sum(
                [
                    p.numel()
                    for p in module.conv_proj_k.conv.parameters()
                ]
            )
            flops += params * H_KV * W_KV

        if (
            hasattr(module, 'conv_proj_v')
            and hasattr(module.conv_proj_v, 'conv')
        ):
            params = sum(
                [
                    p.numel()
                    for p in module.conv_proj_v.conv.parameters()
                ]
            )
            flops += params * H_KV * W_KV

        params = sum([p.numel() for p in module.proj_q.parameters()])
        flops += params * T_Q
        params = sum([p.numel() for p in module.proj_k.parameters()])
        flops += params * T_KV
        params = sum([p.numel() for p in module.proj_v.parameters()])
        flops += params * T_KV
        params = sum([p.numel() for p in module.proj.parameters()])
        flops += params * T

        module.__flops__ += flops


class Block(nn.Module):

    def __init__(self,
                 dim_in,
                 dim_out,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 **kwargs):
        super().__init__()

        self.with_cls_token = kwargs['with_cls_token']

        self.norm1 = norm_layer(dim_in)
        self.attn = Attention(
            dim_in, dim_out, num_heads, qkv_bias, attn_drop, drop,
            **kwargs
        )

        self.drop_path = DropPath(drop_path) \
            if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim_out)

        dim_mlp_hidden = int(dim_out * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim_out,
            hidden_features=dim_mlp_hidden,
            act_layer=act_layer,
            drop=drop
        )

    def forward(self, x, f):
        res = x
        x = self.norm1(x)
        attn = self.attn(x, f)
        x = res + self.drop_path(attn)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
       
        return x


class ConvEmbed(nn.Module):
    """ Image to Conv Embedding

    """

    def __init__(self,
                 in_chans=2048,
                 out_chans=2048,
                 stride=1,
                 kernal_size=3,
                 padding=1,
                 norm_layer=None):
        super().__init__()

            # nn.Conv1d(in_channels=512*4, out_channels=512, kernel_size=3,
            #           stride=1, padding=1, bias=False),  # should we keep the bias?
            # nn.ReLU(),
            # nn.BatchNorm1d(512),

        self.proj = nn.Conv1d(
            in_chans, out_chans,
            kernel_size=kernal_size,
            stride=stride,
            padding=padding
        )
        self.norm = norm_layer(out_chans) if norm_layer else None

    def forward(self, x):
        B, C, F = x.shape

        x = rearrange(x, 'b c f  -> b f c')

        x = self.proj(x)

        
        if self.norm:
            x = self.norm(x)
        x = rearrange(x, 'b f c -> b c f')

        return x


class VisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self,
                 patch_size=16,
                 patch_stride=16,
                 patch_padding=0,
                 in_chans=3,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 init='trunc_norm',
                 **kwargs):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.rearrage = None

        self.patch_embed = ConvEmbed(
            # img_size=img_size,
            in_chans=2048,
        )

        with_cls_token = kwargs['with_cls_token']
        if with_cls_token:
            self.cls_token = nn.Parameter(
                torch.zeros(1, 1, embed_dim)
            )
        else:
            self.cls_token = None

        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.cls_head = nn.Linear(32, 1) 
        trunc_normal_(self.cls_head.weight, std=0.02)
        blocks = []
        self.dropout = nn.Dropout(p=0.6)
        self.fc1 = nn.Linear(2048, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 1)
        self.relu = nn.ReLU()
        self.drop_out = nn.Dropout(0.7)
        self.sigmoid = nn.Sigmoid( )
        self.m = 2
        
        self.R_EASY = 8 # 95-96 AUUC
        self.R_HARD = 16

        # self.R_EASY = 4
        # self.R_HARD = 8
        self.M = 4

        for j in range(depth):
            blocks.append(
                Block(
                    dim_in=embed_dim,
                    dim_out=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[j],
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    **kwargs
                )
            )
        self.blocks = nn.ModuleList(blocks)

        if self.cls_token is not None:
            trunc_normal_(self.cls_token, std=.02)

        if init == 'xavier':
            self.apply(self._init_weights_xavier)
        else:
            self.apply(self._init_weights_trunc_normal)

    def _init_weights_trunc_normal(self, m):
        if isinstance(m, nn.Linear):
            logging.info('=> init weight of Linear from trunc norm')
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                logging.info('=> init bias of Linear to zeros')
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _init_weights_xavier(self, m):
        if isinstance(m, nn.Linear):
            logging.info('=> init weight of Linear from xavier uniform')
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                logging.info('=> init bias of Linear to zeros')
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)



    def select_topk_embeddings(self, scores, embeddings, k):
        _, idx_DESC = scores.sort(descending=True, dim=1)
        idx_topk = idx_DESC[:, :k]
        idx_topk = idx_topk.unsqueeze(2).expand([-1, -1, embeddings.shape[2]])
        selected_embeddings = torch.gather(embeddings, 1, idx_topk)
        return selected_embeddings

    

    def easy_snippets_mining(self, actionness, embeddings, k_easy):
        select_idx = torch.ones_like(actionness).cuda()
        select_idx = self.dropout(select_idx)

        actionness_drop = actionness * select_idx

        actionness_rev = torch.max(actionness, dim=1, keepdim=True)[0] - actionness
        actionness_rev_drop = actionness_rev * select_idx

        nor_len = embeddings.shape[0]//2
        
        easy_abn= self.select_topk_embeddings(actionness_drop, embeddings, k_easy)
        easy_nor = self.select_topk_embeddings(actionness_rev_drop, embeddings, k_easy)

        har_nor= self.select_topk_embeddings(actionness_drop, embeddings, 32 // self.R_HARD)
        easy_nor = easy_nor[:nor_len,:,:]
        easy_abn = easy_abn[nor_len:,:,:]
        hard_nor = har_nor[:nor_len,:,:]
        return easy_abn, easy_nor, hard_nor

    def hard_snippets_mining(self, actionness, embeddings, k_hard):
        aness_np = actionness.cpu().detach().numpy()
        aness_median = np.median(aness_np, 1, keepdims=True)
        aness_bin = np.where(aness_np > aness_median, 1.0, 0.0)
        
        # print(aness_bin)
        # exit(1)
        hard_ano_mask = np.zeros(aness_bin.shape)
        region_len = 10
        num_threshold = region_len-2
        for b in range(aness_bin.shape[0]):
        ## find pesudo abnormal region with 8 snippets
            for i in range(32-region_len):
                counts = np.count_nonzero(aness_bin[b, i:i+region_len] == 1)
                # print(counts)
                if counts > num_threshold:
                    # if this is a pesudo abnormal region 
                    # apply the reverse values (incorrect low scores with 0 as peusdo labels - should be 1 - so)
                    hard_ano_mask[b, i:i+region_len] = 1 - aness_bin[b, i:i+region_len] 
        
        # all_ind = np.where(aness_bin == 1)
       
        # for i in range(all_ind[0]):
        
        
        # for b in range(aness_bin.shape[0]):
        #     ## find pesudo abnormal region with 8 snippets
        #     for i in range(32-region_len):

        erosion_M = ndimage.binary_erosion(aness_bin, structure=np.ones((1,self.M))).astype(aness_np.dtype)
        erosion_m = ndimage.binary_erosion(aness_bin, structure=np.ones((1,self.m))).astype(aness_np.dtype)
        idx_region_inner = actionness.new_tensor(erosion_m - erosion_M)

        hard_ano_mask = torch.tensor(hard_ano_mask)
        idx_region_inner = torch.add(hard_ano_mask, idx_region_inner)  # combine two type of anomalies


        aness_region_inner = actionness * idx_region_inner
        
        # print(aness_region_inner)
        # exit(1)
        hard_ano = self.select_topk_embeddings(aness_region_inner, embeddings, k_hard)
        
        dilation_m = ndimage.binary_dilation(aness_bin, structure=np.ones((1,self.m))).astype(aness_np.dtype)
        dilation_M = ndimage.binary_dilation(aness_bin, structure=np.ones((1,self.M))).astype(aness_np.dtype)
        idx_region_outer = actionness.new_tensor(dilation_M - dilation_m)
        aness_region_outer = actionness * idx_region_outer
        hard_abn = self.select_topk_embeddings(aness_region_outer, embeddings, k_hard)

        nor_len = embeddings.shape[0]//2
        ## transitional areas between abnormal and normal events has to happen for the abnormal videos
        hard_abn = hard_abn[nor_len:,:,:]
        hard_ano = hard_ano[nor_len:,:,:]

        # hard_ano = torch.cat((hard_ano, hard_abn), dim=0)
        
        return hard_ano, hard_abn
    

    def forward(self, x):
        k_easy = 32 // self.R_EASY
        k_hard = 32 // self.R_HARD

        if len(x.size()) > 3:
            x = x.squeeze(2)
       
        x = self.patch_embed(x)
       
        B, T, F  = x.size()
        
        if B != 1:
            normal_size = B // 2
        else: 
            normal_size = 1

        # print()
       
        cls_tokens = None
        x = rearrange(x, 'b t f -> b f t')
        if self.cls_token is not None:
            # stole cls_tokens impl from Phil Wang, thanks
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
        
        x = self.pos_drop(x)

        for i, blk in enumerate(self.blocks):
            x = blk(x, F)
        
        if self.cls_token is not None:
            cls_tokens, x = torch.split(x, [1, F], 1)
        x = rearrange(x, 'b f c -> b c f')
        

        cls_prob = self.cls_head(cls_tokens)
        cls_prob = self.sigmoid(cls_prob)
        embeddings = x
        
        features = x
        scores = self.relu(self.fc1(features))
        scores = self.drop_out(scores)
        scores = self.relu(self.fc2(scores))
        scores = self.drop_out(scores)
        scores = self.sigmoid(self.fc3(scores))
        
        easy_abn, easy_nor, hard_nor = self.easy_snippets_mining(scores.squeeze(2), embeddings, k_easy)
        hard_abn,hard_abn_2 = self.hard_snippets_mining(scores.squeeze(2), embeddings, k_hard)
        # hard_nor = torch.cat((hard_nor, hard_nor_2), dim=0)
        # print(easy_abn.shape)
        # print(easy_nor.shape)
        # print(hard_nor.shape)
        # print(hard_abn.shape)
        # exit(1)
        contrast_pairs = {
            'E_Abn': easy_abn,
            'E_Nor': easy_nor,
            'H_Abn': hard_abn,
            'H_Abn2': hard_abn_2,
            'H_Nor_top_k': hard_nor
        }
       
        return x, cls_tokens, cls_prob,  scores, contrast_pairs, embeddings


class ConvolutionalVisionTransformer(nn.Module):
    def __init__(self,
                 in_chans=3,
                 num_classes=1000,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 init='trunc_norm',
                 spec=None):
        super().__init__()
        self.num_classes = num_classes

        self.num_stages = spec['NUM_STAGES']
        for i in range(self.num_stages):
            kwargs = {
                'patch_size': spec['PATCH_SIZE'][i],
                'patch_stride': spec['PATCH_STRIDE'][i],
                'patch_padding': spec['PATCH_PADDING'][i],
                'embed_dim': spec['DIM_EMBED'][i],
                'depth': spec['DEPTH'][i],
                'num_heads': spec['NUM_HEADS'][i],
                'mlp_ratio': spec['MLP_RATIO'][i],
                'qkv_bias': spec['QKV_BIAS'][i],
                'drop_rate': spec['DROP_RATE'][i],
                'attn_drop_rate': spec['ATTN_DROP_RATE'][i],
                'drop_path_rate': spec['DROP_PATH_RATE'][i],
                'with_cls_token': spec['CLS_TOKEN'][i],
                'method': spec['QKV_PROJ_METHOD'][i],
                'kernel_size': spec['KERNEL_QKV'][i],
                'padding_q': spec['PADDING_Q'][i],
                'padding_kv': spec['PADDING_KV'][i],
                'stride_kv': spec['STRIDE_KV'][i],
                'stride_q': spec['STRIDE_Q'][i],
            }

            stage = VisionTransformer(
                in_chans=in_chans,
                init=init,
                act_layer=act_layer,
                norm_layer=norm_layer,
                **kwargs
            )
            setattr(self, f'stage{i}', stage)

            in_chans = spec['DIM_EMBED'][i]

        dim_embed = spec['DIM_EMBED'][-1]
        self.norm = norm_layer(dim_embed)
        self.cls_token = spec['CLS_TOKEN'][-1]

        # Classifier head
        self.head = nn.Linear(dim_embed, num_classes) if num_classes > 0 else nn.Identity()
        trunc_normal_(self.head.weight, std=0.02)

    def init_weights(self, pretrained='', pretrained_layers=[], verbose=True):
        if os.path.isfile(pretrained):
            pretrained_dict = torch.load(pretrained, map_location='cpu')
            logging.info(f'=> loading pretrained model {pretrained}')
            model_dict = self.state_dict()
            pretrained_dict = {
                k: v for k, v in pretrained_dict.items()
                if k in model_dict.keys()
            }
            need_init_state_dict = {}
            for k, v in pretrained_dict.items():
                need_init = (
                        k.split('.')[0] in pretrained_layers
                        or pretrained_layers[0] is '*'
                )
                if need_init:
                    if verbose:
                        logging.info(f'=> init {k} from {pretrained}')
                    if 'pos_embed' in k and v.size() != model_dict[k].size():
                        size_pretrained = v.size()
                        size_new = model_dict[k].size()
                        logging.info(
                            '=> load_pretrained: resized variant: {} to {}'
                            .format(size_pretrained, size_new)
                        )

                        ntok_new = size_new[1]
                        ntok_new -= 1

                        posemb_tok, posemb_grid = v[:, :1], v[0, 1:]

                        gs_old = int(np.sqrt(len(posemb_grid)))
                        gs_new = int(np.sqrt(ntok_new))

                        logging.info(
                            '=> load_pretrained: grid-size from {} to {}'
                            .format(gs_old, gs_new)
                        )

                        posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)
                        zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                        posemb_grid = scipy.ndimage.zoom(
                            posemb_grid, zoom, order=1
                        )
                        posemb_grid = posemb_grid.reshape(1, gs_new ** 2, -1)
                        v = torch.tensor(
                            np.concatenate([posemb_tok, posemb_grid], axis=1)
                        )

                    need_init_state_dict[k] = v
            self.load_state_dict(need_init_state_dict, strict=False)

    @torch.jit.ignore
    def no_weight_decay(self):
        layers = set()
        for i in range(self.num_stages):
            layers.add(f'stage{i}.pos_embed')
            layers.add(f'stage{i}.cls_token')

        return layers

    def forward_features(self, x):
        for i in range(self.num_stages):
           
            x, cls_tokens = getattr(self, f'stage{i}')(x)

        if self.cls_token:
            x = self.norm(cls_tokens)
            x = torch.squeeze(x)
        else:
            x = rearrange(x, 'b c h w -> b (h w) c')
            x = self.norm(x)
            x = torch.mean(x, dim=1)

        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)

        return x





spec ={
    'INIT': 'trunc_norm',
    "NUM_STAGES": 1,
    'PATCH_SIZE': [7, 3, 3],
    'PATCH_STRIDE': [4, 2, 2],
    'PATCH_PADDING': [2, 1, 1],
    'DIM_EMBED': [32, 32, 32],
    'NUM_HEADS': [8, 8, 8],
    'DEPTH': [6, 6, 10],
    'MLP_RATIO': [4.0, 4.0, 4.0],
    'ATTN_DROP_RATE': [0.0, 0.0, 0.0],
    'DROP_RATE': [0.0, 0.0, 0.0],
    'DROP_PATH_RATE': [0.0, 0.0, 0.1],
    'QKV_BIAS': [True, True, True],
    'CLS_TOKEN': [True, True, True],
    'POS_EMBED': [False, False, False],
    'QKV_PROJ_METHOD': ['dw_bn', 'dw_bn', 'dw_bn'],
    'KERNEL_QKV': [3, 3, 3],
    'PADDING_KV': [1, 1, 1],
    'STRIDE_KV': [2, 2, 2],
    'PADDING_Q': [1, 1, 1],
    'STRIDE_Q': [1, 1, 1],
    }

    
def get_model_transformer():

    # msvit = ConvolutionalVisionTransformer(
    #     in_chans=3,
    #     num_classes=20,
    #     act_layer=QuickGELU,
    #     norm_layer=partial(LayerNorm, eps=1e-5),
    #     spec = SPEC
    # )

    kwargs = {
                'patch_size': spec['PATCH_SIZE'][0],
                'patch_stride': spec['PATCH_STRIDE'][0],
                'patch_padding': spec['PATCH_PADDING'][0],
                'embed_dim': spec['DIM_EMBED'][0],
                'depth': spec['DEPTH'][0],
                'num_heads': spec['NUM_HEADS'][0],
                'mlp_ratio': spec['MLP_RATIO'][0],
                'qkv_bias': spec['QKV_BIAS'][0],
                'drop_rate': spec['DROP_RATE'][0],
                'attn_drop_rate': spec['ATTN_DROP_RATE'][0],
                'drop_path_rate': spec['DROP_PATH_RATE'][0],
                'with_cls_token': spec['CLS_TOKEN'][-1],
                'method': spec['QKV_PROJ_METHOD'][0],
                'kernel_size': spec['KERNEL_QKV'][0],
                'padding_q': spec['PADDING_Q'][0],
                'padding_kv': spec['PADDING_KV'][0],
                'stride_kv': spec['STRIDE_KV'][0],
                'stride_q': spec['STRIDE_Q'][0],
            }
    in_chans=3
    stage = VisionTransformer(
        in_chans=in_chans,
        init='trunc_norm',
        act_layer=nn.GELU,
        norm_layer=partial(LayerNorm, eps=1e-5),
        **kwargs
            )
    stage = stage.cuda()
    # img = torch.ones([8, 32, 2048])
    # img = img.cuda()
    # out = stage(img)
    # print("Shape of out :", out.shape)  # [B, num_classes]
    return stage