import copy
import os.path as osp
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.metrics import compute_accuracy
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from .imagenet_templates import IMAGENET_TEMPLATES

import learn2learn as l2l
from meta_learning import MAML
from torch.autograd import grad
import random


_tokenizer = _Tokenizer()


def load_clip_to_cpu(cfg, zero_shot_model=False):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    if not zero_shot_model:
        design_details = {"trainer": 'IVLP',
                          "vision_depth": cfg.TRAINER.PROMETAR.PROMPT_DEPTH_VISION,
                          "language_depth": cfg.TRAINER.PROMETAR.PROMPT_DEPTH_TEXT,
                          "vision_ctx": cfg.TRAINER.PROMETAR.N_CTX_VISION,
                          "language_ctx": cfg.TRAINER.PROMETAR.N_CTX_TEXT}
        model = clip.build_model(state_dict or model.state_dict(), design_details)
    else:
        # Return original CLIP model for generating frozen VL features
        design_details = {"trainer": 'IVLP',
                          "vision_depth": 0,
                          "language_depth": 0, "vision_ctx": 0,
                          "language_ctx": 0}
        model = clip.build_model(state_dict or model.state_dict(), design_details)
        return model
    return model

def gradient_update(model, loss1, loss2, loss3, grad_func=None):
    diff_params = [p for p in model.parameters() if p.requires_grad]

    grad_params1 = grad(loss1,
                        diff_params,
                        retain_graph=True,
                        create_graph=True,
                        allow_unused=True)
    grad_params2 = grad(loss2,
                        diff_params,
                        retain_graph=True,
                        create_graph=True,
                        allow_unused=True)
    grad_params3 = grad(loss3,
                        diff_params,
                        retain_graph=True,
                        create_graph=True,
                        allow_unused=True)

    gradients = []
    grad_counter = 0
    # Handles gradients for non-differentiable parameters
    for name, param in model.named_parameters():
        if param.requires_grad:
            gradient1 = grad_params1[grad_counter]
            gradient2 = grad_params2[grad_counter]
            gradient3 = grad_params3[grad_counter]
            if grad_func:
                if gradient2 == None:
                    gradient = grad_func(gradient1, gradient2, gradient3, name)
                elif gradient3 == None:
                    gradient = grad_func(gradient1, gradient2, gradient3, name)
                else:
                    raise NotImplemented
            grad_counter += 1
        else:
            gradient = None
        gradients.append(gradient)
        
    if gradients is not None:
        params = list(model.parameters())
        if not len(gradients) == len(list(params)):
            msg = 'WARNING:maml_update(): Parameters and gradients have different length. ('
            msg += str(len(params)) + ' vs ' + str(len(gradients)) + ')'
            print(msg)
        for p, g in zip(params, gradients):
            if g is not None:
                p.grad = g.type(p.dtype)

    else:
        print("Gradients are not updated!")
    
    return model

## Original ###
class VNet(nn.Module):
    def __init__(self, cfg, clip_model):
        super().__init__()
        #self.dtype = clip_model.dtype
        vision_ctx_dim = clip_model.visual.conv1.weight.size(0)
        text_ctx_dim = clip_model.ln_final.weight.shape[0]
        n_ctx = cfg.TRAINER.PROMETAR.N_CTX_TEXT
        n_vis = cfg.TRAINER.PROMETAR.N_CTX_VISION

        # self.linear_vision_gamma = nn.ModuleList([nn.Sequential(nn.Linear(vision_ctx_dim*2, vision_ctx_dim//cfg.TRAINER.PROMETAR.DIM_RATE, bias=False),nn.Linear(vision_ctx_dim//cfg.TRAINER.PROMETAR.DIM_RATE, vision_ctx_dim//cfg.TRAINER.PROMETAR.DIM_RATE, bias=False)) for i in range(cfg.TRAINER.PROMETAR.PROMPT_DEPTH_VISION)])
        # self.linear_text_gamma = nn.ModuleList([nn.Sequential(nn.Linear(text_ctx_dim*2, text_ctx_dim//cfg.TRAINER.PROMETAR.DIM_RATE, bias=False),nn.Linear(text_ctx_dim//cfg.TRAINER.PROMETAR.DIM_RATE, text_ctx_dim//cfg.TRAINER.PROMETAR.DIM_RATE, bias=False)) for i in range(cfg.TRAINER.PROMETAR.PROMPT_DEPTH_TEXT)])
        self.linear_vision_gamma = nn.Sequential(nn.Linear(vision_ctx_dim*2*n_vis, vision_ctx_dim), nn.Sigmoid(), nn.Linear(vision_ctx_dim, vision_ctx_dim))
        self.linear_text_gamma = nn.Sequential(nn.Linear(text_ctx_dim*2*n_ctx, text_ctx_dim), nn.Sigmoid(), nn.Linear(text_ctx_dim, text_ctx_dim))       

    def forward(self, gradients1, gradients2, gradients3, param_name):

        if "image_encoder" in param_name:
            linear_gamma = self.linear_vision_gamma
        elif "text_encoder" in param_name or (param_name == "prompt_learner.ctx"):
            linear_gamma = self.linear_text_gamma
            
        d_1, d_2 = gradients1.size()
        # changed_gradients1, changed_gradients2, changed_gradients3 = None, None, None 
        if gradients2 == None:
            input_gradients = torch.cat((gradients1, gradients3), 0)
            gamma_t = torch.sigmoid(linear_gamma(input_gradients.reshape(1,-1).float())).repeat_interleave(2,-1).reshape(d_1, d_2).half()
            changed_gradients = gamma_t*(gradients3)*2
            changed_gradients = gradients1 + changed_gradients
            
        elif gradients3 == None:
            input_gradients = torch.cat((gradients1, gradients2), 0)
            gamma_t = torch.sigmoid(linear_gamma(input_gradients.reshape(1,-1).float())).repeat_interleave(2,-1).reshape(d_1, d_2).half()
            changed_gradients = gamma_t*(gradients2)*2
            changed_gradients = gradients1 + changed_gradients
        else:
            raise NotImplemented

        return changed_gradients
     
    def forward_adapt(self, gradients1, gradients2, param_name):

        if "image_encoder" in param_name:
            linear_gamma = self.linear_vision_gamma
        elif "text_encoder" in param_name or (param_name == "prompt_learner.ctx"):
            linear_gamma = self.linear_text_gamma
            
        d_1, d_2 = gradients1.size()
        # changed_gradients1, changed_gradients2, changed_gradients3 = None, None, None 

        input_gradients = torch.cat((gradients1, gradients2), 0)
        gamma_t = torch.sigmoid(linear_gamma(input_gradients.reshape(1,-1).float())).repeat_interleave(2,-1).reshape(d_1, d_2).half()
        changed_gradients = gamma_t*(gradients2)*2
        changed_gradients = gradients1 + changed_gradients
        # beta_t = torch.sigmoid(linear_beta(gradients.reshape(1, -1))).reshape(d_1, d_2)

        return changed_gradients

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class VLPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        # Make sure Language depth >= 1
        assert cfg.TRAINER.PROMETAR.PROMPT_DEPTH_TEXT >= 1, "In Independent VL prompting, Language prompt depth should be >=1" \
                                                        "\nPlease use VPT trainer if you want to learn only vision " \
                                                        "branch"
        n_ctx = cfg.TRAINER.PROMETAR.N_CTX_TEXT
        ctx_init = cfg.TRAINER.PROMETAR.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init and n_ctx <= 4:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = n_ctx
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # random initialization
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)
        print(f"Independent V-L design")
        print(f'Initial text context: "{prompt_prefix}"')
        print(f"Number of context words (tokens) for Language prompting: {n_ctx}")
        print(f"Number of context words (tokens) for Vision prompting: {cfg.TRAINER.PROMETAR.N_CTX_VISION}")
        self.ctx = nn.Parameter(ctx_vectors)

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)
        # Also create frozen CLIP
        clip_model_temp = load_clip_to_cpu(cfg, True).float().cuda()
        clip_model_temp_image = load_clip_to_cpu(cfg, True)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
            self.ZS_image_encoder = clip_model_temp_image.visual
            # Now pre-compute the frozen VL embeddings
            all_teacher_features = []
            # Using multiple text templates to ensure textual diversity during training
            for single_template in IMAGENET_TEMPLATES:
                x = [single_template.replace("{}", name) for name in classnames]
                x_tokenized = torch.cat([clip.tokenize(p) for p in x])
                text_features = clip_model_temp.encode_text(x_tokenized.cuda())
                all_teacher_features.append(text_features.unsqueeze(1))

        self.text_features_zs = torch.cat(all_teacher_features, dim=1).mean(dim=1)
        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,  # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix
        prompts = self.construct_prompts(ctx, prefix, suffix)

        return prompts


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = VLPromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.total_epochs = cfg.OPTIM.MAX_EPOCH
        self.n_cls = len(classnames)


    def forward(self, image, label_idx=None):
        logit_scale = self.logit_scale.exp()

        with torch.no_grad():
            if label_idx != None:
                text_features_zs = self.prompt_learner.text_features_zs[label_idx, :]
            else:
                text_features_zs = self.prompt_learner.text_features_zs
            text_features_zs = text_features_zs / text_features_zs.norm(dim=-1, keepdim=True)
            image_features_zs = self.prompt_learner.ZS_image_encoder(image.type(self.dtype))
            image_features_zs = image_features_zs / image_features_zs.norm(dim=-1, keepdim=True)

        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        if label_idx != None:
            prompts = prompts[label_idx]
            tokenized_prompts = tokenized_prompts[label_idx]
        text_features = self.text_encoder(prompts, tokenized_prompts)

        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # logits_i = logit_scale * (image_features @ text_features_zs.t())
        # logits_t = logit_scale * (image_features_zs @ text_features.t())
        logits_ce = logit_scale * (image_features @ text_features.t())

        reg_text = F.l1_loss(text_features, text_features_zs,
                                    reduction='mean')
        reg_image = F.l1_loss(image_features, image_features_zs,
                                    reduction='mean')
        
        if self.training:
            return logits_ce, reg_text, reg_image
        else:
            return logits_ce
 

@TRAINER_REGISTRY.register()
class ProMetaR(TrainerX):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.PROMETAR.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        self.classnames = classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.PROMETAR.PREC == "fp32" or cfg.TRAINER.PROMETAR.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)
        self.vnet = VNet(cfg, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        name_to_update = "prompt_learner"

        for name, param in self.model.named_parameters():
            if name_to_update not in name:
                # Make sure that VPT prompts are updated
                if "VPT" in name:
                    param.requires_grad_(True)
                else:
                    param.requires_grad_(False)
            else:
                if "ZS_image_encoder" in name:
                    param.requires_grad_(False)

        # Double check
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")
        print(f"Parameters count: {len(enabled)}")
        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        self.vnet.to(self.device)
        
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("VLPromptLearner", self.model, self.optim, self.sched)

        self.optim_vnet = build_optimizer(self.vnet, cfg.OPTIM_VNET)
        self.sched_vnet = build_lr_scheduler(self.optim_vnet, cfg.OPTIM_VNET)

        # Cosine scheduler
        self.total_epochs = cfg.OPTIM.MAX_EPOCH
        self.step_counter = 1
        # N = cfg.OPTIM.MAX_EPOCH
        
        self.adapt_lr = cfg.TRAINER.PROMETAR.ADAPT_LR
        self.lr_ratio = cfg.TRAINER.PROMETAR.LR_RATIO
        self.fast_adaptation = cfg.TRAINER.PROMETAR.FAST_ADAPTATION

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)

        model = self.model
        optim = self.optim
        optim_vnet = self.optim_vnet
        
        logits_ce, reg_text, reg_image = model(image)
        loss_ce = F.cross_entropy(logits_ce, label)

        optim.zero_grad()
        model = gradient_update(model, loss_ce, reg_text, reg_image, grad_func=self.vnet)
        optim.step()

        maml = MAML(model, lr=self.adapt_lr, first_order=self.fast_adaptation)

        total_losses = []

        label_lst = label.tolist()
        label_set = list(range(len(self.classnames)))
        random.shuffle(label_set)

        N = 2
        l = len(label_set)
        m = (l-1) // N + 1
        cls_li = [label_set[i*m:(i+1)*m] for i in range(N)]
        relabeler = [{y: y_new for y_new, y in enumerate(cls_li[i])} for i in range(N)]
        index = [[j for j, l in enumerate(label_lst) if l in cls_li[i]] for i in range(N)]
        label_li = [torch.tensor([relabeler[i][j] for j in label[index[i]].tolist()]).cuda() for i in range(N)]
        image_li = [image[index[i]].cuda() for i in range(N)]

        for i in range(N):
            image1 = torch.cat([image_li[j] for j in range(N) if j != i])
            label1 = torch.cat([label_li[j] for j in range(N) if j != i])
            cls = [item for j in range(N) if j != i for item in cls_li[j]]
            
            if len(cls_li[i]) < 2 or len(image_li[i]) == 0 or len(image1) == 0:
                continue

            task_model = maml.clone(allow_nograd=True)
            adaptation_logits_ce, adaptation_rag_text, adaptation_rag_image = task_model(image=image1, label_idx=cls)
            adaptation_loss_ce = F.cross_entropy(adaptation_logits_ce, label1)
            
            task_model.adapt_all(adaptation_loss_ce, adaptation_rag_text, adaptation_rag_image, allow_nograd=True, grad_func=self.vnet, allow_unused=True)
            logits2_ce, _, _ = task_model(image=image_li[i], label_idx=cls_li[i])
            
            loss2_ce = F.cross_entropy(logits2_ce, label_li[i])
            total_losses.append(loss2_ce)

        if len(total_losses) != 0:
            loss = torch.mean(torch.stack(total_losses))
            loss = loss* self.lr_ratio
            optim.zero_grad()
            optim_vnet.zero_grad()
            loss.backward()
            optim.step()
            optim_vnet.step()

        for name, param in self.model.named_parameters():
            if "VPT_shallow" in name:
                if "image_encoder.transformer.resblocks." in name:
                    param.requires_grad_(False)
            if "image_encoder.VPT" in name:
                param.requires_grad_(False)

        maml = MAML(model, lr=self.adapt_lr, first_order=self.fast_adaptation)

        total_losses = []

        label_lst = label.tolist()
        label_set = list(range(len(self.classnames)))
        random.shuffle(label_set)

        N = 2
        l = len(label_set)
        m = (l-1) // N + 1
        cls_li = [label_set[i*m:(i+1)*m] for i in range(N)]
        relabeler = [{y: y_new for y_new, y in enumerate(cls_li[i])} for i in range(N)]
        index = [[j for j, l in enumerate(label_lst) if l in cls_li[i]] for i in range(N)]
        label_li = [torch.tensor([relabeler[i][j] for j in label[index[i]].tolist()]).cuda() for i in range(N)]
        image_li = [image[index[i]].cuda() for i in range(N)]

        for i in range(N):
            image1 = torch.cat([image_li[j] for j in range(N) if j != i])
            label1 = torch.cat([label_li[j] for j in range(N) if j != i])
            cls = [item for j in range(N) if j != i for item in cls_li[j]]
            
            if len(cls_li[i]) < 2 or len(image_li[i]) == 0 or len(image1) == 0:
                continue

            task_model = maml.clone(allow_nograd=True)
            adaptation_logits_ce, adaptation_rag_text, adaptation_rag_image = task_model(image=image1, label_idx=cls)
            adaptation_loss_ce = F.cross_entropy(adaptation_logits_ce, label1)
            
            task_model.adapt(adaptation_loss_ce, adaptation_rag_text, allow_nograd=True, grad_func=self.vnet, allow_unused=True)
            logits2_ce, _, _ = task_model(image=image_li[i], label_idx=cls_li[i])
            
            loss2_ce = F.cross_entropy(logits2_ce, label_li[i])
            total_losses.append(loss2_ce)

        if len(total_losses) != 0:
            loss = torch.mean(torch.stack(total_losses))
            loss = loss* self.lr_ratio
            optim.zero_grad()
            optim_vnet.zero_grad()
            loss.backward()
            optim.step()
            optim_vnet.step()

        for name, param in self.model.named_parameters():
            if "VPT_shallow" in name:
                if "image_encoder.transformer.resblocks." in name:
                    param.requires_grad_(True)
            if "image_encoder.VPT" in name:
                param.requires_grad_(True)

        for name, param in self.model.named_parameters():
            if "VPT_shallow" in name:
                if "text_encoder.transformer.resblocks." in name:
                    param.requires_grad_(False)
            if "prompt_learner.ctx" in name:
                param.requires_grad_(False)

        maml = MAML(model, lr=self.adapt_lr, first_order=self.fast_adaptation)

        total_losses = []

        label_lst = label.tolist()
        label_set = list(range(len(self.classnames)))
        random.shuffle(label_set)

        N = 2
        l = len(label_set)
        m = (l-1) // N + 1
        cls_li = [label_set[i*m:(i+1)*m] for i in range(N)]
        relabeler = [{y: y_new for y_new, y in enumerate(cls_li[i])} for i in range(N)]
        index = [[j for j, l in enumerate(label_lst) if l in cls_li[i]] for i in range(N)]
        label_li = [torch.tensor([relabeler[i][j] for j in label[index[i]].tolist()]).cuda() for i in range(N)]
        image_li = [image[index[i]].cuda() for i in range(N)]

        for i in range(N):
            image1 = torch.cat([image_li[j] for j in range(N) if j != i])
            label1 = torch.cat([label_li[j] for j in range(N) if j != i])
            cls = [item for j in range(N) if j != i for item in cls_li[j]]
            
            if len(cls_li[i]) < 2 or len(image_li[i]) == 0 or len(image1) == 0:
                continue

            task_model = maml.clone(allow_nograd=True)
            adaptation_logits_ce, adaptation_rag_text, adaptation_rag_image = task_model(image=image1, label_idx=cls)
            adaptation_loss_ce = F.cross_entropy(adaptation_logits_ce, label1)
            
            task_model.adapt(adaptation_loss_ce, adaptation_rag_image, allow_nograd=True, grad_func=self.vnet, allow_unused=True)
            logits2_ce, _, _ = task_model(image=image_li[i], label_idx=cls_li[i])
            
            loss2_ce = F.cross_entropy(logits2_ce, label_li[i])
            total_losses.append(loss2_ce)

        if len(total_losses) != 0:
            loss = torch.mean(torch.stack(total_losses))
            loss = loss* self.lr_ratio
            optim.zero_grad()
            optim_vnet.zero_grad()
            loss.backward()
            optim.step()
            optim_vnet.step()

        for name, param in self.model.named_parameters():
            if "VPT_shallow" in name:
                if "text_encoder.transformer.resblocks." in name:
                    param.requires_grad_(True)
            if "prompt_learner.ctx" in name:
                param.requires_grad_(True)

        loss_summary = {
            "loss": loss_ce.item(),
            # "meta_loss": meta_loss.item(),
            "acc": compute_accuracy(logits_ce, label)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()
            
        return loss_summary

    def state_dict_weighting(self, main_dict, weightage, prompt_only=False):
        # Average all parameters
        updated_dict = copy.deepcopy(main_dict)
        if not prompt_only:
            for key in main_dict:
                updated_dict[key] = main_dict[key] * weightage
            return updated_dict
        else:
            return main_dict * weightage

    def state_dict_add(self, dict1, dict2, prompt_only=False):
        # Average all parameters
        if not prompt_only:
            modified_dict = dict2
            for key in dict1:
                modified_dict[key] = (modified_dict[key] + dict1[key])
            return modified_dict
        else:
            return dict1 + dict2

    def get_gauss(self, mu, sigma):
        gauss = lambda x: (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
        return gauss

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "prompt_learner.token_prefix" in state_dict:
                del state_dict["prompt_learner.token_prefix"]

            if "prompt_learner.token_suffix" in state_dict:
                del state_dict["prompt_learner.token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)