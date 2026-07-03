                  
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Callable, List, Optional, Tuple

from var_d2m_model import VARD2MFFN, D2MRouter


LossFn = Callable[[torch.Tensor, int], torch.Tensor]                                  


def build_var_loss_fn(var, vae, mode: str, args) -> LossFn:
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
       
    if mode == "custom_stub":
        def _stub(label_B: torch.Tensor, seed: int) -> torch.Tensor:
            raise RuntimeError(
                "No differentiable loss is available. "
                "You must implement build_var_loss_fn() using your VAR training forward "
                "(e.g., CE on next tokens / teacher-forcing)."
            )
        return _stub

    if mode == "model":
                             
        def _model_loss(label_B: torch.Tensor, seed: int) -> torch.Tensor:
                                                                                 
                                                    
            if hasattr(var, "forward_loss"):
                return var.forward_loss(label_B=label_B, g_seed=seed, cfg=getattr(args, "cfg", 1.5))
            if hasattr(var, "compute_loss"):
                return var.compute_loss(label_B=label_B, g_seed=seed, cfg=getattr(args, "cfg", 1.5))
            raise RuntimeError(
                "mode=model selected but var.forward_loss / var.compute_loss not found. "
                "Switch to --loss_mode trainer and bind to your training pipeline."
            )
        return _model_loss

    if mode == "trainer":
                                                                      
                                                                          
                                                                      
                                                        
                                                    
                                             
                                     
                                                                      
        import torch.nn as nn
        loss_fn_ce = nn.CrossEntropyLoss(reduction='mean')
        
        def _trainer_loss(label_B: torch.Tensor, seed: int) -> torch.Tensor:
\
\
\
\
\
\
\
\
\
\
\
\
\
\
               
            B = label_B.shape[0]
            device = label_B.device
            
                                                                    
            torch.manual_seed(seed)
            
                                                                             
                                                                                
                                                                         
            H, W = 256, 256                          
            dummy_images = torch.rand(B, 3, H, W, device=device)
            
                                                                          
                                                     
            gt_idx_Bl = vae.img_to_idxBl(dummy_images)
            gt_BL = torch.cat(gt_idx_Bl, dim=1)          
            x_BLCv_wo_first_l = vae.quantize.idxBl_to_var_input(gt_idx_Bl)                  
            
                                                              
            original_prog_si = var.prog_si
            var.prog_si = -1                  
            
                                                           
                                                                                                 
            logits_BLV = var(label_B, x_BLCv_wo_first_l)             
            
                                      
            var.prog_si = original_prog_si
            
                                      
                                                     
            V = logits_BLV.shape[-1]
            loss = loss_fn_ce(logits_BLV.view(-1, V), gt_BL.view(-1))
            
            return loss
        
        return _trainer_loss

    raise ValueError(f"Unknown loss mode: {mode}")


def compute_hidden_neuron_importance_taylor(
    model,
    layer_idx: int,
    calib_pairs: List[Tuple[torch.Tensor, int]],
    loss_fn: LossFn,
    device: torch.device,
) -> torch.Tensor:
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
       
    block = model.blocks[layer_idx]
    ffn = block.ffn
    fc1 = ffn.fc1
    fc2 = ffn.fc2

    H = fc1.out_features

                                                                  
                                                                           
                                                                  
                                          
    original_training = model.training
    original_requires_grad = {}
    for name, p in model.named_parameters():
        original_requires_grad[name] = p.requires_grad
    
                                                                          
                                                                       
    model.eval()
    
                                                     
    for p in model.parameters():
        p.requires_grad_(False)
    for p in fc1.parameters():
        p.requires_grad_(True)
    for p in fc2.parameters():
        p.requires_grad_(True)

    imp = torch.zeros(H, device=device, dtype=torch.float32)

    for (label_B, seed) in calib_pairs:
        model.zero_grad(set_to_none=True)
        loss = loss_fn(label_B, seed)
        if loss.dim() != 0:
            loss = loss.mean()
                                                                        
        loss.backward()                                                      

                                 
        if fc1.weight.grad is None or fc2.weight.grad is None:
            raise RuntimeError("Gradients not found. Ensure your loss_fn is differentiable and uses this FFN.")

        g1 = fc1.weight.grad.detach()
        w1 = fc1.weight.detach()
        imp_fc1 = (g1 * w1).abs().sum(dim=1)       

        imp_bias = 0.0
        if fc1.bias is not None and fc1.bias.grad is not None:
            imp_bias = (fc1.bias.grad.detach() * fc1.bias.detach()).abs()       

                                                      
        g2 = fc2.weight.grad.detach()
        w2 = fc2.weight.detach()
        imp_fc2 = (g2 * w2).abs().sum(dim=0)       

        imp += (imp_fc1 + imp_fc2 + imp_bias.float())

                                                                  
                                                              
                                                                  
    if original_training:
        model.train()
    else:
        model.eval()
    
                                           
    for name, p in model.named_parameters():
        p.requires_grad_(original_requires_grad[name])
    
                                                               
    return imp.detach().cpu()


def split_hidden_by_importance(
    importance: torch.Tensor,
    shared_hidden: int,
    n_experts: int,
) -> Tuple[np.ndarray, List[np.ndarray]]:
\
\
\
\
\
\
\
       
    imp = importance.numpy()
    order = np.argsort(-imp)              

    shared_idx = order[:shared_hidden]
    rest = order[shared_hidden:]

                
    remain = len(rest)
    assert remain % n_experts == 0, "Remaining hidden must be divisible by n_experts"
    expert_hidden = remain // n_experts
    expert_idx_list = []
    for i in range(n_experts):
        expert_idx_list.append(rest[i * expert_hidden:(i + 1) * expert_hidden])

    return shared_idx.astype(np.int64), [x.astype(np.int64) for x in expert_idx_list]


def init_router_from_expert_weights(
    router: D2MRouter,
    ffn_fc1_weight: torch.Tensor,
    expert_indices_list: List[np.ndarray],
    device: torch.device,
) -> None:
\
\
\
\
\
       
    W = ffn_fc1_weight.to(device).float()         
    rows = []
    for idx in expert_indices_list:
        w = W[idx]            
        w = F.normalize(w, p=2, dim=1)
        c = w.mean(dim=0, keepdim=True)
        c = F.normalize(c, p=2, dim=1)
        rows.append(c.squeeze(0))
    R = torch.stack(rows, dim=0)         
    with torch.no_grad():
        router.proj.weight.data = R.to(dtype=router.proj.weight.dtype).clone()
        if router.proj.bias is not None:
            router.proj.bias.zero_()


def build_var_d2m_from_ffn(
    ffn: nn.Module,
    shared_indices: np.ndarray,
    expert_indices_list: List[np.ndarray],
    n_experts: int,
    topk: int,
    hard_mode: bool,
    device: torch.device,
    router_bias: bool = False,
) -> nn.Module:
\
\
\
\
\
\
\
\
\
\
\
       
    fc1 = ffn.fc1
    fc2 = ffn.fc2
    C = fc1.in_features
    H = fc1.out_features

    shared_hidden = len(shared_indices)
    remain = H - shared_hidden
    assert len(expert_indices_list) == n_experts
    assert remain % n_experts == 0
    expert_hidden = remain // n_experts

    drop_rate = 0.0
    if hasattr(ffn, "drop") and isinstance(ffn.drop, nn.Dropout):
        drop_rate = float(ffn.drop.p)

    moe = VARD2MFFN(
        in_features=C,
        shared_hidden=shared_hidden,
        expert_hidden=expert_hidden,
        n_experts=n_experts,
        topk=topk,
        drop=drop_rate,
        hard_mode=hard_mode,
        router_bias=router_bias,
    ).to(device)

                           
    w1 = fc1.weight.data.to(device)
    b1 = fc1.bias.data.to(device) if fc1.bias is not None else None
    w2 = fc2.weight.data.to(device)
    b2 = fc2.bias.data.to(device) if fc2.bias is not None else None

    sidx = torch.tensor(shared_indices, device=device, dtype=torch.long)
    with torch.no_grad():
        moe.shared.fc1.weight.data.copy_(w1.index_select(0, sidx))
        if b1 is not None:
            moe.shared.fc1.bias.data.copy_(b1.index_select(0, sidx))
        moe.shared.fc2.weight.data.copy_(w2.index_select(1, sidx))

                            
    for i, idx_np in enumerate(expert_indices_list):
        eidx = torch.tensor(idx_np, device=device, dtype=torch.long)
        with torch.no_grad():
            moe.experts[i].fc1.weight.data.copy_(w1.index_select(0, eidx))
            if b1 is not None:
                moe.experts[i].fc1.bias.data.copy_(b1.index_select(0, eidx))
            moe.experts[i].fc2.weight.data.copy_(w2.index_select(1, eidx))

                           
    if b2 is not None:
        with torch.no_grad():
                                                                              
            moe.out_bias.copy_(b2.detach().float().to(device=moe.out_bias.device))

    return moe
