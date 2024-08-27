import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
def initialize_reference_pool(net_model, train_loader_memory, opt, feat_dim, device):
    with torch.no_grad():
        Pool_ag = torch.zeros((opt["model"]["pool_size"], feat_dim), dtype=torch.float16).cuda() # 1024
        Pool_sp = torch.zeros((opt["model"]["pool_size"], opt["model"]["dim"]), dtype=torch.float16).cuda()
        Pool_iter = iter(train_loader_memory)

        for _iter in tqdm(range(len(train_loader_memory))):
            data = next(Pool_iter)
            img: torch.Tensor = data['image'].unsqueeze(0).to(device, non_blocking=True)

            if _iter >= opt["model"]["pool_size"]:
                break
            img = img.cuda()
            with torch.cuda.amp.autocast(enabled=True):
                model_output = net_model(img)

                modeloutput_f = model_output[0].clone().detach()
                modeloutput_f = modeloutput_f.view(modeloutput_f.size(0), modeloutput_f.size(1), -1)

                modeloutput_s_pr = model_output[2].clone().detach()
                modeloutput_s_pr = modeloutput_s_pr.view(modeloutput_s_pr.size(0), modeloutput_s_pr.size(1), -1)

            for _iter2 in range(modeloutput_f.size(0)):
                randidx = np.random.randint(0, model_output[0].size(-1) * model_output[0].size(-2))
                Pool_ag[_iter + _iter2] = modeloutput_f[_iter2][:, randidx]

            for _iter2 in range(modeloutput_s_pr.size(0)):
                randidx = np.random.randint(0, model_output[2].size(-1) * model_output[2].size(-2))
                Pool_sp[_iter + _iter2] = modeloutput_s_pr[_iter2][:, randidx]

        Pool_ag = F.normalize(Pool_ag, dim=1)
        Pool_sp = F.normalize(Pool_sp, dim=1)
    return Pool_ag, Pool_sp

def renew_reference_pool(net_model, train_loader_memory, opt, device):
    with torch.no_grad():
        Pool_sp = torch.zeros((opt["model"]["pool_size"], opt["model"]["dim"]), dtype=torch.float16).cuda()
        pool_idxs = torch.randperm(len(train_loader_memory))[:opt['model']['pool_size']]
        for _iter, p_idx in tqdm(enumerate(pool_idxs)):
            if _iter >= opt["model"]["pool_size"]:
                break
            img_net: torch.Tensor = train_loader_memory[p_idx]['image'].unsqueeze(0).to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=True):
                model_output = net_model(img_net)

                modeloutput_s_pr = model_output[2].clone().detach()
                modeloutput_s_pr = modeloutput_s_pr.view(modeloutput_s_pr.size(0), modeloutput_s_pr.size(1), -1)

            for _iter2 in range(modeloutput_s_pr.size(0)):
                randidx = np.random.randint(0, model_output[2].size(-1) * model_output[2].size(-2))
                Pool_sp[_iter + _iter2] = modeloutput_s_pr[_iter2][:, randidx]

            # print("Filling Pool Memory [{} / {}]".format(
            #     (_iter + 1), opt["model"]["pool_size"]))

        Pool_sp = F.normalize(Pool_sp, dim=1)

    return Pool_sp