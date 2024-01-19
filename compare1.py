                    loss_sc        = (torch.sigmoid(pred1)-torch.sigmoid(pred2)).abs()
                    loss_sc        = loss_sc[mask[:,0:1]==1].mean()
                    ## M2B transformation
                    pred           = torch.cat([pred1, pred2], dim=0)
                    mask           = torch.cat([mask, mask], dim=0)
                    predW, predH   = pred.max(dim=2, keepdim=True)[0], pred.max(dim=3, keepdim=True)[0]
                    pred           = torch.minimum(predW, predH)
                    pred, mask     = pred[:,0], mask[:,0]
                    ## loss_ce + loss_dice 
                    loss_ce        = F.binary_cross_entropy_with_logits(pred, mask)
                    pred           = torch.sigmoid(pred)
                    inter          = (pred*mask).sum(dim=(1,2))
                    union          = (pred+mask).sum(dim=(1,2))
                    loss_dice      = 1-(2*inter/(union+1)).mean()