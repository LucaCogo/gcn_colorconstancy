import torch
import math

class AngularError:

    def __init__(self):
        self.name = "AngularError"

    def __call__(self, pred, gt):
        return self.compute(pred, gt)

    def compute(self, pred, gt, safe_v: float = 0.999999):
        norm_est = torch.linalg.norm(pred, dim=1)    
        norm_gt = torch.linalg.norm(gt, dim=1)
        dot = torch.sum(pred * gt, dim=1) / (norm_gt * norm_est) 
        dot = torch.clamp(dot, -safe_v, safe_v)

        rad = torch.mean(torch.acos(dot))
        deg = rad * 180 / math.pi


        return rad, deg