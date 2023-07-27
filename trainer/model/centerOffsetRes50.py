import torch
import torch.nn

from models.centerNetOffset import CenterNetResidual, CenterNetLoss, centerNetEvaluation, CenterNetHourglass
from models.losses.focal import focalLoss
from models.losses.regression import L1LossMask
from evaluations.detection import averagePrecisionPlots, averagePrecisionAll

model = CenterNetResidual
# model = CenterNetHourglass
loss = CenterNetLoss(0.1, 0.1, focal = focalLoss, regression = L1LossMask)

modelParams = {'numLayers': 50,
               'dims': [64, 64, 128, 256, 512, 256, 256, 256] }
# modelParams = {}
evaluation = centerNetEvaluation

def expression(batches):
    eval = { 'mIoU': 0,
             'mIoUC': 0,
             'mIoUwoO': 0,
             'mIouO': 0,
             'ap30': 0,
             'ap50': 0,
             'ap70': 0,
             'ap90': 0,
             'orthogonity': 0,
             'majMAE': 0,
             'minMAE': 0,
             'radMAE': 0,
             'avgScore': 0
           }
    
    objNum = 0
    ious = torch.tensor([])
    scores = torch.tensor([])
    orthos = torch.tensor([])

    ioucenters = torch.tensor([])
    iouoffsets = torch.tensor([])
    iouoffsetwos = torch.tensor([])

    aemajs = torch.tensor([])
    aemins = torch.tensor([])
    aerads = torch.tensor([])
    
    for batch in batches:
        objNum += torch.sum(torch.tensor(batch['objs'])).item()
        iou, score = batch['iouscore']
        ortho = batch['ortho']
        ioucenter = batch['ioucenter']
        iouoffset = batch['iouoffset']
        iouoffsetwo = batch['iouoffsetwo']
        aemaj, aemin, aerad = batch['maes']
        
        ious = torch.cat([ious, iou.cpu()], 0)
        ioucenters = torch.cat([ioucenters, ioucenter.cpu()], 0)
        iouoffsets = torch.cat([iouoffsets, iouoffset.cpu()], 0)
        iouoffsetwos = torch.cat([iouoffsetwos, iouoffsetwo.cpu()], 0)
        scores = torch.cat([scores, score.cpu()], 0)
        orthos = torch.cat([orthos, ortho.cpu()], 0)
        aemajs = torch.cat([aemajs, aemaj.cpu()], 0)
        aemins = torch.cat([aemins, aemin.cpu()], 0)
        aerads = torch.cat([aerads, aerad.cpu()], 0)
    
    eval['mIoU'] = torch.mean(ious if len(ious) > 0 else torch.zeros(1))
    eval['mIoUC'] = torch.mean(ioucenters if len(ioucenters) > 0 else torch.zeros(1))
    eval['mIoUO'] = torch.mean(iouoffsets if len(iouoffsets) > 0 else torch.zeros(1))
    eval['mIoUwoO'] = torch.mean(iouoffsetwos if len(iouoffsetwos) > 0 else torch.zeros(1))

    orthoNonNan = torch.isnan(orthos) == False
    eval['orthogonity'] = torch.mean(orthos[orthoNonNan] if len(orthos[orthoNonNan]) > 0 else torch.zeros(1))
    eval['avgScore'] = torch.mean(scores if len(scores) > 0 else torch.zeros(1))

    eval['majMAE'] = torch.mean(aemajs if len(aemajs) > 0 else torch.zeros(1))
    eval['minMAE'] = torch.mean(aemins if len(aemins) > 0 else torch.zeros(1))
    eval['radMAE'] = torch.mean(aerads if len(aerads) > 0 else torch.zeros(1))

    objNum = max(objNum, len(ious))
    plots70 = averagePrecisionPlots(ious, scores, objNum, 0.7)
    plots90 = averagePrecisionPlots(ious, scores, objNum, 0.9)
    plots30 = averagePrecisionPlots(ious, scores, objNum, 0.3)
    plots50 = averagePrecisionPlots(ious, scores, objNum, 0.5)
    eval['ap70'] = averagePrecisionAll(plots70)
    eval['ap90'] = averagePrecisionAll(plots90)
    eval['ap30'] = averagePrecisionAll(plots30)
    eval['ap50'] = averagePrecisionAll(plots50)

    evalr = "[mIoU] {}    [mIoUC] {}    [mIoUwoO] {}    [mIoUO] {}    [AP30] {}    [AP50] {}    [AP70] {}    [AP90] {}    [Orth] {}    [majMAE] {}    [minMAE] {}    [radMAE] {}    [avgS] {}".format(
                
                format(eval['mIoU'] * 100, '-10.8f'),
                format(eval['mIoUC'] * 100, '-10.8f'),
                format(eval['mIoUwoO'] * 100, '-10.8f'),
                format(eval['mIoUO'] * 100, '-10.8f'),
                format(eval['ap30'] * 100, '-5.2f'),
                format(eval['ap50'] * 100, '-5.2f'),
                format(eval['ap70'] * 100, '-5.2f'),
                format(eval['ap90'] * 100, '-5.2f'),
                format(eval['orthogonity'], '-8.6f'),
                format(eval['majMAE'], '-8.6f'),
                format(eval['minMAE'], '-8.6f'),
                format(eval['radMAE'], '-8.6f'),
                format(eval['avgScore'], '-6.4f')
            )
    
    return evalr
