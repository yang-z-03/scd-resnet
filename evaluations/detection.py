
import sys
import torch
import torch.cuda
from configuration import defaultConfig

# returns a list of intersection-over-union for corresponding ground-truth detections.
# where detections are (BATCH, K, 4) tensors, and groundTruth are (BATCH, MAXTAGLEN, 4) tensors.
# each bounding boxes (in 4 dimensions) represents the top-left corner and bottom-right corners
# (tlX, tlY, brX, brY).

def IoU(detections:torch.Tensor, groundTruth:torch.Tensor, validMask = None):

    N, K, _ = detections.shape
    _, L, _ = groundTruth.shape
    tlX1, tlY1, brX1, brY1 = detections[:,:,0], detections[:,:,1], detections[:,:,2], detections[:,:,3]
    tlX2, tlY2, brX2, brY2 = groundTruth[:,:,0], groundTruth[:,:,1], groundTruth[:,:,2], groundTruth[:,:,3]

    tlX1 = tlX1.view(N, K, 1).expand(N, K, L)
    tlY1 = tlY1.view(N, K, 1).expand(N, K, L)
    brX1 = brX1.view(N, K, 1).expand(N, K, L)
    brY1 = brY1.view(N, K, 1).expand(N, K, L)
    detMask = validMask.view(N, K, 1).expand(N, K, L)

    detArea = (brX1 - tlX1) * (brY1 - tlY1)

    tlX2 = tlX2.view(N, 1, L).expand(N, K, L)
    tlY2 = tlY2.view(N, 1, L).expand(N, K, L)
    brX2 = brX2.view(N, 1, L).expand(N, K, L)
    brY2 = brY2.view(N, 1, L).expand(N, K, L)

    gtArea = (brX2 - tlX2) * (brY2 - tlY2)

    # the intersection coordinates, in the form of (N, K, L). some of the elements (for those
    # who doesn't intersects) has negative values.
    tlXI = torch.max(tlX1, tlX2)
    tlYI = torch.max(tlY1, tlY2)
    brXI = torch.min(brX1, brX2)
    brYI = torch.min(brY1, brY2)

    deltaX = brXI - tlXI
    deltaY = brYI - tlYI
    mask = (deltaX > 1e-5) * (deltaY > 1e-5) * (gtArea > 1e-5) * detMask

    intersection = torch.masked_select(deltaX * deltaY, mask)
    detArea = torch.masked_select(detArea, mask)
    gtArea = torch.masked_select(gtArea, mask)

    return intersection / (detArea + gtArea - intersection)

# variation of IoU function.
# select the corresponding boxes according to IoU, and measures their orthogonity of the major axis.
# majDetections in the form of [BATCH, K, 3](majX, majY, majL) and ground-truth of [BATCH, MAXTAGLEN, 3].

def Orthogonity(posDetections:torch.Tensor, posGroundTruth:torch.Tensor, majDetections:torch.Tensor, majGroundTruth:torch.Tensor, validMask = None):

    N, K, _ = posDetections.shape
    _, L, _ = posGroundTruth.shape
    tlX1, tlY1, brX1, brY1 = posDetections[:,:,0], posDetections[:,:,1], posDetections[:,:,2], posDetections[:,:,3]
    tlX2, tlY2, brX2, brY2 = posGroundTruth[:,:,0], posGroundTruth[:,:,1], posGroundTruth[:,:,2], posGroundTruth[:,:,3]
    majXd, majYd, majLd = majDetections[:,:,0], majDetections[:,:,1], majDetections[:,:,2]
    majXg, majYg, majLg = majGroundTruth[:,:,0], majGroundTruth[:,:,1], majGroundTruth[:,:,2]

    tlX1 = tlX1.view(N, K, 1).expand(N, K, L)
    tlY1 = tlY1.view(N, K, 1).expand(N, K, L)
    brX1 = brX1.view(N, K, 1).expand(N, K, L)
    brY1 = brY1.view(N, K, 1).expand(N, K, L)

    majXd = majXd.view(N, K, 1).expand(N, K, L)
    majYd = majYd.view(N, K, 1).expand(N, K, L)
    majLd = majLd.view(N, K, 1).expand(N, K, L)
    detMask = validMask.view(N, K, 1).expand(N, K, L)

    detArea = (brX1 - tlX1) * (brY1 - tlY1)

    tlX2 = tlX2.view(N, 1, L).expand(N, K, L)
    tlY2 = tlY2.view(N, 1, L).expand(N, K, L)
    brX2 = brX2.view(N, 1, L).expand(N, K, L)
    brY2 = brY2.view(N, 1, L).expand(N, K, L)

    majXg = majXg.view(N, 1, L).expand(N, K, L)
    majYg = majYg.view(N, 1, L).expand(N, K, L)
    majLg = majLg.view(N, 1, L).expand(N, K, L)

    gtArea = (brX2 - tlX2) * (brY2 - tlY2)
    orthoganityCos = ((majXd * majXg) + (majYd * majYg)) / (majLd * majLg)
    orthoganitySin = torch.sqrt(1 - orthoganityCos ** 2)

    # the intersection coordinates, in the form of (N, K, L). some of the elements (for those
    # who doesn't intersects) has negative values.
    tlXI = torch.max(tlX1, tlX2)
    tlYI = torch.max(tlY1, tlY2)
    brXI = torch.min(brX1, brX2)
    brYI = torch.min(brY1, brY2)

    deltaX = brXI - tlXI
    deltaY = brYI - tlYI
    mask = (deltaX > 1e-5) * (deltaY > 1e-5) * (gtArea > 1e-5) * detMask * (majLg > 1e-5)

    intersection = torch.masked_select(deltaX * deltaY, mask)
    detArea = torch.masked_select(detArea, mask)
    gtArea = torch.masked_select(gtArea, mask)
    orthogonity = torch.masked_select(orthoganitySin, mask)

    return orthogonity

# regr in the form of [BATCH, K, 3](majL, minL, radius)

def MAE(posDetections:torch.Tensor, posGroundTruth:torch.Tensor, regr:torch.Tensor, regrGroundTruth:torch.Tensor, validMask = None):

    N, K, _ = posDetections.shape
    _, L, _ = posGroundTruth.shape
    tlX1, tlY1, brX1, brY1 = posDetections[:,:,0], posDetections[:,:,1], posDetections[:,:,2], posDetections[:,:,3]
    tlX2, tlY2, brX2, brY2 = posGroundTruth[:,:,0], posGroundTruth[:,:,1], posGroundTruth[:,:,2], posGroundTruth[:,:,3]
    majLd, minLd, radiusd = regr[:,:,0], regr[:,:,1], regr[:,:,2]
    majLg, minLg, radiusg = regrGroundTruth[:,:,0], regrGroundTruth[:,:,1], regrGroundTruth[:,:,2]

    tlX1 = tlX1.view(N, K, 1).expand(N, K, L)
    tlY1 = tlY1.view(N, K, 1).expand(N, K, L)
    brX1 = brX1.view(N, K, 1).expand(N, K, L)
    brY1 = brY1.view(N, K, 1).expand(N, K, L)

    radiusd = radiusd.view(N, K, 1).expand(N, K, L)
    minLd = minLd.view(N, K, 1).expand(N, K, L)
    majLd = majLd.view(N, K, 1).expand(N, K, L)
    detMask = validMask.view(N, K, 1).expand(N, K, L)

    detArea = (brX1 - tlX1) * (brY1 - tlY1)

    tlX2 = tlX2.view(N, 1, L).expand(N, K, L)
    tlY2 = tlY2.view(N, 1, L).expand(N, K, L)
    brX2 = brX2.view(N, 1, L).expand(N, K, L)
    brY2 = brY2.view(N, 1, L).expand(N, K, L)

    radiusg = radiusg.view(N, 1, L).expand(N, K, L)
    minLg = minLg.view(N, 1, L).expand(N, K, L)
    majLg = majLg.view(N, 1, L).expand(N, K, L)

    gtArea = (brX2 - tlX2) * (brY2 - tlY2)
    aeMinL = torch.abs(minLd - minLg)
    aeMajL = torch.abs(majLd - majLg)
    aeRadius = torch.abs(radiusd - radiusg)

    # the intersection coordinates, in the form of (N, K, L). some of the elements (for those
    # who doesn't intersects) has negative values.
    tlXI = torch.max(tlX1, tlX2)
    tlYI = torch.max(tlY1, tlY2)
    brXI = torch.min(brX1, brX2)
    brYI = torch.min(brY1, brY2)

    deltaX = brXI - tlXI
    deltaY = brYI - tlYI
    mask = (deltaX > 1e-5) * (deltaY > 1e-5) * (gtArea > 1e-5) * detMask * (majLg > 1e-5)

    intersection = torch.masked_select(deltaX * deltaY, mask)
    detArea = torch.masked_select(detArea, mask)
    gtArea = torch.masked_select(gtArea, mask)

    return [torch.masked_select(aeMajL, mask),
            torch.masked_select(aeMinL, mask),
            torch.masked_select(aeRadius, mask)]

# confidence scores are ctScores in (N, K).
def IoUConfidence(detections:torch.Tensor, groundTruth:torch.Tensor, confidenceScore:torch.Tensor, validMask = None):

    N, K, _ = detections.shape
    _, L, _ = groundTruth.shape
    tlX1, tlY1, brX1, brY1 = detections[:,:,0], detections[:,:,1], detections[:,:,2], detections[:,:,3]
    tlX2, tlY2, brX2, brY2 = groundTruth[:,:,0], groundTruth[:,:,1], groundTruth[:,:,2], groundTruth[:,:,3]

    tlX1 = tlX1.view(N, K, 1).expand(N, K, L)
    tlY1 = tlY1.view(N, K, 1).expand(N, K, L)
    brX1 = brX1.view(N, K, 1).expand(N, K, L)
    brY1 = brY1.view(N, K, 1).expand(N, K, L)
    detMask = validMask.view(N, K, 1).expand(N, K, L)
    scores = confidenceScore.view(N, K, 1).expand(N, K, L)

    detArea = (brX1 - tlX1) * (brY1 - tlY1)

    tlX2 = tlX2.view(N, 1, L).expand(N, K, L)
    tlY2 = tlY2.view(N, 1, L).expand(N, K, L)
    brX2 = brX2.view(N, 1, L).expand(N, K, L)
    brY2 = brY2.view(N, 1, L).expand(N, K, L)

    gtArea = (brX2 - tlX2) * (brY2 - tlY2)

    # the intersection coordinates, in the form of (N, K, L). some of the elements (for those
    # who doesn't intersects) has negative values.
    tlXI = torch.max(tlX1, tlX2)
    tlYI = torch.max(tlY1, tlY2)
    brXI = torch.min(brX1, brX2)
    brYI = torch.min(brY1, brY2)

    deltaX = brXI - tlXI
    deltaY = brYI - tlYI
    # mask = (deltaX > 1e-5) * (deltaY > 1e-5) * (gtArea > 1e-5) * detMask
    mask = (deltaX > 1e-5) * (deltaY > 1e-5) * (gtArea > 1e-5) * detMask

    intersection = torch.masked_select(deltaX * deltaY, mask)
    detArea = torch.masked_select(detArea, mask)
    gtArea = torch.masked_select(gtArea, mask)
    selScore = torch.masked_select(scores, mask)

    return [intersection / (detArea + gtArea - intersection), selScore]

# ground-truth heatmap in form of array 1d.
def averagePrecisionPlots(ious, scores, objNum, threshold):
    _, sortIndices = torch.sort(scores)
    sortIndices = sortIndices.flip(0)

    plots = []
    accumulatedTruth = 0
    accumulatedFalse = 0
    recall = 0
    total = objNum

    for i in range(sortIndices.shape[0]):
        currId = sortIndices[i]
        currScore = scores[currId]
        currIoU = ious[currId]

        if currIoU < threshold:
            # false prediction
            accumulatedFalse += 1
        else:
            accumulatedTruth += 1
            recall += 1

        plots += [[recall / total, accumulatedTruth / (accumulatedTruth + accumulatedFalse)]]

    return plots

def averagePrecisionAll(apPlots) -> float:
    length = len(apPlots)
    x1 = 1
    x2 = 1
    y = 0

    ap = 0
    for i in range(length):
        reverseOrder = length - 1 - i
        recall, precision = apPlots[reverseOrder]

        if precision > y :
            ap += (x2 - x1) * y
            x2 = recall
            x1 = recall
            y = precision
        
        else:
            x1 = recall
    
    ap += x2 * y

    return ap

def apAll (detections:torch.Tensor, groundTruth:torch.Tensor, confidenceScore:torch.Tensor,
           objNum, threshold, validMask = None) -> float:
    iou, score = IoUConfidence(detections, groundTruth, confidenceScore, validMask)
    plots = averagePrecisionPlots(iou, score, objNum, threshold)
    return averagePrecisionAll(plots)

def apPlots (detections:torch.Tensor, groundTruth:torch.Tensor, confidenceScore:torch.Tensor,
             objNum, threshold, validMask = None) -> float:
    iou, score = IoUConfidence(detections, groundTruth, confidenceScore, validMask)
    plots = averagePrecisionPlots(iou, score, objNum, threshold)
    return plots
