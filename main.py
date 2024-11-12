from metric_set import MeanAveragePrecision

from torch import tensor
from pprint import pprint
if __name__ == '__main__':
    preds = [dict(
        boxes=tensor([[258.0, 41.0, 606.0, 285.0]]),
        scores=tensor([0.536]),
        labels=tensor([0]),
    )]
    target = [dict(
        boxes=tensor([[214.0, 41.0, 562.0, 285.0]]),
        labels=tensor([0]),
    )]
    # initialize the metric
    metric = MeanAveragePrecision()
    metric.update(preds, target)
    map_dict = metric.compute()
    # specify the expected key
    metric = MeanAveragePrecision(num_classes=1, prefix='val_', keyword='map')
    metric.update(preds, target)
    map_tensor = metric.compute()

    # print the result
    pprint(map_dict)
    pprint(map_tensor)
