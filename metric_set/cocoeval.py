import numpy as np
import datetime
import time
from collections import defaultdict
from pycocotools import mask as maskUtils
from pycocotools.cocoeval import COCOeval
import copy


class COCOeval(COCOeval):

    def accumulate_conf_mat(self, p=None):
        """
        Accumulate per image evaluation results and store the result in self.eval
        :param p: input params for evaluation
        :return: None
        """
        print('Accumulating evaluation results...')
        tic = time.time()
        if not self.evalImgs:
            print('Please run evaluate() first')
        # allows input customized parameters
        if p is None:
            p = self.params
        p.catIds = p.catIds if p.useCats == 1 else [-1]
        T = len(p.iouThrs)
        K = len(p.catIds) if p.useCats else 1
        A = len(p.areaRng)
        M = len(p.maxDets)
        coff_mat = np.zeros((T, K, A, M, 3))

        # create dictionary for future indexing
        _pe = self._paramsEval
        catIds = _pe.catIds if _pe.useCats else [-1]
        setK = set(catIds)
        setA = set(map(tuple, _pe.areaRng))
        setM = set(_pe.maxDets)
        setI = set(_pe.imgIds)
        # get inds to evaluate
        k_list = [n for n, k in enumerate(p.catIds) if k in setK]
        m_list = [m for n, m in enumerate(p.maxDets) if m in setM]
        a_list = [n for n, a in enumerate(map(lambda x: tuple(x), p.areaRng)) if a in setA]
        i_list = [n for n, i in enumerate(p.imgIds) if i in setI]
        I0 = len(_pe.imgIds)
        A0 = len(_pe.areaRng)
        # retrieve E at each category, area range, and max number of detections
        for k, k0 in enumerate(k_list):
            Nk = k0 * A0 * I0
            for a, a0 in enumerate(a_list):
                Na = a0 * I0
                for m, maxDet in enumerate(m_list):
                    E = [self.evalImgs[Nk + Na + i] for i in i_list]
                    E = [e for e in E if e is not None]
                    if len(E) == 0:
                        continue

                    # gtm[T, G],gt匹配的dt的id. dtm[T, D],dt匹配的gt的id.
                    # gtIg[G],gt的_ignore标签,dtIg[T, D],dt匹配的gt的_ignore标签.

                    # 在目标检测问题中:
                    # tp为真正例,整个数据集中被识别出来的Ground Truth
                    # fn为假负例,整个数据集中没有被识别出来的Ground Truth
                    # fp为假正例,整个数据集中错误预测出来的Preds
                    # tn不存在是0

                    dtm = np.concatenate([e['dtMatches'][:, 0:maxDet] for e in E], axis=1)
                    dtIg = np.concatenate([e['dtIgnore'][:, 0:maxDet] for e in E], axis=1)
                    gtIg = np.concatenate([e['gtIgnore'] for e in E])
                    npig = np.count_nonzero(gtIg == 0)
                    if npig == 0:
                        continue
                    tps = np.logical_and(dtm, np.logical_not(dtIg))
                    fps = np.logical_and(np.logical_not(dtm), np.logical_not(dtIg))

                    tp_sum = np.sum(tps, axis=1).astype(dtype=float)
                    fp_sum = np.sum(fps, axis=1).astype(dtype=float)
                    fn_sum = npig - tp_sum

                    coff_mat[:, k, a, m, 0] = tp_sum
                    coff_mat[:, k, a, m, 1] = fp_sum
                    coff_mat[:, k, a, m, 2] = fn_sum


        toc = time.time()
        print('DONE (t={:0.2f}s).'.format(toc - tic))
        return coff_mat


class Params:


    """
    Params for coco evaluation api
    """

    imgIds:list = []
    catIds:list = []
    # np.arange causes trouble.  the data point on arange is slightly larger than the true value
    iouThrs:np.ndarray
    recThrs:np.ndarray
    maxDets:list = [1, 10, 100]
    areaRng:list = [[0 ** 2, 1e5 ** 2], [0 ** 2, 32 ** 2], [32 ** 2, 96 ** 2], [96 ** 2, 1e5 ** 2]]
    areaRngLbl:list = ['all', 'small', 'medium', 'large']
    useCats:int = 1
    kpt_oks_sigmas:np.ndarray

    def setDetParams(self):
        self.imgIds = []
        self.catIds = []
        # np.arange causes trouble.  the data point on arange is slightly larger than the true value
        self.iouThrs = np.linspace(.5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
        self.recThrs = np.linspace(.0, 1.00, int(np.round((1.00 - .0) / .01)) + 1, endpoint=True)
        self.maxDets = [1, 10, 100]
        self.areaRng = [[0 ** 2, 1e5 ** 2], [0 ** 2, 32 ** 2], [32 ** 2, 96 ** 2], [96 ** 2, 1e5 ** 2]]
        self.areaRngLbl = ['all', 'small', 'medium', 'large']
        self.useCats = 1

    def setKpParams(self):
        self.imgIds = []
        self.catIds = []
        # np.arange causes trouble.  the data point on arange is slightly larger than the true value
        self.iouThrs = np.linspace(.5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
        self.recThrs = np.linspace(.0, 1.00, int(np.round((1.00 - .0) / .01)) + 1, endpoint=True)
        self.maxDets = [20]
        self.areaRng = [[0 ** 2, 1e5 ** 2], [32 ** 2, 96 ** 2], [96 ** 2, 1e5 ** 2]]
        self.areaRngLbl = ['all', 'medium', 'large']
        self.useCats = 1
        self.kpt_oks_sigmas = np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62,.62, 1.07, 1.07, .87, .87, .89, .89])/10.0

    def __init__(self, iouType='segm'):
        if iouType == 'segm' or iouType == 'bbox':
            self.setDetParams()
        elif iouType == 'keypoints':
            self.setKpParams()
        else:
            raise Exception('iouType not supported')
        self.iouType = iouType
        # useSegm is deprecated
        self.useSegm = None
