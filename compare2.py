class PredsmIoU_DynamicQueries(Metric):
    """
    Subclasses Metric. Computes mean Intersection over Union (mIoU) given ground-truth and predictions.
    .update() can be called repeatedly to add data from multiple validation loops.
    """

    def __init__(self,
                 num_gt_classes: int):
        """
        :param num_gt_classes: The number of gt classes.
        """
        super().__init__(dist_sync_on_step=False, compute_on_step=False)
        self.num_gt_classes = num_gt_classes
        self.add_state("iou", [])
        self.add_state("iou_excludeFirst", [])
        self.n_jobs = -1

    def update(self, gt: torch.Tensor, pred: torch.Tensor, many_to_one=True, precision_based=True, linear_probe=False):
        pred = pred.cpu().numpy().astype(int)
        gt = gt.cpu().numpy().astype(int)
        self.num_pred_classes = len(np.unique(pred))
        iou_all, iou_excludeFirst = self.compute_miou(gt, pred, self.num_pred_classes, len(np.unique(gt)),
                                            many_to_one=many_to_one, precision_based=precision_based, linear_probe=linear_probe)
        self.iou.append(iou_all)
        self.iou_excludeFirst.append(iou_excludeFirst)

    def compute(self):
        """
        Compute mIoU
        """
        mIoU = np.mean(self.iou)
        mIoU_excludeFirst = np.mean(self.iou_excludeFirst)
        print('---mIoU computed---', mIoU)
        print('---mIoU exclude first---', mIoU_excludeFirst)
        return mIoU

    def compute_miou(self, gt: np.ndarray, pred: np.ndarray, num_pred: int, num_gt: int,
                     many_to_one=False, precision_based=False, linear_probe=False):
        """
        Compute mIoU with optional hungarian matching or many-to-one matching (extracts information from labels).
        :param gt: numpy array with all flattened ground-truth class assignments per pixel
        :param pred: numpy array with all flattened class assignment predictions per pixel
        :param num_pred: number of predicted classes
        :param num_gt: number of ground truth classes
        :param many_to_one: Compute a many-to-one mapping of predicted classes to ground truth instead of hungarian
        matching.
        :param precision_based: Use precision as matching criteria instead of IoU for assigning predicted class to
        ground truth class.
        :param linear_probe: Skip hungarian / many-to-one matching. Used for evaluating predictions of fine-tuned heads.
        :return: mIoU over all classes, true positives per class, false negatives per class, false positives per class,
        reordered predictions matching gt
        """
        assert pred.shape == gt.shape
        # print(f"unique semantic class = {np.unique(gt)}")
        gt_class = np.unique(gt).tolist()
        tp = [0] * num_gt
        fp = [0] * num_gt
        fn = [0] * num_gt
        iou = [0] * num_gt # 13个类别

        if linear_probe:
            reordered_preds = pred
        else:
            if many_to_one:
                match = self._original_match(num_pred, num_gt, pred, gt, precision_based=precision_based) # gt->list[pred_cls]
                # remap predictions
                reordered_preds = np.zeros(len(pred))
                for target_i, matched_preds in match.items():
                    for pred_i in matched_preds:
                        reordered_preds[pred == int(pred_i)] = int(target_i)
            else:
                match = self._hungarian_match(num_pred, num_gt, pred, gt)
                # remap predictions
                reordered_preds = np.zeros(len(pred))
                for target_i, pred_i in zip(*match):
                    reordered_preds[pred == int(pred_i)] = int(target_i)
                # merge all unmatched predictions to background
                # 1. gt>5, 但是pred因为softmax+max没有类2
                # 2. gt<5, matched到的Pred没有2
                for unmatched_pred in np.delete(np.arange(num_pred), np.array(match[1])):
                    reordered_preds[pred == int(unmatched_pred)] = 0

        # tp, fp, and fn evaluation
        for i_part in range(0, num_gt):
            tmp_all_gt = (gt == gt_class[i_part])
            tmp_pred = (reordered_preds == gt_class[i_part])
            tp[i_part] += np.sum(tmp_all_gt & tmp_pred)
            fp[i_part] += np.sum(~tmp_all_gt & tmp_pred)
            fn[i_part] += np.sum(tmp_all_gt & ~tmp_pred)

        # Calculate IoU per class
        for i_part in range(0, num_gt):
            iou[i_part] = float(tp[i_part]) / max(float(tp[i_part] + fp[i_part] + fn[i_part]), 1e-8)

        print('\tiou = ', iou, np.mean(iou[1:]))
        if len(iou) > 1:
            return np.mean(iou), np.mean(iou[1:])
        else:
            # return np.mean(iou), tp, fp, fn, reordered_preds.astype(int).tolist()
            return np.mean(iou), np.mean(iou)

    @staticmethod
    def get_score(flat_preds: np.ndarray, flat_targets: np.ndarray, c1: int, c2: int, precision_based: bool = False) \
            -> float:
        """
        Calculates IoU given gt class c1 and prediction class c2.
        :param flat_preds: flattened predictions
        :param flat_targets: flattened gt
        :param c1: ground truth class to match
        :param c2: predicted class to match
        :param precision_based: flag to calculate precision instead of IoU.
        :return: The score if gt-c1 was matched to predicted c2.
        """
        tmp_all_gt = (flat_targets == c1)
        tmp_pred = (flat_preds == c2)
        tp = np.sum(tmp_all_gt & tmp_pred)
        fp = np.sum(~tmp_all_gt & tmp_pred)
        if not precision_based:
            fn = np.sum(tmp_all_gt & ~tmp_pred)
            jac = float(tp) / max(float(tp + fp + fn), 1e-8)
            return jac
        else:
            prec = float(tp) / max(float(tp + fp), 1e-8)
            # print('\tgt, pred = ', c1, c2, ' | precision=', prec)
            return prec

    def compute_score_matrix(self, num_pred: int, num_gt: int, pred: np.ndarray, gt: np.ndarray,
                             precision_based: bool = False) -> np.ndarray:
        """
        Compute score matrix. Each element i, j of matrix is the score if i was matched j. Computation is parallelized
        over self.n_jobs.
        :param num_pred: number of predicted classes
        :param num_gt: number of ground-truth classes
        :param pred: flattened predictions
        :param gt: flattened gt
        :param precision_based: flag to calculate precision instead of IoU.
        :return: num_pred x num_gt matrix with A[i, j] being the score if ground-truth class i was matched to
        predicted class j.
        """
        # print("Parallelizing iou computation")
        # start = time.time()
        score_mat = []
        for c2 in range(num_pred):
            for c1 in np.unique(gt):
                score_mat.append(self.get_score(pred, gt, c1, c2, precision_based=precision_based))
                
        # score_mat = Parallel(n_jobs=self.n_jobs)(delayed(self.get_score)(pred, gt, c1, c2, precision_based=precision_based)
        #                                          for c2 in range(num_pred) for c1 in np.unique(gt))
        # print(f"took {time.time() - start} seconds")
        score_mat = np.array(score_mat)
        return score_mat.reshape((num_pred, num_gt)).T
    def _hungarian_match(self, num_pred: int, num_gt: int, pred: np.ndarray, gt: np.ndarray):
        # do hungarian matching. If num_pred > num_gt match will be partial only.
        iou_mat = self.compute_score_matrix(num_pred, num_gt, pred, gt)
        match = linear_sum_assignment(1 - iou_mat)  # pred中的类2因为softmax+max没有了, 那和2类Match到的gt类没有什么用, num_gt定义的是5
        print("Matched clusters to gt classes:")
        print(match)
        return match
    def _original_match(self, num_pred, num_gt, pred, gt, precision_based=False) -> Dict[int, list]:
        score_mat = self.compute_score_matrix(num_pred, num_gt, pred, gt, precision_based=precision_based)
        gt_class = np.unique(gt).tolist()
        preds_to_gts = {}
        preds_to_gt_scores = {}
        # Greedily match predicted class to ground-truth class by best score.
        for pred_c in range(num_pred):
            for gt_i in range(num_gt):
                score = score_mat[gt_i, pred_c]
                if (pred_c not in preds_to_gts) or (score > preds_to_gt_scores[pred_c]):
                    preds_to_gts[pred_c] = gt_class[gt_i]
                    preds_to_gt_scores[pred_c] = score
        gt_to_matches = defaultdict(list)
        for k, v in preds_to_gts.items():
            gt_to_matches[v].append(k)
        # print('original match:', gt_to_matches)
        return gt_to_matches
