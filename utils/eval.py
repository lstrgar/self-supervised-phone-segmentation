import numpy as np

class PrecisionRecallMetric:
    def __init__(self, tolerance, mode):
        self.precision_counter = 0
        self.recall_counter = 0
        self.pred_counter = 0
        self.gt_counter = 0
        self.tolerance = tolerance
        self.mode = mode
        self.eps = 1e-8
        self.data = []

    def get_metrics(self, precision_counter, recall_counter, pred_counter, gt_counter):
        precision = precision_counter / (pred_counter + self.eps)
        recall = recall_counter / (gt_counter + self.eps)
        f1 = 2 * (precision * recall) / (precision + recall + self.eps)
        os = recall / (precision + self.eps) - 1
        r1 = np.sqrt((1 - recall) ** 2 + os ** 2)
        r2 = (-os + recall - 1) / (np.sqrt(2))
        rval = 1 - (np.abs(r1) + np.abs(r2)) / 2
        return precision, recall, f1, rval

    def zero(self):
        self.data = []
        self.phone_data = []

    def update(self, seg, pos_pred):
        for seg_i, pos_pred_i in zip(seg, pos_pred):
            self.data.append((seg_i, pos_pred_i))

    def get_assignments(self, y, yhat):
        matches = dict((i, []) for i in range(len(yhat)))
        for i, yhat_i in enumerate(yhat):
            dists = np.abs(y - yhat_i)
            idxs = np.argsort(dists)
            for idx in idxs:
                if dists[idx] <= self.tolerance:
                    matches[i].append(idx)
        return matches

    def get_counts(self, gt, pred):
        match_counter = 0
        dup_counter = 0
        miss_counter = 0
        used_idxs = []
        matches = self.get_assignments(gt, pred)
        dup_frames = []
        miss_frames = []

        for m, vs in matches.items():
            if len(vs) == 0:
                miss_frames.append(m)
                miss_counter += 1
                continue
            vs = sorted(vs)
            dup = False
            for v in vs:
                if v in used_idxs:
                    dup = True
                else:
                    dup = False
                    used_idxs.append(v)
                    match_counter += 1
                    break
            if dup:
                dup_counter += 1
                dup_frames.append(m)

        return match_counter, dup_counter

    def get_stats(self):
        segs = list(map(lambda x: np.array(x[0], dtype=np.int), self.data))
        yhats = list(map(lambda x: np.array(x[1], dtype=np.int), self.data))

        n_gts = 0
        n_preds = 0
        p_count = 0
        r_count = 0
        p_dup_count = 0
        r_dup_count = 0

        for y, yhat in zip(segs, yhats):
            n_gts += len(y)
            n_preds += len(yhat)
            p, pd = self.get_counts(y, yhat)
            p_count += p
            p_dup_count += pd
            r, rd = self.get_counts(yhat, y)
            r_count += r
            r_dup_count += rd

        if self.mode == "lenient":
            p_count += p_dup_count
            r_count += r_dup_count

        p, r, f1, rval = self.get_metrics(
            p_count, r_count, n_preds, n_gts
        )

        self.zero()
        
        metrics = {
            "precision": p,
            "recall": r,
            "f1": f1,
            "rval": rval
        }

        return metrics

    def update_dict(self, d, u):
        for k, v in u.items():
            if k not in d:
                d[k] = v
            else:
                d[k] += v
        return d
                

                