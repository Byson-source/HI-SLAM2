import os
import torch
import lietorch
import numpy as np

from lietorch import SE3
from factor_graph import FactorGraph


def _prior_scale(disps, disps_prior):
    """ per-keyframe scale mapping the depth prior onto the working-scale disps:
    median(disps) / median(disps_prior). The prior median is taken over VALID
    (> 0) pixels only, so a sparse sensor prior (LiDAR: 0 where out of range / low
    confidence) does not collapse the median to 0. Falls back to the disps median
    (scale 1) when no prior pixel is valid. Identical to the dense case (all pixels
    valid) for the Omnidata prior. """
    valid = disps_prior > 0
    pm = disps_prior[valid].median() if valid.any() else disps.median()
    return disps.median() / pm


# Metric-anchor mode: with a real (metric) sensor depth prior, fix the depth-prior
# scale ``dscales`` to a single GLOBAL working->metric value and freeze it (here +
# in geom/ba.py), instead of re-fitting it per keyframe. This lets the metric prior
# pin absolute scale across all keyframes so it stops drifting. Off => original
# per-frame free-scale behaviour (Omnidata shape prior).
_METRIC_ANCHOR = os.environ.get('HISLAM2_DEPTH_METRIC', '0') == '1'


class TrackFrontend:
    def __init__(self, net, video, config):
        self.video = video
        self.update_op = net.update
        self.graph = FactorGraph(video, net.update, max_factors=48)

        # local optimization window
        self.t1 = 0

        # frontent variables
        self.max_age = 25
        self.iters1 = 4
        self.iters2 = 2
        self.warmup = 12

        self.frontend_nms = config["frontend_nms"]
        self.keyframe_thresh = config["keyframe_thresh"]
        self.frontend_window = config["frontend_window"]
        self.frontend_thresh = config["frontend_thresh"]
        self.frontend_radius = config["frontend_radius"]
        self.video.mono_depth_alpha = config["mono_depth_alpha"]
        # allow raising the depth-prior weight at runtime; the config default (0.01) is
        # tuned for the noisy Omnidata learned prior, whereas a trustworthy sensor
        # (LiDAR) prior warrants a stronger pull. HISLAM2_MONO_ALPHA overrides it.
        _alpha_env = os.environ.get("HISLAM2_MONO_ALPHA")
        if _alpha_env is not None:
            self.video.mono_depth_alpha = float(_alpha_env)

    def __update(self, is_last):
        """ add edges, perform update """

        self.t1 += 1

        if self.graph.corr is not None:
            self.graph.rm_factors(self.graph.age > self.max_age, store=True)

        self.graph.add_proximity_factors(self.t1-5, max(self.t1-self.frontend_window, 0), 
            rad=self.frontend_radius, nms=self.frontend_nms, thresh=self.frontend_thresh, remove=True)

        if _METRIC_ANCHOR:
            # keep the frozen global metric scale: inherit the previous keyframe's
            # dscales (already tracks any global sim3 rescale from loop closure)
            self.video.dscales[self.t1-1] = self.video.dscales[self.t1-2]
        else:
            self.video.dscales[self.t1-1] = _prior_scale(self.video.disps[self.t1-1], self.video.disps_prior[self.t1-1])
        for itr in range(self.iters1):
            self.graph.update(None, None, use_inactive=True, use_mono=itr>1)

        d = self.video.distance([self.t1-3], [self.t1-2], bidirectional=True)
        d_covis = self.video.distance_covis([self.t1-2])
        covis_thresh = 0.1
        cri1 = d.item() < self.keyframe_thresh
        cri2 = d_covis.item() < covis_thresh
        if cri1 and cri2 and not is_last:
            self.graph.rm_keyframe(self.t1 - 2)
            
            with self.video.get_lock():
                self.video.counter.value -= 1
                self.t1 -= 1
            update_idx = []
        else:
            for itr in range(self.iters2):
                self.graph.update(None, None, use_inactive=True)

            if is_last:
                update_idx = torch.arange(self.graph.ii.min(), self.t1, device='cuda')
            else:
                update_idx = torch.arange(self.graph.ii.min(), self.t1-1, device='cuda')

        # set pose for next itration
        self.video.poses[self.t1] = self.video.poses[self.t1-1]
        self.video.disps[self.t1] = self.video.disps[self.t1-1].mean()

        # update visualization
        self.video.dirty[self.graph.ii.min():self.t1] = True
        return update_idx

    def __initialize(self):
        """ initialize the SLAM system """

        self.t1 = self.video.counter.value

        # initial optimization
        self.graph.add_neighborhood_factors(0, self.t1, r=3)
        for itr in range(8):
            self.graph.update(1, use_inactive=True, use_mono=False)

        # refine optimization
        self.graph.add_proximity_factors(0, 0, rad=2, nms=2, thresh=self.frontend_thresh, remove=False)
        for i in range(self.t1):
            self.video.dscales[i] = _prior_scale(self.video.disps[i], self.video.disps_prior[i])
        if _METRIC_ANCHOR:
            # collapse the per-frame prior scales to one GLOBAL median working->metric
            # scale (frozen thereafter) so the metric prior anchors absolute scale
            g = self.video.dscales[:self.t1].median()
            self.video.dscales[:self.t1] = g
        for itr in range(8):
            self.graph.update(1, use_inactive=True, use_mono=itr>2)

        # remove keyframes with too small motion
        while self.t1 > self.warmup-4:
            d = self.video.distance(torch.arange(0, self.t1-2), torch.arange(2, self.t1), bidirectional=True)
            if d.min() < self.keyframe_thresh:
                self.video.shift(d.argmin()+2, n=-1)
                self.t1 -= 1
            else:
                break

        # last optimization after removing too close keyframes
        self.graph.rm_factors(self.graph.ii > -1)
        self.graph.add_proximity_factors(0, 0, rad=2, nms=2, thresh=self.frontend_thresh, remove=False)
        for itr in range(8):
            self.graph.update(1, use_inactive=True, use_mono=itr>2)
        self.video.normalize()

        # initialization complete
        self.video.is_initialized = True
        self.video.poses[self.t1] = self.video.poses[self.t1-1].clone()
        self.video.disps[self.t1] = self.video.disps[self.t1-4:self.t1].mean()
        with self.video.get_lock():
            self.video.dirty[:self.t1] = True

        self.graph.rm_factors(self.graph.ii < self.t1-4, store=True)
        return torch.arange(self.t1-1, device='cuda')

    def __call__(self, is_last):
        """ main update """
        self.to_update = []

        # do initialization
        if not self.video.is_initialized and self.video.counter.value == self.warmup:
            self.to_update = self.__initialize()
            
        # do update
        elif self.video.is_initialized and self.t1 < self.video.counter.value:
            self.to_update = self.__update(is_last)
        
        return self.to_update
