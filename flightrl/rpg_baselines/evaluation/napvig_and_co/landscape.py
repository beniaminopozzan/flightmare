#!/usr/bin/env python3

import torch
import math

from dataclasses import dataclass

@dataclass 
class LandscapeParams:
    measure_radius: float = 0
    smooth_radius: float = 0
    precision: int = 0
    dim: int = 0
    decimation: int = 0
    min_distance: float = 0

class Landscape:
    def __init__(self, params):
        self.params = params
        self.no_amplification_gain = self.get_no_amplification_gain ()
        self.smooth_gain = self.get_smooth_gain ()

    def set_measures (self, measures_np):
        measures = torch.from_numpy (measures_np)
        self.measures = measures.view ([measures.size(0), 1, self.params.dim])
        self.measures = self.measures[::self.params.decimation, :, :]

    def gamma_distance (self, distance):
	    return math.exp (-(distance * distance)/(2 * self.params.measure_radius * self.params.measure_radius))
    
    def collides (self, x):
        return self.value (x) > self.gamma_distance (self.params.min_distance)

    def get_no_amplification_gain (self):
        sqMR = self.params.measure_radius ** 2
        sqSR = self.params.smooth_radius ** 2
        k =  (1/sqMR) + (1/sqSR)
        return math.pow (k / (2 * math.pi), self.params.dim/2)
        
    def get_smooth_gain (self):
        return math.pow (2 * math.pi * self.params.smooth_radius**2, self.params.dim/2)* self.no_amplification_gain

    def gaussian (self, x, sigma):
        return (x * (-0.5/(sigma*sigma))).exp ()

    def exponential_part (self, collapsed_dist):
        return self.gaussian (collapsed_dist, self.params.measure_radius)

    def norm_square (self, x):
        return x.pow(2).sum(2)
    
    def collapsed_diff (self, x):
        measures_diff = x - self.measures
        dist_to_measures = self.norm_square (measures_diff)
        collapsed_dist, idxes = dist_to_measures.min (0)

        return measures_diff.permute ((1,0,2))[torch.arange(measures_diff.size()[1]),idxes,:], collapsed_dist

    def potential_function (self, x):
        collapsed, collapsed_dist = self.collapsed_diff (x)
        
        return self.exponential_part (collapsed_dist)

    def pre_smooth (self, x):
        return self.potential_function (x) * self.smooth_gain

    def pre_smooth_gradient (self, x):
        diff, dist = self.collapsed_diff (x)
        return diff * self.exponential_part (dist).unsqueeze(1) * self.smooth_gain
    
    def montecarlo (self, func, dim, count, center, variance):
        xVar = torch.normal (0.0, variance, (count, dim))
        
        xEval = xVar + center.expand ((count,dim))

        return func (xEval).mean (0)

    def value (self, x):
        return self.montecarlo (self.pre_smooth, self.params.dim, self.params.precision, x.view ((1, self.params.dim)), self.params.smooth_radius)

    def gradient (self, x):
        return self.montecarlo (self.pre_smooth_gradient, self.params.dim, self.params.precision, x.view ((1, self.params.dim)), self.params.smooth_radius)
