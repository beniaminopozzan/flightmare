#!/usr/bin/env python3

import torch
import math
from napvig_and_co.landscape import *

class State:
    def __init__ (self, position = torch.zeros((3), dtype=torch.float32), search = torch.tensor ([1, 0, 0], dtype=torch.float32)):
        self.position = position
        self.search = search
    
    def __str__ (self):
        return "position: %s;\nsearch: %s" % (self.position, self.search)

    def copy (self):
        return State (self.position.clone (), self.search.clone())

from dataclasses import dataclass

@dataclass 
class NapvigParams:
    gradient_step_size : float = math.nan
    termination_distance : float = math.nan
    termination_count : int = -1
    step_ahead_size : float = math.nan
    prediction_window: int = -1
    method: str = ""

class Napvig:
    def __init__ (self, params, landscape_params):
        self.params = params
        self.landscape = Landscape (landscape_params)
    
    def set_goal (self, goal):
        self.goal = torch.from_numpy (goal)
    
    def valley_search (self, x_step, r_search):
    	# Get basis orthonormal to direction
        search_space = self.get_base_orthogonal (r_search)
        
        # Initial condition
        termination_condition = False
        x_curr = x_step.clone()
        iter_count = 0

        while (not termination_condition):
            grad_curr = self.landscape.gradient (x_curr)
            grad_project = self.project_onto (search_space, grad_curr)
            x_next = x_curr + self.params.gradient_step_size * grad_project
            update_distance = (x_next - x_curr).norm ()

            x_curr = x_next.clone ()

            termination_condition = (update_distance < self.params.termination_distance) or iter_count > self.params.termination_count
            iter_count += 1
        #print ("n iter %d" % iter_count)
        
        return x_curr

    def next_search (self, current, next):
        return (next - current) / (next - current).norm (2,0)

    def project_onto (self, space, vector):
        return space.mm (space.t()).mm(vector.unsqueeze(1)).squeeze()
    
    def get_base_orthogonal (self, base):
        _, _, v_matrix = base.unsqueeze(0).svd(False)

        return v_matrix[:,1:]
    
    def step_ahead (self, q):
        return q.position + self.params.step_ahead_size * q.search

    def compute (self, q):

        collides = False
        count = 0
        next = []

        first_step = self.compute_one (q)

        while (not collides and count < self.params.prediction_window):
            next = self.compute_one (q)
            
            # Check collision
            if (self.landscape.collides (next.position)):
                collides = True

        if (collides):
            return State ()

        return first_step

    def compute_one (self, q):
        # If directed to goal, correct search
        if (self.params.method == "goal"):
            q.search = self.next_search (q.position, self.goal)
            
        # Perform step ahead
        x_step = self.step_ahead (q)

        next = State ()

        # Get valley
        next.position = self.valley_search (x_step, q.search)
        
        # Get next bearing
        if (self.params.method == "goal"):
            next.search = self.next_search (q.position, self.goal)
        else:
            next.search = self.next_search (q.position, next.position)

        return next