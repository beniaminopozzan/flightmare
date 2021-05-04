#!/usr/bin/env python3

import torch
import numpy as  np
from napvig_and_co.landscape import Landscape
from napvig_and_co.napvig import Napvig, State


def debug_slice (napvig: Napvig, state: State):
    debug_size = 14
    debug_precision = 1.6
    state.search =napvig.next_search (state.position, napvig.goal)
    x_step = napvig.step_ahead (state)

    search_space = napvig.get_base_orthogonal (state.search)
    g1 = search_space[:,0]
    g2 = search_space[:,1]

    x = torch.arange (-debug_size, debug_size, debug_precision)
    g1x = g1 * x.unsqueeze (1)
    x_len = len (x)
    values = torch.zeros ([x_len, x_len])
    points = torch.zeros ([x_len**2, 3])

    for i in range(x_len):
        min_idx = i * x_len
        max_idx = (i + 1) * x_len
        
        test_points =  g1x + x[i] * g2  
        points[min_idx:max_idx,:] = test_points+ x_step
    
        for j in range(x_len):
            curr_val = napvig.landscape.value (test_points[j,:]+ x_step)

            values[i, j] = curr_val.norm ()
            #print (values[i,j])
    
    # Save dat
    np.savetxt ("/root/challenge/slice_values", values.numpy ())
    return points.numpy ()