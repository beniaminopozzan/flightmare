#!/usr/bin/env python3

import torch
import numpy as  np
from napvig_and_co.landscape import Landscape
from napvig_and_co.napvig import Napvig, State


def debug_slice (napvig: Napvig, state: State):
    debug_size = 1
    debug_precision = 0.1
    state.search =napvig.next_search (state.position, napvig.goal)
    x_step = napvig.step_ahead (state)

    search_space = napvig.get_base_orthogonal (state.search)
    g1 = search_space[:,0]
    g2 = search_space[:,1]

    x = torch.arange (-debug_size, debug_size, debug_precision)
    g1x = g1 * x.unsqueeze (1)
    x_len = len (x)
    values = torch.zeros ([x_len, x_len])

    for i in range(x_len):
        min_idx = i * x_len
        max_idx = (i + 1) * x_len
        test_points =  g1x + x[i] * g2

        for j in range(x_len):
            values[i, j] = napvig.landscape.value (test_points[i])

    # Save dat
    np.savetxt ("/root/challenge/slice_values", values.numpy ())