import numpy as np
from PIL import Image


def create_point_grid_mask(k, width=64, height=64):
    mask_array = np.full((height, width), 255, dtype=np.uint8)
    mask_array[::k, ::k] = 0
    return Image.fromarray(mask_array).convert("L")
        

def create_chessboard_grid_mask(k, width=64, height=64):
    cols = (width // k) + 1 # we use +1 for the case where width,height % k != 0
    rows = (height // k) + 1
    
    mask_array = np.indices((rows, cols)).sum(axis=0) % 2 # check if the sum of col+row is even
    mask_array = mask_array.repeat(k, axis=0).repeat(k, axis=1) # multiply each pixel to k*k pixels
    
    mask_array = mask_array[:height, :width] # make the size right
    
    mask_array = (mask_array * 255).astype(np.uint8) # make the 1 into 255 (white)
    
    return Image.fromarray(mask_array).convert("L") 

def create_row_slit_grid_mask(stripe_width, gap_size, width=64, height=64):

    period_pattern = np.array([0] * stripe_width + [255] * gap_size)
    
    num_reps = (height // len(period_pattern)) + 1
    vertical_column = np.tile(period_pattern, num_reps)[:height]
    
    mask_array = np.tile(vertical_column[:, np.newaxis], (1, width)).astype(np.uint8)
    
    return Image.fromarray(mask_array).convert("L")

def create_col_slit_grid_mask(stripe_width, gap_size, width=64, height=64):

    period_pattern = np.array([0] * stripe_width + [255] * gap_size)
    
    num_reps = (width // len(period_pattern)) + 1
    horizontal_row = np.tile(period_pattern, num_reps)[:width]
    
    mask_array = np.tile(horizontal_row[np.newaxis, :], (height, 1)).astype(np.uint8)
    
    return Image.fromarray(mask_array).convert("L")


def mask_to_array(mask):
    return np.array(mask).astype(np.uint8)

def combine_masks_and(mask1, mask2):
    m1, m2 = mask_to_array(mask1), mask_to_array(mask2)
    result = np.maximum(m1, m2)
    return Image.fromarray(result)

def combine_maske_or(mask2, mask1):
    m1, m2 = mask_to_array(mask1), mask_to_array(mask2)
    result = np.minimum(m1, m2)
    return Image.fromarray(result)

def combine_masks_xor(mask1, mask2):
    m1, m2 = mask_to_array(mask1), mask_to_array(mask2)
    result = (m1 == m2).astype(np.uint8) * 255
    return Image.fromarray(result)

def combine_masks_not(mask):
    m = mask_to_array(mask)
    result = 255 - m
    return Image.fromarray(result)

def create_stochastic_mask(p, width=64, height=64):
    random_matrix = np.random.rand(height, width)
    mask_array = np.where(random_matrix < p, 0, 255).astype(np.uint8)
    
    return Image.fromarray(mask_array).convert("L")

def create_focus_mask(center_x, center_y, sigma, width=64, height=64):
    y, x = np.ogrid[:height, :width]
    dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    
    prob_matrix = np.exp(-dist / (2 * sigma**2))
    
    random_matrix = np.random.rand(height, width)
    mask_array = np.where(random_matrix < prob_matrix, 0, 255).astype(np.uint8)
    return Image.fromarray(mask_array).convert("L")

def create_not_focus_mask(center_x, center_y, decay_rate, width=64, height=64):             
    return combine_masks_not(create_focus_mask(center_x, center_y, decay_rate, width, height))
