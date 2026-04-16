import numpy as np
from PIL import Image


def create_point_grid_mask(k, width=512, height=512):
    mask_array = np.full((height, width), 255, dtype=np.uint8)
    mask_array[::k, ::k] = 0
    return Image.fromarray(mask_array).convert("L")

def create_full_point_grid_mask(k, width=512, height=512):
    mask_array = np.full((height, width), 255, dtype=np.uint8)
    for y in range(0, height, k):
        for x in range(0, width, k):
            mask_array[y:y+8, x:x+8] = 0
        
    return Image.fromarray(mask_array).convert("L")

def create_chessboard_grid_mask(k, width=512, height=512):
    cols = (width // k) + 1 # we use +1 for the case where width,height % k != 0
    rows = (height // k) + 1
    
    mask_array = np.indices((rows, cols)).sum(axis=0) % 2 # check if the sum of col+row is even
    mask_array = mask_array.repeat(k, axis=0).repeat(k, axis=1) # multiply each pixel to k*k pixels
    
    mask_array = mask_array[:height, :width] # make the size right
    
    mask_array = (mask_array * 255).astype(np.uint8) # make the 1 into 255 (white)
    

def create_row_slit_grid_mask(stripe_width, gap_size, width=512, height=512):

    period_pattern = np.array([0] * stripe_width + [255] * gap_size)
    
    num_reps = (height // len(period_pattern)) + 1
    vertical_column = np.tile(period_pattern, num_reps)[:height]
    
    mask_array = np.tile(vertical_column[:, np.newaxis], (1, width)).astype(np.uint8)
    
    return Image.fromarray(mask_array).convert("L")

def create_col_slit_grid_mask(stripe_width, gap_size, width=512, height=512):

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

def create_stochastic_mask(p, width=512, height=512):
    random_matrix = np.random.rand(height, width)
    mask_array = np.where(random_matrix < p, 0, 255).astype(np.uint8)
    
    return Image.fromarray(mask_array).convert("L")

def create_letant_stochastic_mask(p, width=512, height=512):
    random_matrix = np.random.rand(height//8, width//8)
    mask_array = np.where(random_matrix < p, 0, 255).astype(np.uint8)
    mask_array = mask_array.repeat(8, axis=0).repeat(8, axis=1) # multiply each pixel to 8*8 pixels
    
    return Image.fromarray(mask_array).convert("L")


def create_focus_mask(center_x, center_y, decay_rate, width=512, height=512):
    mask_array = np.full((height, width), 255, dtype=np.uint8)
    
    for y in range(0, height, 8):
        for x in range(0, width, 8):
            dist = np.sqrt((x + 4 - center_x)**2 + (y + 4 - center_y)**2)

            k = 1 + (dist / 8 * decay_rate)
            grid_x, grid_y = x // 8, y // 8
            if (grid_x % int(k) == 0) and (grid_y % int(k) == 0):
                mask_array[y:y+8, x:x+8] = 0
                
    return Image.fromarray(mask_array).convert("L")

def create_not_focus_mask(center_x, center_y, decay_rate, width=512, height=512):             
    return combine_masks_not(create_focus_mask(center_x, center_y, decay_rate, width, height))
