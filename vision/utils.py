
def sliding_window(image, window_size, step_size):
    x_start = window_size[0]//2 - 1
    x_end = image.shape[0] - window_size[0]//2 - 1
    y_start = window_size[1]//2 - 1
    y_end = image.shape[1] - window_size[1]//2 - 1

    for y in xrange(y_start, y_end, step_size):
	for x in xrange(x_start, x_end, step_size):
	    yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])
