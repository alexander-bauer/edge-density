import cv2 as cv
import numpy as np

def stitch(images, reference=None, **kwargs):
    """Stitch multiple images together. Images must be provided in left-to-right
    order.

    If reference is set, then the image at that index will be the reference
    frame. By default, it is the middle (or middle-left) image.
    """
    if reference == None:
        # Use integer division to round down and ensure that it's an integer
        # index.
        reference = len(images)//2

    transforms = [None] * len(images)
    transforms[reference] = np.eye(3)
    # Find transforms on the left of the reference. We iterate backward from the
    # reference to the leftmost, and progressively compute the transform.
    for index in range(reference-1,-1,-1): # Iterate until 0
        imleft, imright = images[index], images[index+1]

        # Compute the transformation from the left to the right using stitch2.
        _, transform = stitch2(imleft, imright, transform_only=True)

        # Multiply this transformation by the transformation from the right
        # image to the reference. This yields the transformation from the left
        # image to the reference.
        transforms[index] = transform * transforms[index+1]

    # Do similarly for the images on the right.
    for index in range(reference+1,len(images)): # Iterate until the end
        imleft, imright = images[index-1], images[index]

        # Compute the transformation from the left to the right using stitch2.
        _, transform = stitch2(imleft, imright, transform_only=True)

        # Invert the transformation, because it will be applied to the right
        # image instead of the left.
        itransform = np.linalg.inv(transform)

        # Multiply this transformation by the transformation from the left
        # image to the reference. This yields the transformation from the right
        # image to the reference.
        transforms[index] = transforms[index-1] * itransform

    maxshape = sum((im.shape[1] for im in images)), \
            sum((im.shape[0] for im in images))
    reprojected = [cv.warpPerspective(im, H, maxshape)
            for im, H in zip(images, transforms)]

    for im in reprojected:
        cv.imshow('reprojected', im)
        cv.waitKey(0)

    result = sum(reprojected)
    return result

def stitch2(imleft, imright, transform_only=False):
    """Stitch together two images using RANSAC. Returns a panorama and the
    transformation used to produce it.

    If `transform_only` is True, then the panorama is not computed."""
    left_points, left_features = detect_and_describe(imleft)
    right_points, right_features = detect_and_describe(imright)

    print(left_points, right_points)

    return None, np.array([[1, 0, 50], [0, 1, 0], [0, 0, 1]])

def detect_and_describe(im, descriptor='SIFT'):
    gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    detector = cv.FastFeatureDetector_create(descriptor)
    extractor = cv.FastFeatureExtractor_create(descriptor)

    key_points = detector.detect(gray)
    return extractor.compute(gray, key_points)
