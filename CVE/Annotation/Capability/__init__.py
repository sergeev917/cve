__all__ = (
    'bounding_box_capability',
    'confidence_capability',
)

def get_slots(AnnotationClass):
    if not hasattr(AnnotationClass, '__slots__'):
        raise Exception('Annotation classes without slots are not supported') # FIXME
    return AnnotationClass.__slots__

def bounding_box_capability(AnnotationClass):
    '''Returns a function which produces bounding box in a predefined format.

    This function provides a unified way to obtain bounding box in a predefined
    format from various sources. For example we may need to work with bounding
    boxes but the annotation we have contains only ellipses, so we need an
    adapter to perform conversion.'''
    slots = get_slots(AnnotationClass)
    # checking in the order of preference:
    # if we have a bbox markup itself we should not convert anything to bbox
    if 'numpy_bounding_boxes' in slots:
        return lambda markup: markup.numpy_bounding_boxes.value
    # it is all formats which are supported for now
    raise Exception('Bounding-box capability is missing in the annotation') # FIXME

def confidence_capability(AnnotationClass):
    '''Returns a function which produces confidence in a predefined format.

    This function provides a unified way to obtain confidence value in
    a predefined format.'''
    slots = get_slots(AnnotationClass)
    if 'numpy_confidences' in slots:
        return lambda markup: markup.numpy_confidences.value
    raise Exception('Confidence capability is missing in the annotation') # FIXME
