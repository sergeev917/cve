__all__ = (
    'bounding_box_capability',
    'confidence_capability',
)

def get_slots(AnnotationClass):
    # proxy-classes should be processed separately
    try:
        return AnnotationClass.__proxied_slots__
    except AttributeError:
        pass
    # with no proxy only slotted classes are supported
    try:
        return AnnotationClass.__slots__
    except AttributeError:
        raise Exception('Annotation classes without slots are not supported') from None# FIXME

def bounding_box_capability(AnnotationClass):
    '''Returns a function which produces bounding box in a predefined format.

    This function provides a unified way to obtain bounding box in a predefined
    format from various sources. For example we may need to work with bounding
    boxes but the annotation we have contains only ellipses, so we need an
    adapter to perform conversion.

    NOTE: be aware that an instance could have own method while its class could
    have the function or not. Don't forget about storage_class member for
    annotations and that it is checked for capability-related logic. Also note
    that instance-related implementation will be invoked, so an instance could
    override logic behind the function.'''
    # checking for an explicit capability implementation
    if hasattr(AnnotationClass, 'bounding_box_capability'):
        return lambda markup: markup.bounding_box_capability()
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

def blacklist_capability(AnnotationClass):
    '''Returns a function which indicates whether a sample should be ignored.

    The ignored samples should not be just thrown away. On a training/fitting
    stage it is correct behavior not to use those samples, but additionally
    the ignored samples samples should not be consumed by a hard-negatives
    sampling method. On a testing stage a match with an ignored sample should
    not be considered as a false-positive and not-matching should not be
    considered as a false-negative.'''
    # NOTE: with no actual markup we need to produce an numpy array of booleans
    #       with correct shape: (samples,). We can't know the samples count
    #       from here, thus simulation of always-false responses is not
    #       implemented.
    slots = get_slots(AnnotationClass)
    if 'numpy_blacklist' in slots:
        return lambda markup: markup.numpy_blacklist.value
    raise Exception('Blacklist capability is missing in the annotation') # FIXME
