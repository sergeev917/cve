__all__ = (
    'bounding_box_capability',
    'confidence_capability',
    'whitelist_capability',
)

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
    # FIXME: add exception for Dataset instance argument
    # checking for an explicit capability implementation
    if hasattr(AnnotationClass, 'bounding_box_capability'):
        return lambda markup: markup.bounding_box_capability()
    # if we have a bbox markup itself we should not convert anything to bbox
    bboxes_idx = AnnotationClass.signatures.get('std_bboxes', None)
    if bboxes_idx is not None:
        def bbox_getter(markup):
            obj = markup[bboxes_idx]
            return obj.value[:, : obj.top]
        return bbox_getter
    # it is all formats we support right now
    raise Exception('Bounding-box capability is missing in the annotation') # FIXME

def confidence_capability(AnnotationClass):
    '''Returns a function which produces confidence in a predefined format.

    This function provides a unified way to obtain confidence value in
    a predefined format.'''
    confs_idx = AnnotationClass.signatures.get('std_scores', None)
    if confs_idx is not None:
        def conf_getter(markup):
            obj = markup[confs_idx]
            return obj.value[: obj.top]
        return conf_getter
    raise Exception('Confidence capability is missing in the annotation') # FIXME

def whitelist_capability(AnnotationClass):
    '''Returns a function which shows whether a sample should not be ignored.

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
    whitelst_idx = AnnotationClass.signatures.get('std_whitelist', None)
    if whitelst_idx is not None:
        def whlst_getter(markup):
            obj = markup[whitelst_idx]
            return obj.value[: obj.top]
        return whlst_getter
    raise Exception('Whitelist capability is missing in the annotation') # FIXME
