from unittest import TestCase
from numpy import array, ndarray
from numpy.testing import (
    assert_array_equal,
    assert_array_almost_equal,
)
from CVE.Verifier import BBoxIoU

class BBoxIoUPatched(BBoxIoU):
    '''The BBoxIoU with removed bbox-capability interactions.

    This wrapper class removes BBoxIoU-class capabilities-interactions to avoid
    testing them too. Instead of passing annotation instances we're invoking
    __call__ operator with pre-converted samples in the expected bbox format
    (the job which is normally handled by capability-support code).'''

    def __init__(self, threshold):
        self.get_bbox = lambda x: x
        self.threshold = threshold

    def __call__(self, np_bb_base, np_bb_test):
        return super(self.__class__, self).__call__(np_bb_base, np_bb_test)

class IsolatedTests(TestCase):
    def setUp(self):
        self.verifier = BBoxIoUPatched(-1.)
        self.verifier_l = BBoxIoUPatched(1e-5)
        def check_idx_shape(indices, ans):
            got = indices.shape
            self.assertTupleEqual(got, ans, msg = 'wrong indices shape')
        def check_iou_shape(scores, ans):
            got = scores.shape
            self.assertTupleEqual(got, ans, msg = 'wrong scores shape')
        def check_indices(indices, expected_indices):
            msg = 'wrong the best iou indices'
            assert_array_equal(indices, expected_indices, err_msg = msg)
        def check_iou(scores, expected_scores):
            msg = 'wrong the best iou values'
            assert_array_almost_equal(scores, expected_scores,
                                      decimal = 6, err_msg = msg)
        self.check_idx_shape = check_idx_shape
        self.check_iou_shape = check_iou_shape
        self.check_indices = check_indices
        self.check_iou = check_iou
        self.opts = {'dtype': 'int32', 'order': 'C'}

    def tearDown(self):
        del self.verifier
        del self.verifier_l
        del self.check_idx_shape
        del self.check_iou_shape
        del self.check_indices
        del self.check_iou
        del self.opts

    def test_output_sizes(self):
        '''BBoxIoU arrays sizes must match bboxes count in the tested sample'''
        # since intersection-over-union is always positive we're using negative
        # number to disable filtering completely
        gt_sample = array([[1], [2], [3], [4]])
        # zero-sized array of test bounding boxes
        indices, scores = self.verifier(gt_sample, ndarray((4, 0), **self.opts))
        self.assertTupleEqual(indices.shape, (0,))
        self.assertTupleEqual(scores.shape, (0,))
        # checking output size for other input sizes
        for size in range(1, 20):
            tested_sample = ndarray((4, size), **self.opts)
            indices, scores = self.verifier(gt_sample, tested_sample)
            expected_size = (size,)
            self.assertTupleEqual(indices.shape, expected_size)
            self.assertTupleEqual(scores.shape, expected_size)

    def test_empty(self):
        '''BBoxIoU must return empty arrays for empty input'''
        indices, scores = self.verifier_l(
            array([[], [], [], []], **self.opts),
            array([[], [], [], []], **self.opts)
        )
        self.check_idx_shape(indices, (0,))
        self.check_iou_shape(scores, (0,))
        self.check_indices(indices, array([], dtype = 'int32'))
        self.check_iou(scores, array([], dtype = 'float32'))

    def test_output_indices_in_range(self):
        '''BBoxIoU best-iou indices must be in range [0, base_bboxes_count)'''
        sample = array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6]])
        indices, scores = self.verifier(sample, sample)
        for idx in indices:
            self.assertTrue(0 <= idx < 3)

    def test_output_iou_in_range(self):
        '''BBoxIoU best-iou values must be in range [0, 1]'''
        sample = array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6]])
        indices, scores = self.verifier(sample, sample)
        for score in scores:
            self.assertTrue(0.0 <= score <= 1.0)

    def test_no_intersections_iou(self):
        '''BBoxIoU must return 0 iou for non-intersected bboxes'''
        # testcase 1: no intersections here (on the left)
        base = array([[10], [10], [20], [30]], **self.opts)
        test = array([[2,  2,  2,  2,  2,  8],
                      [3,  6, 12, 25, 31,  9],
                      [9,  9,  9,  9,  9,  9],
                      [6, 15, 25, 32, 32, 31]], **self.opts)
        indices, scores = self.verifier(base, test)
        self.check_idx_shape(indices, (6,))
        self.check_iou_shape(scores, (6,))
        self.check_iou(scores, array([0, 0, 0, 0, 0, 0], dtype = 'float32'))
        # testcase 2: no intersections here (on the right)
        base = array([[10], [10], [20], [30]], **self.opts)
        test = array([[21, 21, 21, 21, 21, 21],
                      [ 3,  6, 12, 25, 31,  9],
                      [23, 23, 23, 23, 23, 23],
                      [ 6, 15, 25, 32, 32, 31]], **self.opts)
        indices, scores = self.verifier(base, test)
        self.check_idx_shape(indices, (6,))
        self.check_iou_shape(scores, (6,))
        self.check_iou(scores, array([0, 0, 0, 0, 0, 0], dtype = 'float32'))
        # testcase 3: no intersections here (lower)
        base = array([[10], [10], [30], [20]], **self.opts)
        test = array([[3,  6, 12, 25, 31,  9],
                      [2,  2,  2,  2,  2,  8],
                      [6, 15, 25, 32, 32, 31],
                      [9,  9,  9,  9,  9,  9]], **self.opts)
        indices, scores = self.verifier(base, test)
        self.check_idx_shape(indices, (6,))
        self.check_iou_shape(scores, (6,))
        self.check_iou(scores, array([0, 0, 0, 0, 0, 0], dtype = 'float32'))
        # testcase 4: no intersections here (upper)
        base = array([[10], [10], [30], [20]], **self.opts)
        test = array([[ 3,  6, 12, 25, 31,  9],
                      [21, 21, 21, 21, 21, 21],
                      [ 6, 15, 25, 32, 32, 31],
                      [23, 23, 23, 23, 23, 23]], **self.opts)
        indices, scores = self.verifier(base, test)
        self.check_idx_shape(indices, (6,))
        self.check_iou_shape(scores, (6,))
        self.check_iou(scores, array([0, 0, 0, 0, 0, 0], dtype = 'float32'))

    def test_no_intersections_idx(self):
        '''BBoxIoU must return -1 indices for non-intersected bboxes'''
        expected_indices = array([-1, -1, -1, -1, -1, -1], dtype = 'int32')
        # testcase 1: no intersections here (on the left)
        base = array([[10], [10], [20], [30]], **self.opts)
        test = array([[2,  2,  2,  2,  2,  8],
                      [3,  6, 12, 25, 31,  9],
                      [9,  9,  9,  9,  9,  9],
                      [6, 15, 25, 32, 32, 31]], **self.opts)
        indices, scores = self.verifier_l(base, test)
        self.check_idx_shape(indices, (6,))
        self.check_iou_shape(scores, (6,))
        self.check_indices(indices, expected_indices)
        # testcase 2: no intersections here (on the right)
        base = array([[10], [10], [20], [30]], **self.opts)
        test = array([[21, 21, 21, 21, 21, 21],
                      [ 3,  6, 12, 25, 31,  9],
                      [23, 23, 23, 23, 23, 23],
                      [ 6, 15, 25, 32, 32, 31]], **self.opts)
        indices, scores = self.verifier_l(base, test)
        self.check_idx_shape(indices, (6,))
        self.check_iou_shape(scores, (6,))
        self.check_indices(indices, expected_indices)
        # testcase 3: no intersections here (lower)
        base = array([[10], [10], [30], [20]], **self.opts)
        test = array([[3,  6, 12, 25, 31,  9],
                      [2,  2,  2,  2,  2,  8],
                      [6, 15, 25, 32, 32, 31],
                      [9,  9,  9,  9,  9,  9]], **self.opts)
        indices, scores = self.verifier_l(base, test)
        self.check_idx_shape(indices, (6,))
        self.check_iou_shape(scores, (6,))
        self.check_indices(indices, expected_indices)
        # testcase 4: no intersections here (upper)
        base = array([[10], [10], [30], [20]], **self.opts)
        test = array([[ 3,  6, 12, 25, 31,  9],
                      [21, 21, 21, 21, 21, 21],
                      [ 6, 15, 25, 32, 32, 31],
                      [23, 23, 23, 23, 23, 23]], **self.opts)
        indices, scores = self.verifier_l(base, test)
        self.check_idx_shape(indices, (6,))
        self.check_iou_shape(scores, (6,))
        self.check_indices(indices, expected_indices)

    def test_correct_output(self):
        '''BBoxIoU must return the correct output for indices and iou'''
        # STAGE 0: simple match test
        base = array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6]], **self.opts)
        test = array([[2], [3], [4], [5]], **self.opts)
        indices, scores = self.verifier(base, test)
        self.check_idx_shape(indices, (1,))
        self.check_iou_shape(scores, (1,))
        self.check_indices(indices, array([1], dtype = 'int32'))
        self.check_iou(scores, array([1.], dtype = 'float32'))
        # STAGE 1: testing intersection-over-union calculation (via 1 gt bbox)
        base = array([[0], [0], [2], [2]], **self.opts)
        test = array([[1, 0, 1, -1],
                      [1, 0, 0, -1],
                      [3, 1, 3,  3],
                      [3, 1, 2,  3]], **self.opts)
        indices, scores = self.verifier(base, test)
        self.check_idx_shape(indices, (4,))
        self.check_iou_shape(scores, (4,))
        self.check_indices(indices, array([0, 0, 0, 0], dtype = 'int32'))
        self.check_iou(scores, array([1./7, 1./4, 1./3, 1./4], dtype = 'float32'))
        # STAGE 2: testing selection of max iou from multiple gt bboxes
        base = array([[10, 15, 20],
                      [25, 20, 20],
                      [18, 18, 26],
                      [38, 24, 27]], **self.opts)
        test = array([[16], [22], [21], [26]], **self.opts)
        indices, scores = self.verifier(base, test)
        self.check_idx_shape(indices, (1,))
        self.check_iou_shape(scores, (1,))
        self.check_indices(indices, array([1], dtype = 'int32'))
        self.check_iou(scores, array([1./7], dtype = 'float32'))
        # STAGE 3: another test
        base = array([[ 1,  5,  3,  4],
                      [ 1,  5,  4,  3],
                      [ 6,  7, 11, 13],
                      [12, 11, 10,  7]], **self.opts)
        test = array([[2,  2],
                      [2,  2],
                      [8, 12],
                      [9,  8]], **self.opts)
        indices, scores = self.verifier(base, test)
        self.check_idx_shape(indices, (2,))
        self.check_iou_shape(scores, (2,))
        self.check_indices(indices, array([0, 3], dtype = 'int32'))
        self.check_iou(scores, array([0.405797, 0.5], dtype = 'float32'))

    def test_one_on_one(self):
        '''BBoxIoU must calculate iou correctly (1 vs 1)'''
        # testcase 1
        indices, scores = self.verifier_l(
            array([[11], [-20], [35], [-10]], **self.opts),
            array([[36], [-25], [40], [-21]], **self.opts)
        )
        self.check_indices(indices, array([-1], dtype = 'int32'))
        self.check_iou(scores, array([0.], dtype = 'float32'))
        # testcase 2
        indices, scores = self.verifier_l(
            array([[11], [-20], [35], [-10]], **self.opts),
            array([[35], [-25], [40], [-21]], **self.opts)
        )
        self.check_indices(indices, array([-1], dtype = 'int32'))
        self.check_iou(scores, array([0.], dtype = 'float32'))
        # testcase 3
        indices, scores = self.verifier_l(
            array([[10], [90], [20], [91]], **self.opts),
            array([[18], [69], [27], [70]], **self.opts)
        )
        self.check_indices(indices, array([-1], dtype = 'int32'))
        self.check_iou(scores, array([0.], dtype = 'float32'))
        # testcase 4
        indices, scores = self.verifier_l(
            array([[50], [21], [60], [23]], **self.opts),
            array([[47], [19], [62], [20]], **self.opts)
        )
        self.check_indices(indices, array([-1], dtype = 'int32'))
        self.check_iou(scores, array([0.], dtype = 'float32'))
        # testcase 5
        indices, scores = self.verifier_l(
            array([[47], [21], [62], [23]], **self.opts),
            array([[50], [19], [60], [20]], **self.opts)
        )
        self.check_indices(indices, array([-1], dtype = 'int32'))
        self.check_iou(scores, array([0.], dtype = 'float32'))
        # testcase 6
        indices, scores = self.verifier_l(
            array([[22], [16], [29], [18]], **self.opts),
            array([[19], [10], [25], [14]], **self.opts)
        )
        self.check_indices(indices, array([-1], dtype = 'int32'))
        self.check_iou(scores, array([0.], dtype = 'float32'))
        # testcase 7
        indices, scores = self.verifier_l(
            array([[60], [16], [69], [18]], **self.opts),
            array([[50], [10], [59], [14]], **self.opts)
        )
        self.check_indices(indices, array([-1], dtype = 'int32'))
        self.check_iou(scores, array([0.], dtype = 'float32'))
        # testcase 8
        indices, scores = self.verifier_l(
            array([[10], [50], [18], [55]], **self.opts),
            array([[19], [43], [20], [52]], **self.opts)
        )
        self.check_indices(indices, array([-1], dtype = 'int32'))
        self.check_iou(scores, array([0.], dtype = 'float32'))
        # testcase 9
        indices, scores = self.verifier_l(
            array([[20], [10], [28], [15]], **self.opts),
            array([[25],  [5], [31], [14]], **self.opts)
        )
        self.check_indices(indices, array([0], dtype = 'int32'))
        self.check_iou(scores, array([0.146341], dtype = 'float32'))
        # testcase 10
        indices, scores = self.verifier_l(
            array([[57], [12], [59], [26]], **self.opts),
            array([[55], [10], [61], [20]], **self.opts)
        )
        self.check_indices(indices, array([0], dtype = 'int32'))
        self.check_iou(scores, array([0.222222], dtype = 'float32'))
        # testcase 11
        indices, scores = self.verifier_l(
            array([[10], [92], [35], [95]], **self.opts),
            array([[20], [90], [30], [93]], **self.opts)
        )
        self.check_indices(indices, array([0], dtype = 'int32'))
        self.check_iou(scores, array([0.105263], dtype = 'float32'))
        # testcase 12
        indices, scores = self.verifier_l(
            array([[16], [20], [22], [41]], **self.opts),
            array([[15], [10], [18], [40]], **self.opts)
        )
        self.check_indices(indices, array([0], dtype = 'int32'))
        self.check_iou(scores, array([0.227272], dtype = 'float32'))
        # testcase 13
        indices, scores = self.verifier_l(
            array([ [16], [20],  [20], [40]], **self.opts),
            array([[-20], [10], [-16], [30]], **self.opts)
        )
        self.check_indices(indices, array([-1], dtype = 'int32'))
        self.check_iou(scores, array([0.], dtype = 'float32'))
        # testcase 14
        indices, scores = self.verifier_l(
            array([[40], [14], [41], [16]], **self.opts),
            array([[42], [10], [44], [25]], **self.opts)
        )
        self.check_indices(indices, array([-1], dtype = 'int32'))
        self.check_iou(scores, array([0.], dtype = 'float32'))
        # testcase 15
        indices, scores = self.verifier_l(
            array([[62], [-28], [68], [-20]], **self.opts),
            array([[66], [-30], [76], [-10]], **self.opts)
        )
        self.check_indices(indices, array([0], dtype = 'int32'))
        self.check_iou(scores, array([0.068965], dtype = 'float32'))
        # testcase 16
        indices, scores = self.verifier_l(
            array([[13], [59], [19], [65]], **self.opts),
            array([[15], [56], [16], [67]], **self.opts)
        )
        self.check_indices(indices, array([0], dtype = 'int32'))
        self.check_iou(scores, array([0.146341], dtype = 'float32'))
        # testcase 17
        indices, scores = self.verifier_l(
            array([[82], [13], [95], [15]], **self.opts),
            array([[81], [10], [96], [21]], **self.opts)
        )
        self.check_indices(indices, array([0], dtype = 'int32'))
        self.check_iou(scores, array([0.157575], dtype = 'float32'))
        # testcase 18
        indices, scores = self.verifier_l(
            array([[80], [125], [95], [127]], **self.opts),
            array([[79], [110], [83], [130]], **self.opts)
        )
        self.check_indices(indices, array([0], dtype = 'int32'))
        self.check_iou(scores, array([0.057692], dtype = 'float32'))
        # testcase 19
        indices, scores = self.verifier_l(
            array([[-14], [14], [-12], [15]], **self.opts),
            array([[-40], [10], [-20], [16]], **self.opts)
        )
        self.check_indices(indices, array([-1], dtype = 'int32'))
        self.check_iou(scores, array([0.], dtype = 'float32'))
        # testcase 20
        indices, scores = self.verifier_l(
            array([[-5], [18], [-3], [40]], **self.opts),
            array([ [3], [21],  [6], [35]], **self.opts)
        )
        self.check_indices(indices, array([-1], dtype = 'int32'))
        self.check_iou(scores, array([0.], dtype = 'float32'))
        # testcase 21
        indices, scores = self.verifier_l(
            array([[20], [10], [31], [19]], **self.opts),
            array([[23], [14], [35], [16]], **self.opts)
        )
        self.check_indices(indices, array([0], dtype = 'int32'))
        self.check_iou(scores, array([0.149532], dtype = 'float32'))
        # testcase 22
        indices, scores = self.verifier_l(
            array([[51], [10], [59], [60]], **self.opts),
            array([[40], [20], [63], [30]], **self.opts)
        )
        self.check_indices(indices, array([0], dtype = 'int32'))
        self.check_iou(scores, array([0.145454], dtype = 'float32'))
        # testcase 23
        indices, scores = self.verifier_l(
            array([[10], [50], [99], [77]], **self.opts),
            array([[33], [51], [69], [59]], **self.opts)
        )
        self.check_indices(indices, array([0], dtype = 'int32'))
        self.check_iou(scores, array([0.119850], dtype = 'float32'))
        # testcase 24
        indices, scores = self.verifier_l(
            array([[31], [80], [55], [95]], **self.opts),
            array([[17], [83], [42], [92]], **self.opts)
        )
        self.check_indices(indices, array([0], dtype = 'int32'))
        self.check_iou(scores, array([0.203703], dtype = 'float32'))
        # testcase 25
        indices, scores = self.verifier_l(
            array([[-15], [12], [-11], [36]], **self.opts),
            array([[-30], [13], [-20], [35]], **self.opts)
        )
        self.check_indices(indices, array([-1], dtype = 'int32'))
        self.check_iou(scores, array([0.], dtype = 'float32'))
        # testcase 26
        indices, scores = self.verifier_l(
            array([[100], [10], [120], [40]], **self.opts),
            array([[121], [35], [122], [47]], **self.opts)
        )
        self.check_indices(indices, array([-1], dtype = 'int32'))
        self.check_iou(scores, array([0.], dtype = 'float32'))
        # testcase 27
        indices, scores = self.verifier_l(
            array([[101], [1], [117], [7]], **self.opts),
            array([[109], [2], [251], [9]], **self.opts)
        )
        self.check_indices(indices, array([0], dtype = 'int32'))
        self.check_iou(scores, array([0.038095], dtype = 'float32'))
        # testcase 28
        indices, scores = self.verifier_l(
            array([[100], [13], [301], [18]], **self.opts),
            array([[150], [16], [152], [45]], **self.opts)
        )
        self.check_indices(indices, array([0], dtype = 'int32'))
        self.check_iou(scores, array([0.003777], dtype = 'float32'))
        # testcase 29
        indices, scores = self.verifier_l(
            array([[1120], [100], [1450], [141]], **self.opts),
            array([[1119], [120], [1451], [156]], **self.opts)
        )
        self.check_indices(indices, array([0], dtype = 'int32'))
        self.check_iou(scores, array([0.373544], dtype = 'float32'))
        # testcase 30
        indices, scores = self.verifier_l(
            array([[1201], [100], [1205], [121]], **self.opts),
            array([[1203], [102], [1206], [124]], **self.opts)
        )
        self.check_indices(indices, array([0], dtype = 'int32'))
        self.check_iou(scores, array([0.339285], dtype = 'float32'))
        # testcase 31
        indices, scores = self.verifier_l(
            array([[10],  [-15],  [31], [15]], **self.opts),
            array([[300], [-10], [310], [40]], **self.opts)
        )
        self.check_indices(indices, array([-1], dtype = 'int32'))
        self.check_iou(scores, array([0.], dtype = 'float32'))
        # testcase 32
        indices, scores = self.verifier_l(
            array([[100], [1], [139], [4]], **self.opts),
            array([[141], [5], [157], [7]], **self.opts)
        )
        self.check_indices(indices, array([-1], dtype = 'int32'))
        self.check_iou(scores, array([0.], dtype = 'float32'))
        # testcase 33
        indices, scores = self.verifier_l(
            array([[100], [1], [139], [4]], **self.opts),
            array([[124], [5], [157], [7]], **self.opts)
        )
        self.check_indices(indices, array([-1], dtype = 'int32'))
        self.check_iou(scores, array([0.], dtype = 'float32'))
        # testcase 34
        indices, scores = self.verifier_l(
            array([[100], [1], [139], [4]], **self.opts),
            array([[-12], [5], [169], [7]], **self.opts)
        )
        self.check_indices(indices, array([-1], dtype = 'int32'))
        self.check_iou(scores, array([0.], dtype = 'float32'))
        # testcase 35
        indices, scores = self.verifier_l(
            array([[-12], [1], [169], [4]], **self.opts),
            array([[100], [5], [139], [7]], **self.opts)
        )
        self.check_indices(indices, array([-1], dtype = 'int32'))
        self.check_iou(scores, array([0.], dtype = 'float32'))
        # testcase 36
        indices, scores = self.verifier_l(
            array([[1023], [1], [1932], [4]], **self.opts),
            array([[1022], [5], [1503], [7]], **self.opts)
        )
        self.check_indices(indices, array([-1], dtype = 'int32'))
        self.check_iou(scores, array([0.], dtype = 'float32'))
        # testcase 37
        indices, scores = self.verifier_l(
            array([[254], [1], [324], [4]], **self.opts),
            array([[120], [5], [234], [7]], **self.opts)
        )
        self.check_indices(indices, array([-1], dtype = 'int32'))
        self.check_iou(scores, array([0.], dtype = 'float32'))

    def test_thresholding(self):
        '''BBoxIoU must filter out low iou matches'''
        base = array([[0], [0], [2], [2]], **self.opts)
        test = array([[1, 0, 1, -1],
                      [1, 0, 0, -1],
                      [3, 1, 3,  3],
                      [3, 1, 2,  3]], **self.opts)
        verifier = BBoxIoUPatched(0.26)
        indices, scores = verifier(base, test)
        self.check_idx_shape(indices, (4,))
        self.check_iou_shape(scores, (4,))
        self.check_indices(indices, array([-1, -1, 0, -1], dtype = 'int32'))
        self.check_iou(scores, array([1./7, 1./4, 1./3, 1./4], dtype = 'float32'))
