# Get Python six functionality:
from __future__ import\
    absolute_import, print_function, division, unicode_literals


###############################################################################
###############################################################################
###############################################################################


import pytest


from signxai.tf_signxai.innvestigate.utils.tests import cases
from signxai.tf_signxai.innvestigate.utils.tests import dryrun

from signxai.tf_signxai.innvestigate.analyzer import PatternNet
from signxai.tf_signxai.innvestigate.analyzer import PatternAttribution


###############################################################################
###############################################################################
###############################################################################


@pytest.mark.fast
@pytest.mark.precommit
@pytest.mark.parametrize("case_id", cases.FAST)
def test_fast__PatternNet(case_id):

    def create_analyzer_f(model):
        # enough for test purposes, only pattern application is tested here
        # pattern computation is tested separately.
        # assume that one dim weights are biases, drop them.
        patterns = [x for x in model.get_weights()
                    if len(x.shape) > 1]
        return PatternNet(model, patterns=patterns)

    dryrun.test_analyzer(case_id, create_analyzer_f)


@pytest.mark.precommit
@pytest.mark.parametrize("case_id", cases.PRECOMMIT)
def test_precommit__PatternNet(case_id):

    def create_analyzer_f(model):
        # enough for test purposes, only pattern application is tested here
        # pattern computation is tested separately.
        # assume that one dim weights are biases, drop them.
        patterns = [x for x in model.get_weights()
                    if len(x.shape) > 1]
        return PatternNet(model, patterns=patterns)

    dryrun.test_analyzer(case_id, create_analyzer_f)


@pytest.mark.fast
@pytest.mark.precommit
@pytest.mark.parametrize("case_id", cases.PRECOMMIT)
def test_fast__PatternAttribution(case_id):

    def create_analyzer_f(model):
        # enough for test purposes, only pattern application is tested here
        # pattern computation is tested separately.
        # assume that one dim weights are biases, drop them.
        patterns = [x for x in model.get_weights()
                    if len(x.shape) > 1]
        return PatternAttribution(model, patterns=patterns)

    dryrun.test_analyzer(case_id, create_analyzer_f)


@pytest.mark.precommit
@pytest.mark.parametrize("case_id", cases.PRECOMMIT)
def test_precommit__PatternAttribution(case_id):

    def create_analyzer_f(model):
        # enough for test purposes, only pattern application is tested here
        # pattern computation is tested separately.
        # assume that one dim weights are biases, drop them.
        patterns = [x for x in model.get_weights()
                    if len(x.shape) > 1]
        return PatternAttribution(model, patterns=patterns)

    dryrun.test_analyzer(case_id, create_analyzer_f)
