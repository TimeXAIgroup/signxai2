# Get Python six functionality:
from __future__ import\
    absolute_import, print_function, division, unicode_literals


###############################################################################
###############################################################################
###############################################################################


import pytest


from signxai.tf_signxai.innvestigate.utils.tests import cases
from signxai.tf_signxai.innvestigate.utils.tests import dryrun

from signxai.tf_signxai.innvestigate.analyzer import DeepTaylor
from signxai.tf_signxai.innvestigate.analyzer import BoundedDeepTaylor


###############################################################################
###############################################################################
###############################################################################


@pytest.mark.fast
@pytest.mark.precommit
@pytest.mark.parametrize("case_id", cases.FAST)
def test_fast__DeepTaylor(case_id):

    def create_analyzer_f(model):
        return DeepTaylor(model)

    dryrun.test_analyzer(case_id, create_analyzer_f)


@pytest.mark.precommit
@pytest.mark.parametrize("case_id", cases.PRECOMMIT)
def test_precommit__DeepTaylor(case_id):

    def create_analyzer_f(model):
        return DeepTaylor(model)

    dryrun.test_analyzer(case_id, create_analyzer_f)


@pytest.mark.fast
@pytest.mark.precommit
@pytest.mark.parametrize("case_id", cases.FAST)
def test_fast__BoundedDeepTaylor(case_id):

    def create_analyzer_f(model):
        return BoundedDeepTaylor(model, low=-1, high=1)

    dryrun.test_analyzer(case_id, create_analyzer_f)


@pytest.mark.precommit
@pytest.mark.parametrize("case_id", cases.PRECOMMIT)
def test_precommit__BoundedDeepTaylor(case_id):

    def create_analyzer_f(model):
        return BoundedDeepTaylor(model, low=-1, high=1)

    dryrun.test_analyzer(case_id, create_analyzer_f)
