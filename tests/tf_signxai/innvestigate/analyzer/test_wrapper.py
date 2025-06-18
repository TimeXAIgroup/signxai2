# Get Python six functionality:
from __future__ import\
    absolute_import, print_function, division, unicode_literals


###############################################################################
###############################################################################
###############################################################################


import pytest


from signxai.tf_signxai.innvestigate.utils.tests import cases
from signxai.tf_signxai.innvestigate.utils.tests import dryrun

from signxai.tf_signxai.innvestigate.analyzer import WrapperBase
from signxai.tf_signxai.innvestigate.analyzer import AugmentReduceBase
from signxai.tf_signxai.innvestigate.analyzer import GaussianSmoother
from signxai.tf_signxai.innvestigate.analyzer import PathIntegrator

from signxai.tf_signxai.innvestigate.analyzer import Gradient


###############################################################################
###############################################################################
###############################################################################


@pytest.mark.fast
@pytest.mark.precommit
@pytest.mark.parametrize("case_id", cases.FAST)
def test_fast__WrapperBase(case_id):

    def create_analyzer_f(model):
        return WrapperBase(Gradient(model))

    dryrun.test_analyzer(case_id, create_analyzer_f)


@pytest.mark.precommit
@pytest.mark.parametrize("case_id", cases.PRECOMMIT)
def test_precommit__WrapperBase(case_id):

    def create_analyzer_f(model):
        return WrapperBase(Gradient(model))

    dryrun.test_analyzer(case_id, create_analyzer_f)


###############################################################################
###############################################################################
###############################################################################


@pytest.mark.fast
@pytest.mark.precommit
@pytest.mark.parametrize("case_id", cases.FAST)
def test_fast__AugmentReduceBase(case_id):

    def create_analyzer_f(model):
        return AugmentReduceBase(Gradient(model))

    dryrun.test_analyzer(case_id, create_analyzer_f)


@pytest.mark.precommit
@pytest.mark.parametrize("case_id", cases.PRECOMMIT)
def test_precommit__AugmentReduceBase(case_id):

    def create_analyzer_f(model):
        return AugmentReduceBase(Gradient(model))

    dryrun.test_analyzer(case_id, create_analyzer_f)


###############################################################################
###############################################################################
###############################################################################


@pytest.mark.fast
@pytest.mark.precommit
@pytest.mark.parametrize("case_id", cases.FAST)
def test_fast__GaussianSmoother(case_id):

    def create_analyzer_f(model):
        return GaussianSmoother(Gradient(model))

    dryrun.test_analyzer(case_id, create_analyzer_f)


@pytest.mark.precommit
@pytest.mark.parametrize("case_id", cases.PRECOMMIT)
def test_precommit__GaussianSmoother(case_id):

    def create_analyzer_f(model):
        return GaussianSmoother(Gradient(model))

    dryrun.test_analyzer(case_id, create_analyzer_f)


###############################################################################
###############################################################################
###############################################################################


@pytest.mark.fast
@pytest.mark.precommit
@pytest.mark.parametrize("case_id", cases.FAST)
def test_fast__PathIntegrator(case_id):

    def create_analyzer_f(model):
        return PathIntegrator(Gradient(model))

    dryrun.test_analyzer(case_id, create_analyzer_f)


@pytest.mark.precommit
@pytest.mark.parametrize("case_id", cases.PRECOMMIT)
def test_precommit__PathIntegrator(case_id):

    def create_analyzer_f(model):
        return PathIntegrator(Gradient(model))

    dryrun.test_analyzer(case_id, create_analyzer_f)
