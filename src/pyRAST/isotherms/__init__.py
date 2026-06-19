# ruff: noqa: F401
# Import parent classes
# Import all model isotherms here so they are available at the package level
from .anrtlvst import ANRTLVST
from .bet import BET
from .dslangmuir import DSLangmuir
from .fhvst import FHVST
from .henry import Henry
from .interpolator_isotherm import CubicIsotherm, InterpolatorIsotherm
from .langmuir import Langmuir
from .model_isotherm import ModelIsotherm
from .quadratic import Quadratic
from .snrtlvst import SNRTLVST
from .temkinapprox import TemkinApprox
from .wvst import WVST

vst_models = ['W-VST', 'FH-VST', 'aNRTL-VST', 'sNRTL-VST']

def is_vst(model_name):
    return model_name in vst_models
