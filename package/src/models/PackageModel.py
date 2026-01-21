
from pydantic import Field, validator
from typing import List, Optional, Union, Literal
from sdks.novavision.src.base.model import Package, Image, Detection, Inputs, Configs, Outputs, Response, Request, Output, Input, Config


class InputImage(Input):
    name: Literal["inputImage"] = "inputImage"
    value: Union[List[Image], Image]
    type: str = "object"

    @validator("type", pre=True, always=True)
    def set_type_based_on_value(cls, value, values):
        value = values.get('value')
        if isinstance(value, Image):
            return "object"
        elif isinstance(value, list):
            return "list"

    class Config:
        title = "Image"


class OutputImage(Output):
    name: Literal["outputImage"] = "outputImage"
    value: Union[List[Image],Image]
    type: str = "object"

    @validator("type", pre=True, always=True)
    def set_type_based_on_value(cls, value, values):
        value = values.get('value')
        if isinstance(value, Image):
            return "object"
        elif isinstance(value, list):
            return "list"

    class Config:
        title = "Image"


class InputDetections(Input):
    name: Literal["inputDetections"] = "inputDetections"
    value: Union[List[Detection], List[ROI]]
    type: str = "list"

    class Config:
        title = "Detections"



# ==========================================
# 1. Zebra Warnings
# ==========================================

class ZebraWarningsFalse(Config):
    name: Literal["False"] = "False"
    value: Literal[False] = False
    type: Literal["bool"] = "bool"
    field: Literal["option"] = "option"

    class Config:
        title = "Disable"


class ZebraWarningsTrue(Config):
    name: Literal["True"] = "True"
    value: Literal[True] = True
    type: Literal["bool"] = "bool"
    field: Literal["option"] = "option"

    class Config:
        title = "Enable"


class ShowZebraWarnings(Config):
    """
        Display diagonal stripes on under/overexposed regions.
    """
    name: Literal["ShowZebraWarnings"] = "ShowZebraWarnings"
    value: Union[ZebraWarningsTrue, ZebraWarningsFalse]
    type: Literal["object"] = "object"
    field: Literal["dropdownlist"] = "dropdownlist"

    class Config:
        title = "Show Zebra Warnings"


# ==========================================
# 2. Focus Peaking
# ==========================================

class FocusPeakingFalse(Config):
    name: Literal["False"] = "False"
    value: Literal[False] = False
    type: Literal["bool"] = "bool"
    field: Literal["option"] = "option"

    class Config:
        title = "Disable"


class FocusPeakingTrue(Config):
    name: Literal["True"] = "True"
    value: Literal[True] = True
    type: Literal["bool"] = "bool"
    field: Literal["option"] = "option"

    class Config:
        title = "Enable"


class ShowFocusPeaking(Config):
    """
        Highlight in-focus areas with green overlay.
    """
    name: Literal["ShowFocusPeaking"] = "ShowFocusPeaking"
    value: Union[FocusPeakingTrue, FocusPeakingFalse]
    type: Literal["object"] = "object"
    field: Literal["dropdownlist"] = "dropdownlist"

    class Config:
        title = "Show Focus Peaking"


# ==========================================
# 3. HUD - Heads Up Display
# ==========================================

class HudFalse(Config):
    name: Literal["False"] = "False"
    value: Literal[False] = False
    type: Literal["bool"] = "bool"
    field: Literal["option"] = "option"

    class Config:
        title = "Disable"


class HudTrue(Config):
    name: Literal["True"] = "True"
    value: Literal[True] = True
    type: Literal["bool"] = "bool"
    field: Literal["option"] = "option"

    class Config:
        title = "Enable"


class ShowHUD(Config):
    """
        Display focus score and histogram overlay.
    """
    name: Literal["ShowHUD"] = "ShowHUD"
    value: Union[HudTrue, HudFalse]
    type: Literal["object"] = "object"
    field: Literal["dropdownlist"] = "dropdownlist"

    class Config:
        title = "Show HUD"


# ==========================================
# 4. Center Marker
# ==========================================

class CenterMarkerFalse(Config):
    name: Literal["False"] = "False"
    value: Literal[False] = False
    type: Literal["bool"] = "bool"
    field: Literal["option"] = "option"

    class Config:
        title = "Disable"


class CenterMarkerTrue(Config):
    name: Literal["True"] = "True"
    value: Literal[True] = True
    type: Literal["bool"] = "bool"
    field: Literal["option"] = "option"

    class Config:
        title = "Enable"


class ShowCenterMarker(Config):
    """
        Display a crosshair at the center of the image.
    """
    name: Literal["ShowCenterMarker"] = "ShowCenterMarker"
    value: Union[CenterMarkerTrue, CenterMarkerFalse]
    type: Literal["object"] = "object"
    field: Literal["dropdownlist"] = "dropdownlist"

    class Config:
        title = "Show Center Marker"


class ConfigOverexposedThresholdPercent(Config):
    """
        Set the brightness percentage threshold above which pixels are marked as overexposed.
    """
    name: Literal["conf_overexposedThreshold"] = "conf_overexposedThreshold"
    value: float = Field(default=0.97, ge=0, le=1)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Overexposed Threshold"


class ConfigUnderexposedThresholdPercent(Config):
    """
        Set the brightness percentage threshold below which pixels are marked as underexposed (blue stripes).
    """
    name: Literal["conf_underexposedThreshold"] = "conf_underexposedThreshold"
    value: float = Field(default=0.03, ge=0, le=1)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Underexposed Threshold"



# Detection Focus

class ConfigDetection(Config):
    name: Literal["configDetection"] = "configDetection"
    value: Literal["configDetection"] = "configDetection"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Detections"

class DetectionInputs(Inputs):
    inputImage: InputImage
    inputDetections: InputDetections


class DetectionConfigs(Configs):
    configDetection: ConfigDetection
    configUnderexposedThresholdPercent: ConfigUnderexposedThresholdPercent
    configOverexposedThresholdPercent: ConfigOverexposedThresholdPercent
    showCenterMarker: ShowCenterMarker
    showZebraWarnings: ShowZebraWarnings
    showHUD: ShowHUD
    showFocusPeaking: ShowFocusPeaking


class DetectionOutputs(Outputs):
    outputImage: OutputImage


class DetectionRequest(Request):
    inputs: Optional[DetectionInputs]

    class Config:
        json_schema_extra = {
            "target": "configs"
        }


class DetectionResponse(Response):
    outputs: DetectionOutputs

class DetectionExecutor(Config):
    name: Literal["Detection"] = "Detection"
    value: Union[DetectionRequest, DetectionResponse]
    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
        title = "Detections"
        json_schema_extra = {
            "target": {
                "value": 0
            }
        }


# General Focus

class ConfigGeneral(Config):
    name: Literal["configGeneral"] = "configGeneral"
    value: Literal["configGeneral"] = "configGeneral"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "General"


class GeneralInputs(Inputs):
    inputImage: InputImage


class GeneralConfigs(Configs):
    configGeneral: ConfigGeneral
    configUnderexposedThresholdPercent: ConfigUnderexposedThresholdPercent
    configOverexposedThresholdPercent: ConfigOverexposedThresholdPercent
    showCenterMarker: ShowCenterMarker
    showZebraWarnings: ShowZebraWarnings
    showHUD: ShowHUD
    showFocusPeaking: ShowFocusPeaking


class GeneralOutputs(Outputs):
    outputImage: OutputImage


class GeneralRequest(Request):
    inputs: Optional[GeneralInputs]

    class Config:
        json_schema_extra = {
            "target": "configs"
        }

class GeneralResponse(Response):
    outputs: GeneralOutputs


class GeneralExecutor(Config):
    name: Literal["General"] = "General"
    value: Union[GeneralRequest, GeneralResponse]
    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
        title = "General"
        json_schema_extra = {
            "target": {
                "value": 0
            }
        }


class ConfigExecutor(Config):
    name: Literal["ConfigExecutor"] = "ConfigExecutor"
    value: Union[GeneralExecutor, DetectionExecutor]
    type: Literal["executor"] = "executor"
    field: Literal["dependentDropdownlist"] = "dependentDropdownlist"
    restart: Literal[True] = True

    class Config:
        title = "Task"


class PackageConfigs(Configs):
    executor: ConfigExecutor


class PackageModel(Package):
    configs: PackageConfigs
    type: Literal["component"] = "component"
    name: Literal["Package"] = "CameraFocus"
