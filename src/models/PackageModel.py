from pydantic import Field, validator
from typing import List, Optional, Union, Literal
from sdks.novavision.src.base.model import Package, Image, Detection, Inputs, Configs, Outputs, Response, Request, \
    Output, Input, Config, ROI

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


class Detection(Detection):
    index: Optional[int] = None
    imgUID: Optional[str] = ""
    focus_confidence: Optional[float] = 0.0


class OutputDetections(Output):
    name: Literal["outputDetections"] = "outputDetections"
    value: List[Detection]
    type: Literal["list"] = "list"

    class Config:
        title = "Detections"


class InputDetections(Input):
    name: Literal["inputDetections"] = "inputDetections"
    value: Union[List[Detection], List[ROI]]
    type: str = "list"

    class Config:
        title = "Detections"


# 1. Zebra Warnings
class ConfigOverexposedThresholdPercent(Config):
    """
    Set the brightness percentage threshold above which pixels are marked as overexposed.
    Pixels brighter than this value will be covered with diagonal stripes.
    """
    name: Literal["ConfigOverexposedThresholdPercent"] = "ConfigOverexposedThresholdPercent"
    value: float = Field(default=0.97, ge=0, le=1)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Overexposed Threshold"
        json_schema_extra = {
            "shortDescription": "High Exposure Limit"
        }


class ConfigUnderexposedThresholdPercent(Config):
    """
    Set the brightness percentage upper threshold for underexposed pixels.
    Pixels darker than this value will be marked in blue.
    """
    name: Literal["ConfigUnderexposedThresholdPercent"] = "ConfigUnderexposedThresholdPercent"
    value: float = Field(default=0.03, ge=0, le=1)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Underexposed Threshold"
        json_schema_extra = {
            "shortDescription": "Low Exposure Limit"
        }


class ZebraWarningsFalse(Config):
    name: Literal["False"] = "False"
    value: Literal[False] = False
    type: Literal["bool"] = "bool"
    field: Literal["option"] = "option"

    class Config:
        title = "Disable"


class ZebraWarningsTrue(Config):
    configUnderexposedThresholdPercent: ConfigUnderexposedThresholdPercent
    configOverexposedThresholdPercent: ConfigOverexposedThresholdPercent
    name: Literal["True"] = "True"
    value: Literal[True] = True
    type: Literal["bool"] = "bool"
    field: Literal["option"] = "option"

    class Config:
        title = "Enable"


class ShowZebraWarnings(Config):
    """
    Display diagonal stripes on under or overexposed regions.
    Helps in identifying areas that are too bright or too dark.
    """
    name: Literal["ShowZebraWarnings"] = "ShowZebraWarnings"
    value: Union[ZebraWarningsFalse, ZebraWarningsTrue]
    type: Literal["object"] = "object"
    field: Literal["dependentDropdownlist"] = "dependentDropdownlist"

    class Config:
        title = "Show Zebra Warnings"
        json_schema_extra = {
            "shortDescription": "Exposure Stripes"
        }


# 2. Focus Peaking


class ConfigPeakingThresholdPercent(Config):
    """
    Set the sharpness threshold above which edges are marked as in-focus.
    Higher values require sharper edges to trigger the peaking overlay.
    """
    name: Literal["ConfigPeakingThresholdPercent"] = "ConfigPeakingThresholdPercent"
    value: float = Field(default=0.05, ge=0, le=1)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Focus Peaking Threshold"
        json_schema_extra = {
            "shortDescription": "Edge Sharpness Sensitivity"
        }


class FocusPeakingFalse(Config):
    name: Literal["focusPeakingFalse"] = "focusPeakingFalse"
    value: Literal[False] = False
    type: Literal["bool"] = "bool"
    field: Literal["option"] = "option"

    class Config:
        title = "Disable"


class FocusPeakingTrue(Config):
    configPeakingThresholdPercent: ConfigPeakingThresholdPercent
    name: Literal["focusPeakingTrue"] = "focusPeakingTrue"
    value: Literal[True] = True
    type: Literal["bool"] = "bool"
    field: Literal["option"] = "option"

    class Config:
        title = "Enable"


class ShowFocusPeaking(Config):
    """
    Highlight in-focus areas with a high-contrast green overlay.
    Essential for checking if the subject is perfectly sharp.
    """
    name: Literal["ShowFocusPeaking"] = "ShowFocusPeaking"
    value: Union[FocusPeakingFalse, FocusPeakingTrue]
    type: Literal["object"] = "object"
    field: Literal["dependentDropdownlist"] = "dependentDropdownlist"

    class Config:
        title = "Show Focus Peaking"
        json_schema_extra = {
            "shortDescription": "Focus Assist Overlay"
        }

# 3. Center Marker

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
    Display a crosshair at the exact center of the image.
    Useful for aiming and alignment.
    """
    name: Literal["ShowCenterMarker"] = "ShowCenterMarker"
    value: Union[CenterMarkerFalse, CenterMarkerTrue]
    type: Literal["object"] = "object"
    field: Literal["dropdownlist"] = "dropdownlist"

    class Config:
        title = "Show Center Marker"
        json_schema_extra = {
            "shortDescription": "Center Crosshair"
        }


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
    showFocusPeaking: ShowFocusPeaking
    showZebraWarnings: ShowZebraWarnings
    showCenterMarker: ShowCenterMarker


class DetectionOutputs(Outputs):
    outputImage: OutputImage
    outputDetections: OutputDetections


class DetectionRequest(Request):
    inputs: Optional[DetectionInputs]
    configs: DetectionConfigs

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
            "target": {"value": 0}
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
    showFocusPeaking: ShowFocusPeaking
    showZebraWarnings: ShowZebraWarnings
    showCenterMarker: ShowCenterMarker


class GeneralOutputs(Outputs):
    outputImage: OutputImage


class GeneralRequest(Request):
    inputs: Optional[GeneralInputs]
    configs: GeneralConfigs

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
            "target": {"value": 0}
        }


class ConfigExecutor(Config):
    """
    Select the focus assistance mode:
    'General' for full-frame analysis, or 'Detection' for specific regions.
    """
    name: Literal["ConfigExecutor"] = "ConfigExecutor"
    value: Union[GeneralExecutor, DetectionExecutor]
    type: Literal["executor"] = "executor"
    field: Literal["dependentDropdownlist"] = "dependentDropdownlist"
    restart: Literal[True] = True

    class Config:
        title = "Task"
        json_schema_extra = {
            "shortDescription": "Focus Mode"
        }


class PackageConfigs(Configs):
    executor: ConfigExecutor


class PackageModel(Package):
    configs: PackageConfigs
    type: Literal["component"] = "component"
    name: Literal["CameraFocus"] = "CameraFocus"