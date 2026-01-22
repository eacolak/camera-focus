"""
    Detection Executor Component
    Focuses on specific detected regions.
    Inputs: inputImage, inputDetections
"""

import os
import sys



sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))

from sdks.novavision.src.media.image import Image
from sdks.novavision.src.base.component import Component
from sdks.novavision.src.helper.executor import Executor
from components.CameraFocus.src.models.PackageModel import PackageModel

from components.CameraFocus.src.utils.utils import process_image

from components.CameraFocus.src.utils.response import build_detections_response



class Detection(Component):
    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)
        self.request.model = PackageModel(**(self.request.data))
        self.center_marker = self.request.get_param("ShowCenterMarker")
        self.show_hud = self.request.get_param("ShowHUD")
        self.focus_peaking = self.request.get_param("ShowFocusPeaking")
        self.zebra_warnings = self.request.get_param("ShowZebraWarnings")
        self.over_exposed = self.request.get_param("ConfigOverexposedThresholdPercent")
        self.under_exposed = self.request.get_param("ConfigUnderexposedThresholdPercent")
        self.peaking_threshold = self.request.get_param("ConfigPeakingThresholdPercent")
        self.image = self.request.get_param("inputImage")
        self.inputDetections = self.request.get_param("inputDetections")
        self.mode = "Detection"

    @staticmethod
    def bootstrap(config: dict) -> dict:
        return {}

    def run(self):
        img = Image.get_frame(img=self.image, redis_db=self.redis_db)

        img.value = process_image(
            image=img.value,
            mode=self.mode,
            detections=self.inputDetections,
            show_center=self.center_marker,
            show_hud=self.show_hud,
            show_peaking=self.focus_peaking,
            show_zebra=self.zebra_warnings,
            thresh_over=self.over_exposed,
            thresh_under=self.under_exposed,
            peaking_threshold=self.peaking_threshold,
        )

        self.image = Image.set_frame(img=img, package_uID=self.uID, redis_db=self.redis_db)
        return build_detections_response(context=self)


if "__main__" == __name__:
    Executor(sys.argv[1]).run()