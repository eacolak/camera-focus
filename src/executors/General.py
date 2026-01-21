"""
    General Executor Component
    Focuses on the entire image.
    Inputs: inputImage
"""

import os
import sys

# Utils fonksiyonları
from components.CameraFocus.src.utils.utils import (
    process_image
)
# Response oluşturucu
from components.CameraFocus.src.utils.response import (
    build_general_response
)

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))

from sdks.novavision.src.media.image import Image
from sdks.novavision.src.base.component import Component
from sdks.novavision.src.helper.executor import Executor
from components.CameraFocus.src.models.PackageModel import PackageModel


class Package(Component):
    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)
        self.request.model = PackageModel(**(self.request.data))

        # --- Config Parametreleri (Ortak) ---
        self.center_marker = self.request.get_param("ShowCenterMarker")
        self.show_hud = self.request.get_param("ShowHUD")
        self.focus_peaking = self.request.get_param("ShowFocusPeaking")
        self.zebra_warnings = self.request.get_param("ShowZebraWarnings")
        self.over_exposed = self.request.get_param("conf_overexposedThreshold")
        self.under_exposed = self.request.get_param("conf_underexposedThreshold")

        # --- INPUTLAR ---
        # General modda sadece inputImage vardır
        self.image = self.request.get_param("inputImage")

        # --- MOD AYARLARI ---
        # Bu dosya General olduğu için mod sabittir
        self.mode = "General"
        self.inputDetections = None  # Detection verisi yok

    @staticmethod
    def bootstrap(config: dict) -> dict:
        return {}

    def run(self):
        img = Image.get_frame(img=self.image, redis_db=self.redis_db)

        img.value = process_image(
            image=img.value,
            mode=self.mode,               # "General"
            detections=None,              # Boş
            show_center=self.center_marker,
            show_hud=self.show_hud,
            show_peaking=self.focus_peaking,
            show_zebra=self.zebra_warnings,
            thresh_over=self.over_exposed,
            thresh_under=self.under_exposed
        )

        self.image = Image.set_frame(img=img, package_uID=self.uID, redis_db=self.redis_db)
        return build_general_response(context=self)


if "__main__" == __name__:
    Executor(sys.argv[1]).run()