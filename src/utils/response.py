
from sdks.novavision.src.helper.package import PackageHelper
from components.CameraFocus.src.models.PackageModel import PackageModel, PackageConfigs, ConfigExecutor, PackageOutputs, PackageResponse, PackageExecutor, OutputImage, DetectionOutputs, DetectionResponse, DetectionExecutor, GeneralOutputs, GeneralResponse, GeneralExecutor


def build_detections_response(context):
    detectionOutputImage = OutputImage(value=context.image)
    detectionOutputs = DetectionOutputs(outputImage=detectionOutputImage)
    detectionResponse = DetectionResponse(outputs=detectionOutputs)
    detectionExecutor = DetectionExecutor(value=detectionResponse)
    executor = ConfigExecutor(value=detectionExecutor)
    packageConfigs = PackageConfigs(executor=executor)
    package = PackageHelper(packageModel=PackageModel, packageConfigs=packageConfigs)
    packageModel = package.build_model(context)
    return packageModel

def build_general_response(context):
    generalOutputImages = OutputImage(value=context.image)
    generalOutputs = GeneralOutputs(outputImage=generalOutputImages)
    generalResponse = GeneralResponse(outputs=generalOutputs)
    generalExecutor = GeneralExecutor(value=generalResponse)
    executor = ConfigExecutor(value=generalExecutor)
    packageConfigs = PackageConfigs(executor=executor)
    package = PackageHelper(packageModel=PackageModel, packageConfigs=packageConfigs)
    packageModel = package.build_model(context)
    return packageModel