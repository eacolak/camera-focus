
from sdks.novavision.src.helper.package import PackageHelper
from components.CameraFocus.src.models.PackageModel import PackageModel, PackageConfigs, ConfigExecutor, OutputImage, DetectionOutputs, DetectionResponse, DetectionExecutor, GeneralOutputs, GeneralResponse, GeneralExecutor, OutputDetections


def build_detections_response(context):
    detectionOutputImage = OutputImage(value=context.image)
    detectionOutputFocus = OutputDetections(output=context.focus_scores)
    detectionOutputs = DetectionOutputs(outputImage=detectionOutputImage, outputDetections=detectionOutputFocus)
    detectionResponse = DetectionResponse(outputs=detectionOutputs)
    detectionExecutor = DetectionExecutor(value=detectionResponse)
    executor = ConfigExecutor(value=detectionExecutor)
    packageConfigs = PackageConfigs(executor=executor)
    package = PackageHelper(packageModel=PackageModel, packageConfigs=packageConfigs)
    packageModel = package.build_model(context)
    return packageModel


def build_general_response(context):
    generalOutputImages = OutputImage(value=context.t)
    generalOutputs = GeneralOutputs(outputImage=generalOutputImages)
    generalResponse = GeneralResponse(outputs=generalOutputs)
    generalExecutor = GeneralExecutor(value=generalResponse)
    executor = ConfigExecutor(value=generalExecutor)
    packageConfigs = PackageConfigs(executor=executor)
    package = PackageHelper(packageModel=PackageModel, packageConfigs=packageConfigs)
    packageModel = package.build_model(context)
    return packageModel