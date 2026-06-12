#include "Workflow.h"

#include "FeatureExtractor.h"

#include <STEPControl_Reader.hxx>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {
constexpr const char* kFaceCsvHeader =
    "graph_id,model_name,id,area,relativeArea,perimeter,compactness,surfaceType,nx,ny,nz,"
    "centerZ,meanCurvature,radius,numWires,innerWireCount,minInnerWireLength,maxInnerWireLength,"
    "numEdges,neighbors,edge_types,label\n";

bool IsStepFile(const fs::path& filePath) {
    const std::string extension = filePath.extension().string();
    return extension == ".stp" || extension == ".step";
}

bool LoadShapeFromStep(const std::string& inputFile, TopoDS_Shape& shape) {
    STEPControl_Reader reader;
    if (reader.ReadFile(inputFile.c_str()) != IFSelect_RetDone) {
        return false;
    }

    reader.TransferRoots();
    shape = reader.OneShape();
    return true;
}

std::vector<FaceFeature> ExtractFaceFeaturesFromStep(const std::string& inputFile) {
    TopoDS_Shape shape;
    if (!LoadShapeFromStep(inputFile, shape)) {
        return {};
    }

    FeatureExtractor extractor(shape);
    extractor.Extract();
    return extractor.GetResults();
}

void WriteFaceCsvHeader(std::ofstream& dataFile) {
    dataFile << kFaceCsvHeader;
}

void WriteFaceRow(
    std::ofstream& dataFile,
    int graphId,
    const std::string& modelName,
    const FaceFeature& feature
) {
    dataFile << graphId << ",\"" << modelName << "\"," << feature.id << "," << feature.area << ","
             << feature.relativeArea << "," << feature.perimeter << "," << feature.compactness << ","
             << feature.surfaceType << "," << feature.normalX << "," << feature.normalY << ","
             << feature.normalZ << "," << feature.centerZ << "," << feature.meanCurvature << ","
             << feature.radius << "," << feature.numWires << "," << feature.innerWireCount << ","
             << feature.minInnerWireLength << "," << feature.maxInnerWireLength << "," << feature.numEdges << ",\"";

    for (size_t j = 0; j < feature.neighborIds.size(); ++j) {
        dataFile << feature.neighborIds[j] << (j + 1 == feature.neighborIds.size() ? "" : " ");
    }

    dataFile << "\",\"";
    for (size_t j = 0; j < feature.neighborEdgeTypes.size(); ++j) {
        dataFile << feature.neighborEdgeTypes[j]
                 << (j + 1 == feature.neighborEdgeTypes.size() ? "" : " ");
    }

    dataFile << "\"," << feature.semanticTag << "\n";
}

int ClassifyFaceForTraining(FaceFeature& feature) {
    const bool isSmallPrimaryFace = feature.area > 0.0 && feature.area <= 5.0;
    const bool isSmallHoleSideFace =
        isSmallPrimaryFace &&
        (feature.surfaceType == GeomAbs_Cylinder ||
         feature.surfaceType == GeomAbs_Cone ||
         feature.surfaceType == GeomAbs_Torus) &&
        feature.radius > 0.0 &&
        feature.radius <= 0.5;
    const bool isSmallHoleCapFace =
        isSmallPrimaryFace &&
        feature.surfaceType == GeomAbs_Plane &&
        feature.numWires >= 2;
    const bool isPlanarFaceWithSmallInnerHole =
        feature.surfaceType == GeomAbs_Plane &&
        feature.innerWireCount > 0 &&
        feature.minInnerWireLength > 0.0 &&
        feature.minInnerWireLength <= 8.0 &&
        feature.area <= 100.0 &&
        feature.relativeArea <= 0.02;

    if (feature.area > 50.0) {
        feature.semanticTag = isPlanarFaceWithSmallInnerHole ? 2 : 0;
    } else if (isSmallHoleSideFace || isSmallHoleCapFace || isPlanarFaceWithSmallInnerHole) {
        feature.semanticTag = 2;
    } else {
        feature.semanticTag = 1;
    }

    return feature.semanticTag;
}

void ExportFaceFeaturesForShape(
    std::ofstream& dataFile,
    const std::vector<FaceFeature>& features,
    int graphId,
    const std::string& modelName,
    bool assignTrainingLabels
) {
    auto rows = features;
    for (auto& feature : rows) {
        if (assignTrainingLabels) {
            ClassifyFaceForTraining(feature);
        } else {
            feature.semanticTag = 0;
        }
        WriteFaceRow(dataFile, graphId, modelName, feature);
    }
}
}

void RunBatchTrainingExport(const std::string& inputDir, const std::string& outputCsv) {
    std::cout << ">>> Exporting batch training data..." << std::endl;

    std::ofstream dataFile(outputCsv);
    WriteFaceCsvHeader(dataFile);

    int graphId = 0;
    for (const auto& entry : fs::directory_iterator(inputDir)) {
        if (!IsStepFile(entry.path())) {
            continue;
        }

        const auto features = ExtractFaceFeaturesFromStep(entry.path().string());
        if (features.empty()) {
            continue;
        }

        const std::string modelName = entry.path().filename().string();
        ExportFaceFeaturesForShape(dataFile, features, graphId, modelName, true);

        graphId++;
        std::cout << "  - Processed: " << entry.path().filename() << std::endl;
    }

    std::cout << ">>> Export complete. Models processed: " << graphId << std::endl;
}

void RunSingleInferenceExport(const std::string& inputFile, const std::string& outputCsv) {
    std::cout << ">>> Exporting inference data for: " << inputFile << std::endl;

    const auto features = ExtractFaceFeaturesFromStep(inputFile);
    if (features.empty()) {
        return;
    }

    const std::string modelName = fs::path(inputFile).filename().string();
    std::ofstream dataFile(outputCsv);
    WriteFaceCsvHeader(dataFile);
    ExportFaceFeaturesForShape(dataFile, features, 0, modelName, false);

    std::cout << ">>> Inference CSV ready." << std::endl;
}
