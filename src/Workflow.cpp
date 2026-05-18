#include "Workflow.h"
#include "FeatureExtractor.h"

#include <STEPControl_Reader.hxx>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>

namespace fs = std::filesystem;

void RunBatchTrainingExport(const std::string& inputDir, const std::string& outputCsv) {
    std::cout << ">>> Exporting batch training data..." << std::endl;

    std::ofstream dataFile(outputCsv);
    dataFile << "graph_id,model_name,id,area,relativeArea,perimeter,compactness,surfaceType,nx,ny,nz,centerZ,meanCurvature,radius,numWires,numEdges,neighbors,edge_types,label\n";

    int graphId = 0;
    for (const auto& entry : fs::directory_iterator(inputDir)) {
        const auto extension = entry.path().extension().string();
        if (extension != ".stp" && extension != ".step") {
            continue;
        }

        STEPControl_Reader reader;
        if (reader.ReadFile(entry.path().string().c_str()) != IFSelect_RetDone) {
            continue;
        }

        reader.TransferRoots();
        FeatureExtractor extractor(reader.OneShape());
        extractor.Extract();
        auto results = extractor.GetResults();
        const std::string modelName = entry.path().filename().string();

        for (auto& feature : results) {
            if (feature.area > 50.0) {
                feature.semanticTag = 0;
            } else if (std::abs(feature.meanCurvature) > 0.05 || feature.radius > 0.1) {
                feature.semanticTag = 1;
            } else if (feature.compactness > 60.0 && feature.area < 2.0) {
                feature.semanticTag = 2;
            } else {
                feature.semanticTag = 1;
            }

            dataFile << graphId << ",\"" << modelName << "\"," << feature.id << "," << feature.area << ","
                     << feature.relativeArea << "," << feature.perimeter << "," << feature.compactness << ","
                     << feature.surfaceType << "," << feature.normalX << "," << feature.normalY << ","
                     << feature.normalZ << "," << feature.centerZ << "," << feature.meanCurvature << ","
                     << feature.radius << "," << feature.numWires << "," << feature.numEdges << ",\"";

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

        graphId++;
        std::cout << "  - Processed: " << entry.path().filename() << std::endl;
    }

    std::cout << ">>> Export complete. Models processed: " << graphId << std::endl;
}

void RunSingleInferenceExport(const std::string& inputFile, const std::string& outputCsv) {
    std::cout << ">>> Exporting inference data for: " << inputFile << std::endl;

    STEPControl_Reader reader;
    if (reader.ReadFile(inputFile.c_str()) != IFSelect_RetDone) {
        return;
    }

    reader.TransferRoots();
    FeatureExtractor extractor(reader.OneShape());
    extractor.Extract();
    auto results = extractor.GetResults();
    const std::string modelName = fs::path(inputFile).filename().string();

    std::ofstream dataFile(outputCsv);
    dataFile << "graph_id,model_name,id,area,relativeArea,perimeter,compactness,surfaceType,nx,ny,nz,centerZ,meanCurvature,radius,numWires,numEdges,neighbors,edge_types,label\n";

    for (const auto& feature : results) {
        dataFile << 0 << ",\"" << modelName << "\"," << feature.id << "," << feature.area << ","
                 << feature.relativeArea << "," << feature.perimeter << "," << feature.compactness << ","
                 << feature.surfaceType << "," << feature.normalX << "," << feature.normalY << ","
                 << feature.normalZ << "," << feature.centerZ << "," << feature.meanCurvature << ","
                 << feature.radius << "," << feature.numWires << "," << feature.numEdges << ",\"";

        for (size_t j = 0; j < feature.neighborIds.size(); ++j) {
            dataFile << feature.neighborIds[j] << (j + 1 == feature.neighborIds.size() ? "" : " ");
        }

        dataFile << "\",\"";
        for (size_t j = 0; j < feature.neighborEdgeTypes.size(); ++j) {
            dataFile << feature.neighborEdgeTypes[j]
                     << (j + 1 == feature.neighborEdgeTypes.size() ? "" : " ");
        }

        dataFile << "\",0\n";
    }

    std::cout << ">>> Inference CSV ready." << std::endl;
}
