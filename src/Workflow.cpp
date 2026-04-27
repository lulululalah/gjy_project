#include "Workflow.h"
#include "FeatureExtractor.h"
#include <STEPControl_Reader.hxx>
#include <iostream>
#include <fstream>
#include <filesystem>

namespace fs = std::filesystem;


void RunBatchTrainingExport(const std::string& inputDir, const std::string& outputCsv) {
    std::cout << ">>> 启动批量训练集导出..." << std::endl;
    std::ofstream dataFile(outputCsv);
    dataFile << "id,area,relativeArea,perimeter,compactness,surfaceType,nx,ny,nz,centerZ,meanCurvature,radius,numWires,numEdges,neighbors,edge_types,label\n";

    int offset = 0;
    for (const auto & entry : fs::directory_iterator(inputDir)) {
        if (entry.path().extension() == ".stp" || entry.path().extension() == ".step") {
            STEPControl_Reader reader;
            if (reader.ReadFile(entry.path().string().c_str()) != IFSelect_RetDone) continue;
            reader.TransferRoots();
            FeatureExtractor extractor(reader.OneShape());
            extractor.Extract();
            auto results = extractor.GetResults();

            for (auto &f : results) {
                // 标注逻辑
                if (f.area > 50.0) f.semanticTag = 0;
                else if (std::abs(f.meanCurvature) > 0.05 || f.radius > 0.1) f.semanticTag = 1;
                else if (f.compactness > 60.0 && f.area < 2.0) f.semanticTag = 2;
                else f.semanticTag = 1;

                dataFile << (offset + f.id) << "," << f.area << "," << f.relativeArea << "," << f.perimeter << "," << f.compactness << "," << f.surfaceType 
                         << "," << f.normalX << "," << f.normalY << "," << f.normalZ << "," << f.centerZ 
                         << "," << f.meanCurvature << "," << f.radius << "," << f.numWires << "," << f.numEdges << ",\"";
                for (size_t j = 0; j < f.neighborIds.size(); ++j) dataFile << (offset + f.neighborIds[j]) << (j == f.neighborIds.size() - 1 ? "" : " ");
                dataFile << "\",\"";
                for (size_t j = 0; j < f.neighborEdgeTypes.size(); ++j) dataFile << f.neighborEdgeTypes[j] << (j == f.neighborEdgeTypes.size() - 1 ? "" : " ");
                dataFile << "\"," << f.semanticTag << "\n";
            }
            offset += (int)results.size();
            std::cout << "  - 已处理: " << entry.path().filename() << std::endl;
        }
    }
    std::cout << ">>> 导出完成，总计 " << offset << " 条记录。" << std::endl;
}

void RunSingleInferenceExport(const std::string& inputFile, const std::string& outputCsv) {
    std::cout << ">>> 启动单文件识别分析: " << inputFile << std::endl;
    STEPControl_Reader reader;
    if (reader.ReadFile(inputFile.c_str()) != IFSelect_RetDone) return;
    reader.TransferRoots();
    FeatureExtractor extractor(reader.OneShape());
    extractor.Extract();
    auto results = extractor.GetResults();

    std::ofstream dataFile(outputCsv);
    dataFile << "id,area,relativeArea,perimeter,compactness,surfaceType,nx,ny,nz,centerZ,meanCurvature,radius,numWires,numEdges,neighbors,edge_types,label\n";

    for (const auto &f : results) {
        dataFile << f.id << "," << f.area << "," << f.relativeArea << "," << f.perimeter << "," << f.compactness << "," << f.surfaceType 
                 << "," << f.normalX << "," << f.normalY << "," << f.normalZ << "," << f.centerZ 
                 << "," << f.meanCurvature << "," << f.radius << "," << f.numWires << "," << f.numEdges << ",\"";
        for (size_t j = 0; j < f.neighborIds.size(); ++j) dataFile << f.neighborIds[j] << (j == f.neighborIds.size() - 1 ? "" : " ");
        dataFile << "\",\"";
        for (size_t j = 0; j < f.neighborEdgeTypes.size(); ++j) dataFile << f.neighborEdgeTypes[j] << (j == f.neighborEdgeTypes.size() - 1 ? "" : " ");
        dataFile << "\",0\n";
    }
    std::cout << ">>> 识别数据已准备。" << std::endl;
}

