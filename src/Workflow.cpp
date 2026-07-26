#include "Workflow.h"

#include "FeatureExtractor.h"

#include <STEPControl_Reader.hxx>

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace
{
    constexpr const char *kFaceCsvHeader =
        "graph_id,model_name,id,area,relativeArea,perimeter,compactness,surfaceType,centerX,centerY,centerZ,"
        "meanCurvature,radius,numWires,innerWireCount,minInnerWireLength,maxInnerWireLength,"
        "numEdges,neighborAreaMean,neighborAreaMax,areaToNeighborMean,areaToNeighborMax,"
        "normalNeighborDotMean,normalNeighborDotMin,normalNeighborDotMax,"
        "neighborPlaneCount,neighborCylinderCount,neighborCurvedCount,convexEdgeCount,concaveEdgeCount,"
        "smoothEdgeCount,convexEdgeRatio,concaveEdgeRatio,neighbors,edge_types,edge_area_ratios,"
        "edge_neighbor_surface_types,shared_edge_lengths,edge_dihedral_means,edge_dihedral_stds,label\n";
    constexpr double kMaxRivetRelativeArea = 1.0e-4;

    bool IsStepFile(const fs::path &filePath)
    {
        const std::string extension = filePath.extension().string();
        return extension == ".stp" || extension == ".step";
    }

    bool LoadShapeFromStep(const std::string &inputFile, TopoDS_Shape &shape)
    {
        STEPControl_Reader reader;
        if (reader.ReadFile(inputFile.c_str()) != IFSelect_RetDone)
        {
            return false;
        }

        reader.TransferRoots();
        shape = reader.OneShape();
        return true;
    }

    std::vector<FaceFeature> ExtractFaceFeaturesFromStep(const std::string &inputFile)
    {
        TopoDS_Shape shape;
        if (!LoadShapeFromStep(inputFile, shape))
        {
            return {};
        }

        FeatureExtractor extractor(shape);
        extractor.Extract();
        return extractor.GetResults();
    }

    void WriteFaceCsvHeader(std::ofstream &dataFile)
    {
        dataFile << kFaceCsvHeader;
    }

    struct TrainingLabelEntry
    {
        int faceId = 0;
        std::string semantic;
    };

    std::string ReadTextFile(const fs::path &filePath)
    {
        std::ifstream input(filePath);
        std::ostringstream buffer;
        buffer << input.rdbuf();
        return buffer.str();
    }

    bool ParseWingRivetLabels(const fs::path &labelsPath, std::vector<TrainingLabelEntry> &labels)
    {
        labels.clear();
        const std::string text = ReadTextFile(labelsPath);
        if (text.empty())
        {
            return false;
        }

        const std::regex faceRegex(
            R"json(\{\s*"face_id":\s*(-?\d+),\s*"semantic":\s*"([^"]+)",\s*"instance_id":\s*(-?\d+),\s*"operation":\s*"([^"]+)"\s*\})json");
        for (std::sregex_iterator it(text.begin(), text.end(), faceRegex), end; it != end; ++it)
        {
            TrainingLabelEntry entry;
            entry.faceId = std::stoi((*it)[1].str());
            entry.semantic = (*it)[2].str();
            labels.push_back(entry);
        }

        return !labels.empty();
    }

    int SemanticToTrainingLabel(const std::string &semantic)
    {
        if (semantic == "rivet")
        {
            return 1;
        }
        if (semantic == "decal")
        {
            return 2;
        }
        if (semantic == "window")
        {
            return 3;
        }
        return 0;
    }

    bool BuildFaceLabelMap(
        const std::vector<TrainingLabelEntry> &labels,
        int expectedFaceCount,
        std::vector<int> &faceLabels)
    {
        if (expectedFaceCount <= 0 || static_cast<int>(labels.size()) != expectedFaceCount)
        {
            return false;
        }

        faceLabels.assign(expectedFaceCount + 1, 0);
        std::vector<bool> seen(expectedFaceCount + 1, false);
        for (const auto &entry : labels)
        {
            if (entry.faceId < 1 || entry.faceId > expectedFaceCount || seen[entry.faceId])
            {
                return false;
            }
            seen[entry.faceId] = true;
            faceLabels[entry.faceId] = SemanticToTrainingLabel(entry.semantic);
        }

        for (int faceId = 1; faceId <= expectedFaceCount; ++faceId)
        {
            if (!seen[faceId])
            {
                return false;
            }
        }

        return true;
    }

    bool HasOversizedRivetLabel(
        const std::vector<FaceFeature> &features,
        const std::vector<int> &faceLabels,
        int &faceId,
        double &relativeArea)
    {
        for (const auto &feature : features)
        {
            if (feature.id < 1 || feature.id >= static_cast<int>(faceLabels.size()))
            {
                continue;
            }
            if (faceLabels[feature.id] == 1 && feature.relativeArea > kMaxRivetRelativeArea)
            {
                faceId = feature.id;
                relativeArea = feature.relativeArea;
                return true;
            }
        }
        return false;
    }

    std::string EscapeJson(const std::string
                               &value)
    {
        std::ostringstream escaped;
        for (const char ch : value)
        {
            switch (ch)
            {
            case '\\':
                escaped << "\\\\";
                break;
            case '"':
                escaped << "\\\"";
                break;
            case '\n':
                escaped << "\\n";
                break;
            case '\r':
                escaped << "\\r";
                break;
            case '\t':
                escaped << "\\t";
                break;
            default:
                escaped << ch;
                break;
            }
        }
        return escaped.str();
    }

    std::map<std::string, int> BuildFaceKeyHistogram(const std::vector<FaceFeature> &features)
    {
        std::map<std::string, int> histogram;
        for (const auto &feature : features)
        {
            histogram[feature.faceKey]++;
        }
        return histogram;
    }

    std::vector<std::string> CollectDuplicateFaceKeys(const std::map<std::string, int> &histogram)
    {
        std::vector<std::string> duplicates;
        for (const auto &[faceKey, count] : histogram)
        {
            if (count > 1)
            {
                duplicates.push_back(faceKey);
            }
        }
        return duplicates;
    }

    std::set<std::string> BuildFaceKeySet(const std::vector<FaceFeature> &features)
    {
        std::set<std::string> keys;
        for (const auto &feature : features)
        {
            keys.insert(feature.faceKey);
        }
        return keys;
    }

    std::vector<std::string> ComputeMissingKeys(
        const std::set<std::string> &lhs,
        const std::set<std::string> &rhs)
    {
        std::vector<std::string> missing;
        for (const auto &key : lhs)
        {
            if (rhs.find(key) == rhs.end())
            {
                missing.push_back(key);
            }
        }
        return missing;
    }

    void WriteFaceRow(
        std::ofstream &dataFile,
        int graphId,
        const std::string &modelName,
        const FaceFeature &feature)
    {
        dataFile << graphId << ",\"" << modelName << "\"," << feature.id << "," << feature.area << ","
                 << feature.relativeArea << "," << feature.perimeter << "," << feature.compactness << ","
                 << feature.surfaceType << "," << feature.centerX << "," << feature.centerY << "," << feature.centerZ << ","
                 << feature.meanCurvature << ","
                 << feature.radius << "," << feature.numWires << "," << feature.innerWireCount << ","
                 << feature.minInnerWireLength << "," << feature.maxInnerWireLength << "," << feature.numEdges << ","
                 << feature.neighborAreaMean << "," << feature.neighborAreaMax << ","
                 << feature.areaToNeighborMean << "," << feature.areaToNeighborMax << ","
                 << feature.normalNeighborDotMean << "," << feature.normalNeighborDotMin << ","
                 << feature.normalNeighborDotMax << ","
                 << feature.neighborPlaneCount << "," << feature.neighborCylinderCount << ","
                 << feature.neighborCurvedCount << "," << feature.convexEdgeCount << ","
                 << feature.concaveEdgeCount << "," << feature.smoothEdgeCount << ","
                 << feature.convexEdgeRatio << "," << feature.concaveEdgeRatio << ",\"";

        for (size_t j = 0; j < feature.neighborIds.size(); ++j)
        {
            dataFile << feature.neighborIds[j] << (j + 1 == feature.neighborIds.size() ? "" : " ");
        }

        dataFile << "\",\"";
        for (size_t j = 0; j < feature.neighborEdgeTypes.size(); ++j)
        {
            dataFile << feature.neighborEdgeTypes[j]
                     << (j + 1 == feature.neighborEdgeTypes.size() ? "" : " ");
        }

        dataFile << "\",\"";
        for (size_t j = 0; j < feature.neighborAreaRatios.size(); ++j)
        {
            dataFile << feature.neighborAreaRatios[j]
                     << (j + 1 == feature.neighborAreaRatios.size() ? "" : " ");
        }

        dataFile << "\",\"";
        for (size_t j = 0; j < feature.neighborSurfaceTypes.size(); ++j)
        {
            dataFile << feature.neighborSurfaceTypes[j]
                     << (j + 1 == feature.neighborSurfaceTypes.size() ? "" : " ");
        }

        dataFile << "\",\"";
        for (size_t j = 0; j < feature.sharedEdgeLengths.size(); ++j)
        {
            dataFile << feature.sharedEdgeLengths[j]
                     << (j + 1 == feature.sharedEdgeLengths.size() ? "" : " ");
        }

        dataFile << "\",\"";
        for (size_t j = 0; j < feature.neighborDihedralMeans.size(); ++j)
        {
            dataFile << feature.neighborDihedralMeans[j]
                     << (j + 1 == feature.neighborDihedralMeans.size() ? "" : " ");
        }

        dataFile << "\",\"";
        for (size_t j = 0; j < feature.neighborDihedralStds.size(); ++j)
        {
            dataFile << feature.neighborDihedralStds[j]
                     << (j + 1 == feature.neighborDihedralStds.size() ? "" : " ");
        }

        dataFile << "\"," << feature.semanticTag << "\n";
    }

    void ExportFaceFeaturesForInference(
        std::ofstream &dataFile,
        const std::vector<FaceFeature> &features,
        int graphId,
        const std::string &modelName)
    {
        auto rows = features;
        for (auto &feature : rows)
        {
            feature.semanticTag = 0;
            WriteFaceRow(dataFile, graphId, modelName, feature);
        }
    }

    void ExportFaceFeaturesWithTrueLabels(
        std::ofstream &dataFile,
        const std::vector<FaceFeature> &features,
        const std::vector<int> &faceLabels,
        int graphId,
        const std::string &modelName)
    {
        auto rows = features;
        for (auto &feature : rows)
        {
            if (feature.id < 1 || feature.id >= static_cast<int>(faceLabels.size()))
            {
                continue;
            }
            feature.semanticTag = faceLabels[feature.id];
            WriteFaceRow(dataFile, graphId, modelName, feature);
        }
    }

    void WriteFaceDumpJson(
        const std::string &inputFile,
        const std::vector<FaceFeature> &features,
        const std::string &outputJson)
    {
        std::ofstream jsonFile(outputJson);
        jsonFile << std::fixed << std::setprecision(6);
        jsonFile << "{\n";
        jsonFile << "  \"model_name\": \"" << EscapeJson(fs::path(inputFile).filename().string()) << "\",\n";
        jsonFile << "  \"source_step\": \"" << EscapeJson(inputFile) << "\",\n";
        jsonFile << "  \"face_count\": " << features.size() << ",\n";
        jsonFile << "  \"faces\": [\n";

        for (size_t index = 0; index < features.size(); ++index)
        {
            const auto &feature = features[index];
            jsonFile << "    {\n";
            jsonFile << "      \"traversal_id\": " << feature.id << ",\n";
            jsonFile << "      \"face_key\": \"" << EscapeJson(feature.faceKey) << "\",\n";
            jsonFile << "      \"surface_type\": " << feature.surfaceType << ",\n";
            jsonFile << "      \"area\": " << feature.area << ",\n";
            jsonFile << "      \"perimeter\": " << feature.perimeter << ",\n";
            jsonFile << "      \"center\": [" << feature.centerX << ", " << feature.centerY << ", " << feature.centerZ << "],\n";
            jsonFile << "      \"normal\": [" << feature.normalX << ", " << feature.normalY << ", " << feature.normalZ << "],\n";
            jsonFile << "      \"mean_curvature\": " << feature.meanCurvature << ",\n";
            jsonFile << "      \"radius\": " << feature.radius << ",\n";
            jsonFile << "      \"num_wires\": " << feature.numWires << ",\n";
            jsonFile << "      \"inner_wire_count\": " << feature.innerWireCount << ",\n";
            jsonFile << "      \"num_edges\": " << feature.numEdges << ",\n";
            jsonFile << "      \"neighbors\": [";

            for (size_t neighborIndex = 0; neighborIndex < feature.neighborIds.size(); ++neighborIndex)
            {
                jsonFile << feature.neighborIds[neighborIndex];
                if (neighborIndex + 1 != feature.neighborIds.size())
                {
                    jsonFile << ", ";
                }
            }

            jsonFile << "],\n";
            jsonFile << "      \"edge_types\": [";

            for (size_t edgeIndex = 0; edgeIndex < feature.neighborEdgeTypes.size(); ++edgeIndex)
            {
                jsonFile << feature.neighborEdgeTypes[edgeIndex];
                if (edgeIndex + 1 != feature.neighborEdgeTypes.size())
                {
                    jsonFile << ", ";
                }
            }

            jsonFile << "]\n";
            jsonFile << "    }";
            if (index + 1 != features.size())
            {
                jsonFile << ",";
            }
            jsonFile << "\n";
        }

        jsonFile << "  ]\n";
        jsonFile << "}\n";
    }
}

int RunWingRivetTrainingExport(const std::string &inputDir, const std::string &outputCsv)
{
    const fs::path inputPath(inputDir);
    const fs::path stepDir = inputPath / "after_two";
    const fs::path labelsDir = inputPath / "label";

    if (!fs::exists(stepDir) || !fs::exists(labelsDir))
    {
        std::cout << ">>> Missing step or label directory under: "
                  << inputDir << std::endl;
        return 1;
    }

    std::ofstream dataFile(outputCsv);
    WriteFaceCsvHeader(dataFile);

    int graphId = 0;
    int skipped = 0;
    for (const auto &entry : fs::directory_iterator(labelsDir))
    {
        if (!entry.is_regular_file() || entry.path().extension() != ".json")
        {
            continue;
        }

        const std::string suffix = ".labels";
        const std::string stem = entry.path().stem().string();
        if (stem.size() <= suffix.size() || stem.rfind(suffix) != stem.size() - suffix.size())
        {
            continue;
        }

        const std::string modelStem = stem.substr(0, stem.size() - suffix.size());
        const fs::path stepPath = stepDir / (modelStem + ".step");
        if (!fs::exists(stepPath))
        {
            std::cout << ">>> Skip " << modelStem << ": missing STEP " << stepPath << std::endl;
            skipped++;
            continue;
        }

        const auto features = ExtractFaceFeaturesFromStep(stepPath.string());
        if (features.empty())
        {
            std::cout << ">>> Skip " << modelStem << ": failed to extract features." << std::endl;
            skipped++;
            continue;
        }

        std::vector<TrainingLabelEntry> labels;
        if (!ParseWingRivetLabels(entry.path(), labels))
        {
            std::cout << ">>> Skip " << modelStem << ": failed to parse labels." << std::endl;
            skipped++;
            continue;
        }

        std::vector<int> faceLabels;
        if (!BuildFaceLabelMap(labels, static_cast<int>(features.size()), faceLabels))
        {
            std::cout << ">>> Skip " << modelStem << ": face_id/label count mismatch." << std::endl;
            skipped++;
            continue;
        }

        int oversizedFaceId = -1;
        double oversizedRelativeArea = 0.0;
        if (HasOversizedRivetLabel(features, faceLabels, oversizedFaceId, oversizedRelativeArea))
        {
            std::cout << ">>> Skip " << modelStem
                      << ": oversized rivet label face_id=" << oversizedFaceId
                      << " relativeArea=" << oversizedRelativeArea
                      << " limit=" << kMaxRivetRelativeArea << std::endl;
            skipped++;
            continue;
        }

        ExportFaceFeaturesWithTrueLabels(
            dataFile,
            features,
            faceLabels,
            graphId,
            stepPath.filename().string());

        graphId++;
        std::cout << "  - Exported: " << stepPath.filename() << std::endl;
    }

    std::cout << ">>> Wing-rivet training export complete. Models processed: " << graphId
              << ", skipped: " << skipped << std::endl;
    return graphId > 0 ? 0 : 2;
}

int RunSingleWingRivetTrainingExport(
    const std::string &inputDir,
    const std::string &modelStem,
    const std::string &outputCsv)
{
    const fs::path inputPath(inputDir);
    const fs::path stepPath = inputPath / "after_two" / (modelStem + ".step");
    const fs::path labelPath = inputPath / "label" / (modelStem + ".labels.json");
    if (!fs::exists(stepPath) || !fs::exists(labelPath))
    {
        std::cout << ">>> Missing STEP or label for: " << modelStem << std::endl;
        return 1;
    }

    const auto features = ExtractFaceFeaturesFromStep(stepPath.string());
    if (features.empty())
    {
        std::cout << ">>> Failed to extract features: " << modelStem << std::endl;
        return 2;
    }

    std::vector<TrainingLabelEntry> labels;
    if (!ParseWingRivetLabels(labelPath, labels))
    {
        std::cout << ">>> Failed to parse labels: " << modelStem << std::endl;
        return 3;
    }

    std::vector<int> faceLabels;
    if (!BuildFaceLabelMap(labels, static_cast<int>(features.size()), faceLabels))
    {
        std::cout << ">>> face_id/label count mismatch: " << modelStem << std::endl;
        return 4;
    }

    int oversizedFaceId = -1;
    double oversizedRelativeArea = 0.0;
    if (HasOversizedRivetLabel(features, faceLabels, oversizedFaceId, oversizedRelativeArea))
    {
        std::cout << ">>> Oversized rivet label: " << modelStem
                  << " face_id=" << oversizedFaceId
                  << " relativeArea=" << oversizedRelativeArea << std::endl;
        return 5;
    }

    std::ofstream dataFile(outputCsv);
    WriteFaceCsvHeader(dataFile);
    ExportFaceFeaturesWithTrueLabels(dataFile, features, faceLabels, 0, stepPath.filename().string());
    std::cout << "  - Exported: " << stepPath.filename() << std::endl;
    return 0;
}

void RunSingleInferenceExport(const std::string &inputFile, const std::string &outputCsv)
{
    std::cout << ">>> Exporting inference data for: " << inputFile << std::endl;

    const auto features = ExtractFaceFeaturesFromStep(inputFile);
    if (features.empty())
    {
        return;
    }

    const std::string modelName = fs::path(inputFile).filename().string();
    std::ofstream dataFile(outputCsv);
    WriteFaceCsvHeader(dataFile);
    ExportFaceFeaturesForInference(dataFile, features, 0, modelName);

    std::cout << ">>> Inference CSV ready." << std::endl;
}

void RunSingleFaceDump(const std::string &inputFile, const std::string &outputJson)
{
    std::cout << ">>> Exporting face dump for: " << inputFile << std::endl;

    const auto features = ExtractFaceFeaturesFromStep(inputFile);
    if (features.empty())
    {
        return;
    }

    WriteFaceDumpJson(inputFile, features, outputJson);
    std::cout << ">>> Face dump ready: " << outputJson << std::endl;
}

int RunFaceIdConsistencyCheck(const std::string &inputFile)
{
    std::cout << ">>> Checking face-id consistency for: " << inputFile << std::endl;

    const auto firstPass = ExtractFaceFeaturesFromStep(inputFile);
    const auto secondPass = ExtractFaceFeaturesFromStep(inputFile);
    if (firstPass.empty() || secondPass.empty())
    {
        std::cout << ">>> Failed to extract faces from input STEP." << std::endl;
        return 1;
    }

    const auto firstHistogram = BuildFaceKeyHistogram(firstPass);
    const auto secondHistogram = BuildFaceKeyHistogram(secondPass);
    const auto firstDuplicates = CollectDuplicateFaceKeys(firstHistogram);
    const auto secondDuplicates = CollectDuplicateFaceKeys(secondHistogram);
    const auto firstKeys = BuildFaceKeySet(firstPass);
    const auto secondKeys = BuildFaceKeySet(secondPass);
    const auto missingFromSecond = ComputeMissingKeys(firstKeys, secondKeys);
    const auto missingFromFirst = ComputeMissingKeys(secondKeys, firstKeys);

    std::cout << "First pass face_count: " << firstPass.size() << std::endl;
    std::cout << "Second pass face_count: " << secondPass.size() << std::endl;
    std::cout << "First pass unique face_key count: " << firstKeys.size() << std::endl;
    std::cout << "Second pass unique face_key count: " << secondKeys.size() << std::endl;
    std::cout << "First pass duplicate face_key count: " << firstDuplicates.size() << std::endl;
    std::cout << "Second pass duplicate face_key count: " << secondDuplicates.size() << std::endl;
    std::cout << "Missing keys from second pass: " << missingFromSecond.size() << std::endl;
    std::cout << "Missing keys from first pass: " << missingFromFirst.size() << std::endl;

    if (!firstDuplicates.empty())
    {
        std::cout << "Sample duplicate keys from first pass:" << std::endl;
        for (size_t index = 0; index < firstDuplicates.size() && index < 5; ++index)
        {
            std::cout << "  - " << firstDuplicates[index] << std::endl;
        }
    }

    if (!secondDuplicates.empty())
    {
        std::cout << "Sample duplicate keys from second pass:" << std::endl;
        for (size_t index = 0; index < secondDuplicates.size() && index < 5; ++index)
        {
            std::cout << "  - " << secondDuplicates[index] << std::endl;
        }
    }

    if (!missingFromSecond.empty())
    {
        std::cout << "Sample keys missing from second pass:" << std::endl;
        for (size_t index = 0; index < missingFromSecond.size() && index < 5; ++index)
        {
            std::cout << "  - " << missingFromSecond[index] << std::endl;
        }
    }

    if (!missingFromFirst.empty())
    {
        std::cout << "Sample keys missing from first pass:" << std::endl;
        for (size_t index = 0; index < missingFromFirst.size() && index < 5; ++index)
        {
            std::cout << "  - " << missingFromFirst[index] << std::endl;
        }
    }

    const bool isConsistent =
        firstPass.size() == secondPass.size() &&
        firstDuplicates.empty() &&
        secondDuplicates.empty() &&
        missingFromSecond.empty() &&
        missingFromFirst.empty();

    std::cout << ">>> Face-id consistency check: " << (isConsistent ? "PASS" : "FAIL") << std::endl;
    return isConsistent ? 0 : 2;
}
