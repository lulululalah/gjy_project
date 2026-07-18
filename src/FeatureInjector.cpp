#include "FeatureInjector.h"

#include "FeatureExtractor.h"

#include <BRepAlgoAPI_Fuse.hxx>
#include <BRepBndLib.hxx>
#include <BRepBuilderAPI_MakeFace.hxx>
#include <BRepBuilderAPI_MakePolygon.hxx>
#include <BRepFeat_SplitShape.hxx>
#include <BRep_Builder.hxx>
#include <BRep_Tool.hxx>
#include <BRepCheck_Analyzer.hxx>
#include <BRepClass_FaceClassifier.hxx>
#include <BRepLProp_SLProps.hxx>
#include <BRepPrimAPI_MakeCylinder.hxx>
#include <BRepPrimAPI_MakePrism.hxx>
#include <BRepTools_ReShape.hxx>
#include <BRepTools.hxx>
#include <BRepAdaptor_Surface.hxx>
#include <Bnd_Box.hxx>
#include <Geom2d_Curve.hxx>
#include <GeomAbs_SurfaceType.hxx>
#include <Precision.hxx>
#include <STEPControl_Reader.hxx>
#include <STEPControl_Writer.hxx>
#include <TopAbs_State.hxx>
#include <TopExp.hxx>
#include <TopExp_Explorer.hxx>
#include <TopTools_IndexedMapOfShape.hxx>
#include <TopTools_IndexedDataMapOfShapeListOfShape.hxx>
#include <TopoDS.hxx>
#include <TopoDS_Compound.hxx>
#include <TopoDS_Edge.hxx>
#include <TopoDS_Face.hxx>
#include <gp_Ax2.hxx>
#include <gp_Dir.hxx>
#include <gp_Pnt.hxx>
#include <gp_Pnt2d.hxx>
#include <gp_Vec.hxx>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

enum class RivetFootprint {
    Round,
    Hexagon,
};

// Data containers used by the wing-rivet injection and validation pipeline.
struct RivetPlacement {
    int instanceId = -1;
    int hostFaceId = -1;
    double u = 0.0;
    double v = 0.0;
    double radius = 0.0;
    double height = 0.0;
    RivetFootprint footprint = RivetFootprint::Round;
    gp_Pnt basePoint;
    gp_Dir normal;
};

struct WingHostFace {
    int faceId = -1;
    double score = -1.0;
    FaceFeature feature;
};

struct LabelsFaceEntry {
    int faceId = -1;
    std::string semantic = "background";
    int instanceId = -1;
    std::string operation = "keep";
};

struct LabelsInstanceEntry {
    int instanceId = -1;
    std::string type = "rivet";
    int hostFaceId = -1;
    double radius = 0.0;
    double height = 0.0;
};

struct RivetFaceSignature {
    int instanceId = -1;
    int surfaceType = 0;
    int numEdges = 0;
    double area = 0.0;
    double centerX = 0.0;
    double centerY = 0.0;
    double centerZ = 0.0;
    double radius = 0.0;
};

struct InjectionTarget {
    TopoDS_Shape originalSubshape;
    TopoDS_Shape workingShape;
    bool isSubshapeMode = false;
    bool isPerSolidMode = false;
    std::map<int, int> solidIndexByHostFaceId;
    std::map<int, TopoDS_Shape> originalSolidByIndex;
    std::map<int, TopoDS_Shape> workingSolidByIndex;
};

struct LabelsData {
    std::vector<LabelsFaceEntry> faces;
    std::vector<LabelsInstanceEntry> instances;
};

struct DatasetValidationResult {
    std::string modelName;
    std::string stepPath;
    std::string labelsPath;
    bool stepExists = false;
    bool labelsParsed = false;
    bool shapeLoaded = false;
    bool shapeValid = false;
    bool t1 = false;
    bool t2 = false;
    bool t3 = false;
    int faceCount = 0;
    int labelFaceCount = 0;
    int rivetFaceCount = 0;
    int backgroundFaceCount = 0;
    int instanceCount = 0;
    int duplicateFaceKeyCount = 0;
    int graphMismatchCount = 0;
    std::string message;
};

bool IsShapeValid(const TopoDS_Shape& shape);
double ExpectedRivetTopArea(const RivetPlacement& placement);

// Basic STEP, CSV, JSON, and face-feature helpers.
std::string ShapeTypeName(TopAbs_ShapeEnum shapeType) {
    switch (shapeType) {
    case TopAbs_COMPOUND:
        return "COMPOUND";
    case TopAbs_COMPSOLID:
        return "COMPSOLID";
    case TopAbs_SOLID:
        return "SOLID";
    case TopAbs_SHELL:
        return "SHELL";
    case TopAbs_FACE:
        return "FACE";
    case TopAbs_WIRE:
        return "WIRE";
    case TopAbs_EDGE:
        return "EDGE";
    case TopAbs_VERTEX:
        return "VERTEX";
    case TopAbs_SHAPE:
        return "SHAPE";
    default:
        return "UNKNOWN";
    }
}

int CountSubShapes(const TopoDS_Shape& shape, TopAbs_ShapeEnum shapeType) {
    if (shape.IsNull()) {
        return 0;
    }

    TopTools_IndexedMapOfShape shapeMap;
    TopExp::MapShapes(shape, shapeType, shapeMap);
    return shapeMap.Extent();
}

bool LoadShapeFromStep(const std::string& inputFile, TopoDS_Shape& shape) {
    STEPControl_Reader reader;
    if (reader.ReadFile(inputFile.c_str()) != IFSelect_RetDone) {
        return false;
    }

    reader.TransferRoots();
    shape = reader.OneShape();
    return !shape.IsNull();
}

bool SaveShapeToStep(const TopoDS_Shape& shape, const std::string& outputFile) {
    STEPControl_Writer writer;
    const IFSelect_ReturnStatus transferStatus = writer.Transfer(shape, STEPControl_AsIs);
    if (transferStatus != IFSelect_RetDone) {
        return false;
    }

    return writer.Write(outputFile.c_str()) == IFSelect_RetDone;
}

std::string EscapeJson(const std::string& value) {
    std::ostringstream escaped;
    for (const char ch : value) {
        switch (ch) {
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

bool IsStepFile(const fs::path& filePath) {
    std::string extension = filePath.extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    return extension == ".stp" || extension == ".step";
}

bool IsGeneratedWingRivetStep(const fs::path& filePath) {
    const std::string stem = filePath.stem().string();
    return stem.size() >= 12 && stem.find("_wing_rivets") != std::string::npos;
}

std::vector<FaceFeature> ExtractFeatures(const TopoDS_Shape& shape) {
    FeatureExtractor extractor(shape);
    extractor.Extract();
    return extractor.GetResults();
}

std::string CsvEscape(const std::string& value) {
    std::ostringstream escaped;
    escaped << '"';
    for (const char ch : value) {
        if (ch == '"') {
            escaped << "\"\"";
        } else {
            escaped << ch;
        }
    }
    escaped << '"';
    return escaped.str();
}

std::string ReadTextFile(const fs::path& filePath) {
    std::ifstream input(filePath);
    std::ostringstream buffer;
    buffer << input.rdbuf();
    return buffer.str();
}

bool ParseLabelsJson(const fs::path& labelsPath, LabelsData& labels) {
    labels = {};
    const std::string text = ReadTextFile(labelsPath);
    if (text.empty()) {
        return false;
    }

    const std::regex faceRegex(
        R"json(\{"face_id":\s*(-?\d+),\s*"semantic":\s*"([^"]+)",\s*"instance_id":\s*(-?\d+),\s*"operation":\s*"([^"]+)"\})json"
    );
    for (std::sregex_iterator it(text.begin(), text.end(), faceRegex), end; it != end; ++it) {
        LabelsFaceEntry entry;
        entry.faceId = std::stoi((*it)[1].str());
        entry.semantic = (*it)[2].str();
        entry.instanceId = std::stoi((*it)[3].str());
        entry.operation = (*it)[4].str();
        labels.faces.push_back(entry);
    }

    const std::regex instanceRegex(
        R"json(\{"instance_id":\s*(-?\d+),\s*"type":\s*"([^"]+)",\s*"host_face":\s*(-?\d+),\s*"inverse_op":\s*\{"kind":\s*"([^"]+)",\s*"radius":\s*([-+0-9.eE]+),\s*"height":\s*([-+0-9.eE]+)\}\})json"
    );
    for (std::sregex_iterator it(text.begin(), text.end(), instanceRegex), end; it != end; ++it) {
        LabelsInstanceEntry entry;
        entry.instanceId = std::stoi((*it)[1].str());
        entry.type = (*it)[2].str();
        entry.hostFaceId = std::stoi((*it)[3].str());
        entry.radius = std::stod((*it)[5].str());
        entry.height = std::stod((*it)[6].str());
        labels.instances.push_back(entry);
    }

    return !labels.faces.empty();
}

// Dataset validation helpers.
std::map<std::string, int> BuildFaceKeyCounts(const std::vector<FaceFeature>& features) {
    std::map<std::string, int> counts;
    for (const auto& feature : features) {
        counts[feature.faceKey]++;
    }
    return counts;
}

int CountDuplicateFaceKeys(const std::vector<FaceFeature>& features) {
    int duplicates = 0;
    for (const auto& [key, count] : BuildFaceKeyCounts(features)) {
        if (count > 1) {
            duplicates++;
        }
    }
    return duplicates;
}

bool HaveStableFaceOrder(const std::vector<FaceFeature>& first, const std::vector<FaceFeature>& second) {
    if (first.size() != second.size()) {
        return false;
    }
    for (size_t index = 0; index < first.size(); ++index) {
        if (first[index].faceKey != second[index].faceKey) {
            return false;
        }
    }
    return true;
}

int CountGraphMismatches(const std::vector<FaceFeature>& first, const std::vector<FaceFeature>& second) {
    if (first.size() != second.size()) {
        return static_cast<int>(std::max(first.size(), second.size()));
    }

    int mismatches = 0;
    for (size_t index = 0; index < first.size(); ++index) {
        if (first[index].neighborIds != second[index].neighborIds ||
            first[index].neighborEdgeTypes != second[index].neighborEdgeTypes) {
            mismatches++;
        }
    }
    return mismatches;
}

bool ValidateLabelFaceIds(const LabelsData& labels, int faceCount) {
    if (faceCount <= 0 || static_cast<int>(labels.faces.size()) != faceCount) {
        return false;
    }

    std::set<int> seen;
    for (const auto& face : labels.faces) {
        if (face.faceId < 1 || face.faceId > faceCount || !seen.insert(face.faceId).second) {
            return false;
        }
    }
    return static_cast<int>(seen.size()) == faceCount;
}

bool ValidateRivetLabels(const LabelsData& labels) {
    std::set<int> instanceIds;
    for (const auto& instance : labels.instances) {
        const bool isRivet = instance.type == "rivet" && instance.height > 0.0;
        const bool isDecal = instance.type == "decal" && instance.height == 0.0;
        if (instance.instanceId <= 0 || (!isRivet && !isDecal) ||
            instance.hostFaceId <= 0 || instance.radius <= 0.0) {
            return false;
        }
        instanceIds.insert(instance.instanceId);
    }

    for (const auto& face : labels.faces) {
        if (face.semantic == "rivet") {
            if (face.instanceId <= 0 || face.operation != "remove_protrusion" ||
                instanceIds.find(face.instanceId) == instanceIds.end()) {
                return false;
            }
        } else if (face.semantic == "decal") {
            if (face.instanceId <= 0 || face.operation != "remove_decal_boundary" ||
                instanceIds.find(face.instanceId) == instanceIds.end()) {
                return false;
            }
        } else if (face.semantic == "background") {
            if (face.instanceId != -1 || face.operation != "keep") {
                return false;
            }
        } else {
            return false;
        }
    }
    return !labels.instances.empty();
}

// Rivet-face matching helpers. STEP export can renumber faces, so labels are
// recovered by matching generated rivet geometry back onto the reloaded model.
double SquaredDistance(const FaceFeature& lhs, const RivetFaceSignature& rhs) {
    const double dx = lhs.centerX - rhs.centerX;
    const double dy = lhs.centerY - rhs.centerY;
    const double dz = lhs.centerZ - rhs.centerZ;
    return dx * dx + dy * dy + dz * dz;
}

RivetFaceSignature BuildRivetFaceSignature(int instanceId, const FaceFeature& feature) {
    return {
        instanceId,
        feature.surfaceType,
        feature.numEdges,
        feature.area,
        feature.centerX,
        feature.centerY,
        feature.centerZ,
        feature.radius,
    };
}

std::vector<RivetFaceSignature> BuildRivetFaceSignatures(
    const TopoDS_Shape& shape,
    const std::vector<std::pair<int, std::vector<TopoDS_Shape>>>& rivetFacesByInstance
) {
    std::vector<RivetFaceSignature> signatures;
    const auto features = ExtractFeatures(shape);

    TopTools_IndexedMapOfShape faceMap;
    TopExp::MapShapes(shape, TopAbs_FACE, faceMap);
    for (const auto& [instanceId, rivetFaces] : rivetFacesByInstance) {
        std::set<int> faceIds;
        for (const auto& rivetFace : rivetFaces) {
            const int faceId = faceMap.FindIndex(rivetFace);
            if (faceId > 0 && faceId <= static_cast<int>(features.size())) {
                faceIds.insert(faceId);
            }
        }
        for (const int faceId : faceIds) {
            signatures.push_back(BuildRivetFaceSignature(instanceId, features[static_cast<size_t>(faceId - 1)]));
        }
    }

    return signatures;
}

double ComputeRivetSignatureScore(const FaceFeature& feature, const RivetFaceSignature& signature) {
    if (feature.surfaceType != signature.surfaceType) {
        return 1.0e100;
    }
    if (std::abs(feature.area - signature.area) > std::max(1.0e-5, std::abs(signature.area) * 0.35)) {
        return 1.0e100;
    }

    const double distanceScale = std::max(1.0, std::sqrt(std::max(signature.area, 1.0e-9)));
    const double areaScale = std::max(std::abs(signature.area), 1.0e-9);
    const double radiusScale = std::max(std::abs(signature.radius), 1.0e-9);
    const double areaTerm = std::abs(feature.area - signature.area) / areaScale;
    const double radiusTerm =
        signature.radius > 0.0 ? std::abs(feature.radius - signature.radius) / radiusScale : 0.0;
    const double edgeTerm = static_cast<double>(std::abs(feature.numEdges - signature.numEdges)) * 0.05;
    const double distanceTerm = SquaredDistance(feature, signature) / (distanceScale * distanceScale);
    return distanceTerm + areaTerm * 4.0 + radiusTerm * 2.0 + edgeTerm;
}

std::map<int, std::set<int>> MatchRivetFacesAfterReload(
    const std::vector<RivetFaceSignature>& signatures,
    const std::vector<FaceFeature>& reloadedFeatures
) {
    std::map<int, std::set<int>> matchedFaceIdsByInstance;
    std::set<int> usedFaceIds;

    for (const auto& signature : signatures) {
        double bestScore = 1.0e100;
        int bestFaceId = -1;
        for (const auto& feature : reloadedFeatures) {
            if (usedFaceIds.find(feature.id) != usedFaceIds.end()) {
                continue;
            }
            const double score = ComputeRivetSignatureScore(feature, signature);
            if (score < bestScore) {
                bestScore = score;
                bestFaceId = feature.id;
            }
        }

        if (bestFaceId > 0 && bestScore < 0.75) {
            usedFaceIds.insert(bestFaceId);
            matchedFaceIdsByInstance[signature.instanceId].insert(bestFaceId);
        }
    }

    return matchedFaceIdsByInstance;
}

// The signature pass can miss top cap faces after boolean fusion. Placement
// geometry gives a second, explicit way to recover those rivet faces.
bool IsRivetFaceByPlacementGeometry(
    const FaceFeature& feature,
    const RivetPlacement& placement
) {
    const double radius = std::max(placement.radius, 1.0e-9);
    const double height = placement.height;
    const gp_Vec axis(placement.normal);
    const gp_Vec centerVector(
        placement.basePoint,
        gp_Pnt(feature.centerX, feature.centerY, feature.centerZ)
    );
    const double axial = centerVector.Dot(axis);
    const double squaredDistance = centerVector.SquareMagnitude();
    const double radialSquared = std::max(0.0, squaredDistance - axial * axial);
    const double radial = std::sqrt(radialSquared);

    if (placement.footprint == RivetFootprint::Hexagon) {
        if (feature.surfaceType != GeomAbs_Plane) {
            return false;
        }

        const double topArea = ExpectedRivetTopArea(placement);
        const double sideArea = 6.0 * radius * std::max(height, 1.0e-9);
        const double maxExpectedArea = std::max(topArea, sideArea);
        const bool axialInRivetRange =
            axial >= -radius * 0.75 &&
            axial <= height + radius * 0.75;
        const bool radialMatches = radial <= radius * 1.75;
        const bool areaMatches =
            feature.area > 0.0 &&
            feature.area <= std::max(1.0e-4, maxExpectedArea * 3.0);
        return axialInRivetRange && radialMatches && areaMatches;
    }

    if (feature.surfaceType == GeomAbs_Cylinder) {
        const bool radiusMatches =
            feature.radius > 0.0 &&
            std::abs(feature.radius - radius) <= std::max(1.0e-4, radius * 0.35);
        const bool axialInRivetRange =
            axial >= -radius * 0.75 &&
            axial <= height + radius * 0.75;
        const bool radialMatches =
            radial >= radius * 0.35 &&
            radial <= radius * 1.65;
        return radiusMatches && axialInRivetRange && radialMatches;
    }

    if (feature.surfaceType == GeomAbs_Plane) {
        const double expectedTopArea = ExpectedRivetTopArea(placement);
        const bool nearTop =
            std::abs(axial - height) <= std::max(1.0e-4, radius * 0.35);
        const bool nearAxis = radial <= radius * 0.55;
        const bool areaMatches =
            expectedTopArea > 0.0 &&
            std::abs(feature.area - expectedTopArea) <=
                std::max(1.0e-4, expectedTopArea * 0.45);
        const double normalDot =
            std::abs(feature.normalX * placement.normal.X() +
                     feature.normalY * placement.normal.Y() +
                     feature.normalZ * placement.normal.Z());
        return nearTop && nearAxis && areaMatches && normalDot >= 0.85;
    }

    return false;
}

bool IsRivetTopNeighborFace(
    const FaceFeature& candidate,
    const RivetPlacement& placement
) {
    if (candidate.surfaceType != GeomAbs_Plane) {
        return false;
    }

    const double radius = std::max(placement.radius, 1.0e-9);
    const double expectedTopArea = ExpectedRivetTopArea(placement);
    const bool areaMatches =
        std::abs(candidate.area - expectedTopArea) <=
        std::max(1.0e-4, expectedTopArea * 0.55);
    const double maxCompactness =
        placement.footprint == RivetFootprint::Hexagon ? 1.45 : 1.35;
    const bool compactEnough =
        candidate.compactness > 0.0 && candidate.compactness <= maxCompactness;
    return areaMatches && compactEnough;
}

bool IsRivetTopSplitFace(const FaceFeature& candidate) {
    if (candidate.surfaceType != GeomAbs_Plane) {
        return false;
    }

    const bool tinyCapPatch =
        candidate.area > 0.0 &&
        candidate.area <= 1.0e-3 &&
        candidate.relativeArea > 0.0 &&
        candidate.relativeArea <= 1.0e-5;
    const bool capLike =
        candidate.compactness > 0.0 &&
        candidate.compactness <= 1.60 &&
        candidate.numEdges >= 4 &&
        candidate.numEdges <= 8;
    return tinyCapPatch && capLike;
}

void AddNeighborTopFacesForMatchedRivet(
    const RivetPlacement& placement,
    const std::vector<FaceFeature>& reloadedFeatures,
    std::map<int, std::set<int>>& matchedFaceIdsByInstance,
    std::set<int>& usedFaceIds
) {
    std::map<int, const FaceFeature*> featureById;
    for (const auto& feature : reloadedFeatures) {
        featureById[feature.id] = &feature;
    }

    const auto existingFaceIds = matchedFaceIdsByInstance[placement.instanceId];
    if (placement.footprint != RivetFootprint::Round) {
        return;
    }

    for (const int faceId : existingFaceIds) {
        const auto faceIt = featureById.find(faceId);
        if (faceIt == featureById.end() ||
            faceIt->second->surfaceType != GeomAbs_Cylinder) {
            continue;
        }

        for (const int neighborId : faceIt->second->neighborIds) {
            const auto neighborIt = featureById.find(neighborId);
            if (neighborIt == featureById.end() ||
                usedFaceIds.find(neighborId) != usedFaceIds.end()) {
                continue;
            }
            if (!IsRivetTopNeighborFace(*neighborIt->second, placement)) {
                continue;
            }

            matchedFaceIdsByInstance[placement.instanceId].insert(neighborId);
            usedFaceIds.insert(neighborId);
        }
    }
}

void AddEnclosedSplitTopFacesForMatchedRivet(
    const RivetPlacement& placement,
    const std::vector<FaceFeature>& reloadedFeatures,
    std::map<int, std::set<int>>& matchedFaceIdsByInstance,
    std::set<int>& usedFaceIds
) {
    std::map<int, const FaceFeature*> featureById;
    for (const auto& feature : reloadedFeatures) {
        featureById[feature.id] = &feature;
    }

    bool addedAny = true;
    while (addedAny) {
        addedAny = false;
        const auto currentFaceIds = matchedFaceIdsByInstance[placement.instanceId];
        for (const auto& feature : reloadedFeatures) {
            if (currentFaceIds.find(feature.id) != currentFaceIds.end() ||
                usedFaceIds.find(feature.id) != usedFaceIds.end()) {
                continue;
            }
            if (!IsRivetTopSplitFace(feature) || feature.neighborIds.empty()) {
                continue;
            }

            bool enclosedByInstance = true;
            for (const int neighborId : feature.neighborIds) {
                if (currentFaceIds.find(neighborId) == currentFaceIds.end()) {
                    enclosedByInstance = false;
                    break;
                }
            }
            if (!enclosedByInstance) {
                continue;
            }

            matchedFaceIdsByInstance[placement.instanceId].insert(feature.id);
            usedFaceIds.insert(feature.id);
            addedAny = true;
        }
    }
}

void AugmentRivetFaceMatchesFromPlacements(
    const std::vector<RivetPlacement>& placements,
    const std::vector<FaceFeature>& reloadedFeatures,
    std::map<int, std::set<int>>& matchedFaceIdsByInstance
) {
    std::set<int> usedFaceIds;
    for (const auto& [instanceId, faceIds] : matchedFaceIdsByInstance) {
        usedFaceIds.insert(faceIds.begin(), faceIds.end());
    }

    for (const auto& placement : placements) {
        for (const auto& feature : reloadedFeatures) {
            const bool alreadyOwnedByThisInstance =
                matchedFaceIdsByInstance[placement.instanceId].find(feature.id) !=
                matchedFaceIdsByInstance[placement.instanceId].end();
            if (!alreadyOwnedByThisInstance && usedFaceIds.find(feature.id) != usedFaceIds.end()) {
                continue;
            }
            if (!IsRivetFaceByPlacementGeometry(feature, placement)) {
                continue;
            }

            matchedFaceIdsByInstance[placement.instanceId].insert(feature.id);
            usedFaceIds.insert(feature.id);
        }

        AddNeighborTopFacesForMatchedRivet(
            placement,
            reloadedFeatures,
            matchedFaceIdsByInstance,
            usedFaceIds
        );
        AddEnclosedSplitTopFacesForMatchedRivet(
            placement,
            reloadedFeatures,
            matchedFaceIdsByInstance,
            usedFaceIds
        );
    }
}

DatasetValidationResult ValidateWingRivetSample(const fs::path& labelsPath, const fs::path& stepPath) {
    DatasetValidationResult result;
    result.modelName = labelsPath.stem().string();
    result.labelsPath = labelsPath.string();
    result.stepPath = stepPath.string();
    result.stepExists = fs::exists(stepPath);

    LabelsData labels;
    result.labelsParsed = ParseLabelsJson(labelsPath, labels);
    result.labelFaceCount = static_cast<int>(labels.faces.size());
    result.instanceCount = static_cast<int>(labels.instances.size());
    for (const auto& face : labels.faces) {
        if (face.semantic == "rivet") {
            result.rivetFaceCount++;
        } else if (face.semantic == "background") {
            result.backgroundFaceCount++;
        }
    }

    if (!result.labelsParsed) {
        result.message = "failed_to_parse_labels";
        return result;
    }
    if (!result.stepExists) {
        result.message = "missing_output_step";
        return result;
    }

    TopoDS_Shape shape;
    result.shapeLoaded = LoadShapeFromStep(stepPath.string(), shape);
    if (!result.shapeLoaded) {
        result.message = "failed_to_load_step";
        return result;
    }

    result.shapeValid = IsShapeValid(shape);
    const auto firstFeatures = ExtractFeatures(shape);
    const auto secondFeatures = ExtractFeatures(shape);
    result.faceCount = static_cast<int>(firstFeatures.size());
    result.duplicateFaceKeyCount =
        std::max(CountDuplicateFaceKeys(firstFeatures), CountDuplicateFaceKeys(secondFeatures));
    result.graphMismatchCount = CountGraphMismatches(firstFeatures, secondFeatures);

    const bool faceIdsValid = ValidateLabelFaceIds(labels, result.faceCount);
    const bool faceOrderStable = HaveStableFaceOrder(firstFeatures, secondFeatures);
    const bool rivetLabelsValid = ValidateRivetLabels(labels);

    result.t1 =
        result.shapeLoaded &&
        faceIdsValid &&
        faceOrderStable &&
        result.duplicateFaceKeyCount == 0;
    result.t2 =
        result.shapeValid &&
        result.rivetFaceCount > 0 &&
        result.instanceCount > 0 &&
        rivetLabelsValid;
    result.t3 =
        result.graphMismatchCount == 0 &&
        faceOrderStable;

    if (result.t1 && result.t2 && result.t3) {
        result.message = "ok";
    } else if (!faceIdsValid) {
        result.message = "label_face_id_mismatch";
    } else if (!faceOrderStable) {
        result.message = "face_order_unstable";
    } else if (result.duplicateFaceKeyCount > 0) {
        result.message = "duplicate_face_keys";
    } else if (!result.shapeValid) {
        result.message = "invalid_brep";
    } else if (!rivetLabelsValid) {
        result.message = "invalid_rivet_labels";
    } else if (result.graphMismatchCount > 0) {
        result.message = "graph_unstable";
    } else {
        result.message = "unknown_failure";
    }

    return result;
}

// Wing host-face selection. These filters avoid placing rivets on fuselage,
// engine, wheel, cap, or other mechanical detail surfaces.
bool IsRevolvedMechanicalSurface(const FaceFeature& feature) {
    return feature.surfaceType == GeomAbs_Cylinder ||
           feature.surfaceType == GeomAbs_Cone ||
           feature.surfaceType == GeomAbs_Sphere ||
           feature.surfaceType == GeomAbs_Torus;
}

bool IsRoundPlanarCapFace(const FaceFeature& feature) {
    if (feature.surfaceType != GeomAbs_Plane) {
        return false;
    }

    const bool isRoundLike = feature.compactness > 0.0 && feature.compactness <= 1.8;
    const bool hasMechanicalDetail = feature.innerWireCount > 0 || feature.numEdges >= 6;
    const bool isLocalPartFace = feature.area <= 200.0;
    return isRoundLike && hasMechanicalDetail && isLocalPartFace;
}

bool IsCurvedOuterMechanicalSkin(const FaceFeature& feature) {
    if (feature.surfaceType != GeomAbs_BSplineSurface) {
        return false;
    }

    return std::abs(feature.meanCurvature) >= 0.08;
}

bool ShouldRejectAsWingHost(const FaceFeature& feature) {
    if (IsRevolvedMechanicalSurface(feature)) {
        return true;
    }

    if (feature.radius > 1.0e-6) {
        return true;
    }

    if (IsRoundPlanarCapFace(feature)) {
        return true;
    }

    if (IsCurvedOuterMechanicalSkin(feature)) {
        return true;
    }

    return false;
}

bool IsPrimaryWingSkinCandidate(const FaceFeature& feature) {
    if (ShouldRejectAsWingHost(feature)) {
        return false;
    }

    const bool isLargeEnough = feature.area >= 4.0;
    const bool isAwayFromCenterline = std::abs(feature.centerZ) >= 1.5;
    const bool isWingSkinSurface =
        feature.surfaceType == GeomAbs_Plane ||
        feature.surfaceType == GeomAbs_BSplineSurface;
    const bool isWingLikeOrientation =
        feature.normalY >= 0.85 &&
        std::abs(feature.normalZ) <= 0.15;
    return isLargeEnough && isAwayFromCenterline && isWingSkinSurface && isWingLikeOrientation;
}

bool IsFallbackWingCandidate(const FaceFeature& feature) {
    if (ShouldRejectAsWingHost(feature)) {
        return false;
    }

    const bool isLargeEnough = feature.area >= 2.0;
    const bool isAwayFromCenterline = std::abs(feature.centerZ) >= 1.5;
    const bool isOuterSurface =
        feature.surfaceType == GeomAbs_BSplineSurface ||
        feature.surfaceType == GeomAbs_Plane;
    const bool isWingLikeOrientation =
        std::abs(feature.normalY) >= 0.55 &&
        std::abs(feature.normalZ) <= 0.75;
    return isLargeEnough && isAwayFromCenterline && isOuterSurface && isWingLikeOrientation;
}

double ComputeWingScore(const FaceFeature& feature) {
    const double verticalPreference = 20.0 + feature.centerY;
    const double surfacePreference =
        feature.surfaceType == GeomAbs_BSplineSurface ? 5.0 :
        feature.surfaceType == GeomAbs_Plane ? 0.2 :
        1.0;
    return feature.area *
           (1.0 + std::abs(feature.centerZ)) *
           std::max(verticalPreference, 0.1) *
           surfacePreference;
}

std::vector<WingHostFace> SelectWingHostFaces(
    const std::vector<FaceFeature>& features,
    bool usePrimaryCandidates
) {
    WingHostFace visiblePositiveWing;
    WingHostFace visibleNegativeWing;
    WingHostFace fallbackPositiveWing;
    WingHostFace fallbackNegativeWing;
    for (const auto& feature : features) {
        const bool isCandidate = usePrimaryCandidates
            ? IsPrimaryWingSkinCandidate(feature)
            : IsFallbackWingCandidate(feature);
        if (!isCandidate) {
            continue;
        }

        const double score = ComputeWingScore(feature);
        WingHostFace* fallbackBucket =
            feature.centerZ >= 0.0 ? &fallbackPositiveWing : &fallbackNegativeWing;
        if (score > fallbackBucket->score) {
            fallbackBucket->faceId = feature.id;
            fallbackBucket->score = score;
            fallbackBucket->feature = feature;
        }

        if (feature.normalY <= 0.0) {
            continue;
        }

        WingHostFace* visibleBucket =
            feature.centerZ >= 0.0 ? &visiblePositiveWing : &visibleNegativeWing;
        if (score > visibleBucket->score) {
            visibleBucket->faceId = feature.id;
            visibleBucket->score = score;
            visibleBucket->feature = feature;
        }
    }

    std::vector<WingHostFace> hostFaces;
    if (visiblePositiveWing.faceId > 0) {
        hostFaces.push_back(visiblePositiveWing);
    }
    if (visibleNegativeWing.faceId > 0) {
        hostFaces.push_back(visibleNegativeWing);
    }
    if (!hostFaces.empty()) {
        return hostFaces;
    }

    if (fallbackPositiveWing.faceId > 0) {
        hostFaces.push_back(fallbackPositiveWing);
    }
    if (fallbackNegativeWing.faceId > 0) {
        hostFaces.push_back(fallbackNegativeWing);
    }
    return hostFaces;
}

// Shape ownership and assembly replacement helpers.
TopoDS_Face GetFaceById(const TopoDS_Shape& shape, int faceId) {
    TopTools_IndexedMapOfShape faceMap;
    TopExp::MapShapes(shape, TopAbs_FACE, faceMap);
    if (faceId <= 0 || faceId > faceMap.Extent()) {
        return TopoDS_Face();
    }
    return TopoDS::Face(faceMap.FindKey(faceId));
}

int FindOwningSolidIndex(const TopoDS_Shape& shape, int faceId) {
    TopTools_IndexedMapOfShape faceMap;
    TopExp::MapShapes(shape, TopAbs_FACE, faceMap);
    if (faceId <= 0 || faceId > faceMap.Extent()) {
        return -1;
    }

    TopTools_IndexedMapOfShape solidMap;
    TopExp::MapShapes(shape, TopAbs_SOLID, solidMap);
    if (solidMap.IsEmpty()) {
        return -1;
    }

    TopTools_IndexedDataMapOfShapeListOfShape faceSolidMap;
    TopExp::MapShapesAndAncestors(shape, TopAbs_FACE, TopAbs_SOLID, faceSolidMap);
    const TopoDS_Shape targetFace = faceMap.FindKey(faceId);
    if (!faceSolidMap.Contains(targetFace)) {
        return -1;
    }

    const TopTools_ListOfShape& owners = faceSolidMap.FindFromKey(targetFace);
    for (TopTools_ListIteratorOfListOfShape it(owners); it.More(); it.Next()) {
        return solidMap.FindIndex(it.Value());
    }
    return -1;
}

TopoDS_Shape GetSolidByIndex(const TopoDS_Shape& shape, int solidIndex) {
    TopTools_IndexedMapOfShape solidMap;
    TopExp::MapShapes(shape, TopAbs_SOLID, solidMap);
    if (solidIndex <= 0 || solidIndex > solidMap.Extent()) {
        return TopoDS_Shape();
    }
    return solidMap.FindKey(solidIndex);
}

bool IsDecalHostSurface(const FaceFeature& feature) {
    if (feature.area < 2.0) {
        return false;
    }
    return feature.surfaceType == GeomAbs_BSplineSurface ||
           feature.surfaceType == GeomAbs_Cylinder ||
           feature.surfaceType == GeomAbs_Plane;
}

double ComputeDecalHostScore(const FaceFeature& feature) {
    const double surfacePreference =
        feature.surfaceType == GeomAbs_BSplineSurface ? 5.0 :
        feature.surfaceType == GeomAbs_Cylinder ? 4.0 :
        1.0;
    const bool isTypicalWingSkin =
        std::abs(feature.normalY) >= 0.75 && std::abs(feature.centerZ) >= 1.5;
    const double nonWingPreference = isTypicalWingSkin ? 0.02 : 1.0;
    return feature.area * surfacePreference * nonWingPreference;
}

std::vector<WingHostFace> SelectDecalHostFaces(const TopoDS_Shape& shape) {
    const auto features = ExtractFeatures(shape);
    std::vector<WingHostFace> hostFaces;
    int smoothFaceCount = 0;
    int solidOwnedFaceCount = 0;
    for (const auto& feature : features) {
        if (!IsDecalHostSurface(feature)) {
            continue;
        }
        smoothFaceCount++;
        const int solidIndex = FindOwningSolidIndex(shape, feature.id);
        const TopoDS_Shape ownerSolid = GetSolidByIndex(shape, solidIndex);
        if (ownerSolid.IsNull() || !IsShapeValid(ownerSolid)) {
            continue;
        }
        solidOwnedFaceCount++;
        hostFaces.push_back({feature.id, ComputeDecalHostScore(feature), feature});
    }
    std::sort(hostFaces.begin(), hostFaces.end(),
              [](const WingHostFace& lhs, const WingHostFace& rhs) {
                  return lhs.score > rhs.score;
              });
    if (hostFaces.size() > 12) {
        hostFaces.resize(12);
    }
    std::cout << ">>> Decal host candidates: smooth=" << smoothFaceCount
              << " solid_owned=" << solidOwnedFaceCount << std::endl;
    return hostFaces;
}

bool BuildManualDecalHostFace(
    const TopoDS_Shape& shape,
    int faceId,
    WingHostFace& hostFace
) {
    const auto features = ExtractFeatures(shape);
    if (faceId <= 0 || faceId > static_cast<int>(features.size())) {
        std::cout << ">>> Selected host face ID is outside the model range: " << faceId << std::endl;
        return false;
    }
    const FaceFeature& feature = features[static_cast<size_t>(faceId - 1)];
    if (!IsDecalHostSurface(feature)) {
        std::cout << ">>> Selected host face is not a supported smooth surface: " << faceId << std::endl;
        return false;
    }
    const int solidIndex = FindOwningSolidIndex(shape, faceId);
    const TopoDS_Shape ownerSolid = GetSolidByIndex(shape, solidIndex);
    if (ownerSolid.IsNull() || !IsShapeValid(ownerSolid)) {
        std::cout << ">>> Selected host face is not owned by a valid solid: " << faceId << std::endl;
        return false;
    }
    hostFace = {faceId, ComputeDecalHostScore(feature), feature};
    return true;
}

InjectionTarget BuildInjectionTarget(
    const TopoDS_Shape& originalShape,
    const std::vector<WingHostFace>& hostFaces
) {
    InjectionTarget target;
    if (hostFaces.empty()) {
        target.workingShape = originalShape;
        return target;
    }

    bool canUsePerSolidMode = true;
    int sharedSolidIndex = -1;
    for (const auto& hostFace : hostFaces) {
        const int currentIndex = FindOwningSolidIndex(originalShape, hostFace.faceId);
        if (currentIndex <= 0) {
            canUsePerSolidMode = false;
            break;
        }

        target.solidIndexByHostFaceId[hostFace.faceId] = currentIndex;
        if (target.originalSolidByIndex.find(currentIndex) == target.originalSolidByIndex.end()) {
            TopoDS_Shape ownerSolid = GetSolidByIndex(originalShape, currentIndex);
            if (ownerSolid.IsNull()) {
                canUsePerSolidMode = false;
                break;
            }
            target.originalSolidByIndex[currentIndex] = ownerSolid;
            target.workingSolidByIndex[currentIndex] = ownerSolid;
        }

        if (sharedSolidIndex < 0) {
            sharedSolidIndex = currentIndex;
        } else if (sharedSolidIndex != currentIndex) {
            sharedSolidIndex = 0;
        }
    }

    if (canUsePerSolidMode && sharedSolidIndex > 0) {
        target.originalSubshape = target.originalSolidByIndex[sharedSolidIndex];
        target.workingShape = target.originalSubshape;
        target.isSubshapeMode = true;
        target.isPerSolidMode = true;
        return target;
    }

    if (canUsePerSolidMode && !target.workingSolidByIndex.empty()) {
        target.workingShape = originalShape;
        target.isPerSolidMode = true;
        return target;
    }

    target.workingShape = originalShape;
    return target;
}

TopoDS_Shape RebuildShapeWithReplacement(
    const TopoDS_Shape& originalShape,
    const TopoDS_Shape& originalSubshape,
    const TopoDS_Shape& replacementSubshape
) {
    Handle(BRepTools_ReShape) reshape = new BRepTools_ReShape();
    reshape->Replace(originalSubshape, replacementSubshape);
    return reshape->Apply(originalShape);
}

TopoDS_Shape RebuildShapeWithSolidReplacements(
    const TopoDS_Shape& originalShape,
    const std::map<int, TopoDS_Shape>& originalSolidByIndex,
    const std::map<int, TopoDS_Shape>& replacementSolidByIndex
) {
    Handle(BRepTools_ReShape) reshape = new BRepTools_ReShape();
    for (const auto& [solidIndex, replacementSolid] : replacementSolidByIndex) {
        const auto originalIt = originalSolidByIndex.find(solidIndex);
        if (originalIt == originalSolidByIndex.end() || replacementSolid.IsNull()) {
            continue;
        }
        reshape->Replace(originalIt->second, replacementSolid);
    }
    return reshape->Apply(originalShape);
}

// Rivet placement and geometry construction helpers.
gp_Pnt ComputeShapeCenter(const TopoDS_Shape& shape) {
    Bnd_Box box;
    BRepBndLib::Add(shape, box);
    if (box.IsVoid()) {
        return gp_Pnt(0.0, 0.0, 0.0);
    }

    double xMin = 0.0;
    double yMin = 0.0;
    double zMin = 0.0;
    double xMax = 0.0;
    double yMax = 0.0;
    double zMax = 0.0;
    box.Get(xMin, yMin, zMin, xMax, yMax, zMax);
    return gp_Pnt(
        (xMin + xMax) * 0.5,
        (yMin + yMax) * 0.5,
        (zMin + zMax) * 0.5
    );
}

void OrientNormalAwayFromCenter(const gp_Pnt& point, const gp_Pnt& shapeCenter, gp_Dir& normal) {
    gp_Vec centerToPoint(shapeCenter, point);
    if (centerToPoint.SquareMagnitude() <= Precision::SquareConfusion()) {
        return;
    }

    if (gp_Vec(normal).Dot(centerToPoint) < 0.0) {
        normal.Reverse();
    }
}

bool IsUvInsideFace(const TopoDS_Face& face, double u, double v) {
    BRepClass_FaceClassifier classifier;
    classifier.Perform(face, gp_Pnt2d(u, v), Precision::Confusion());
    const TopAbs_State state = classifier.State();
    return state == TopAbs_IN || state == TopAbs_ON;
}

bool EvaluateFacePointAndNormal(
    const TopoDS_Face& face,
    double u,
    double v,
    gp_Pnt& point,
    gp_Dir& normal
) {
    BRepAdaptor_Surface surface(face);
    BRepLProp_SLProps props(surface, u, v, 1, Precision::Confusion());
    if (!props.IsNormalDefined()) {
        return false;
    }

    point = surface.Value(u, v);
    normal = props.Normal();
    if (face.Orientation() == TopAbs_REVERSED) {
        normal.Reverse();
    }
    return true;
}

bool IsDuplicateUvSample(
    const std::vector<gp_Pnt2d>& samples,
    double u,
    double v,
    double tolerance
) {
    const double toleranceSquared = tolerance * tolerance;
    for (const auto& sample : samples) {
        const double du = sample.X() - u;
        const double dv = sample.Y() - v;
        if (du * du + dv * dv <= toleranceSquared) {
            return true;
        }
    }
    return false;
}

const char* RivetFootprintName(RivetFootprint footprint) {
    return footprint == RivetFootprint::Hexagon ? "hexagon" : "round";
}

RivetFootprint SelectRivetFootprint(int instanceId, int startingInstanceId) {
    const int ordinal = instanceId - startingInstanceId;
    return ordinal % 3 == 2 ? RivetFootprint::Hexagon : RivetFootprint::Round;
}

double ExpectedRivetTopArea(const RivetPlacement& placement) {
    const double radius = std::max(placement.radius, 1.0e-9);
    if (placement.footprint == RivetFootprint::Hexagon) {
        return 3.0 * std::sqrt(3.0) * radius * radius / 2.0;
    }
    return M_PI * radius * radius;
}

std::vector<gp_Pnt2d> BuildTrimInsetUvSamples(
    const TopoDS_Face& hostFace,
    double uMin,
    double uMax,
    double vMin,
    double vMax,
    int maxSamples
) {
    const double uRange = std::max(uMax - uMin, Precision::Confusion());
    const double vRange = std::max(vMax - vMin, Precision::Confusion());
    const gp_Pnt2d uvCenter((uMin + uMax) * 0.5, (vMin + vMax) * 0.5);
    const double uvScale = std::max(uRange, vRange);
    const double duplicateTolerance = uvScale * 1.0e-3;
    const std::vector<double> insetDistances = {
        uvScale * 0.080,
        uvScale * 0.120,
        uvScale * 0.180,
    };

    std::vector<gp_Pnt2d> candidates;
    for (TopExp_Explorer edgeExplorer(hostFace, TopAbs_EDGE);
         edgeExplorer.More();
         edgeExplorer.Next()) {
        const TopoDS_Edge edge = TopoDS::Edge(edgeExplorer.Current());
        Standard_Real first = 0.0;
        Standard_Real last = 0.0;
        Handle(Geom2d_Curve) curve = BRep_Tool::CurveOnSurface(edge, hostFace, first, last);
        if (curve.IsNull() || !std::isfinite(first) || !std::isfinite(last) ||
            std::abs(last - first) <= Precision::PConfusion()) {
            continue;
        }

        const int perEdgeSamples = 4;
        for (int i = 1; i <= perEdgeSamples; ++i) {
            const double t = first + (last - first) * (static_cast<double>(i) / (perEdgeSamples + 1));
            const gp_Pnt2d boundaryUv = curve->Value(t);
            const double dirU = uvCenter.X() - boundaryUv.X();
            const double dirV = uvCenter.Y() - boundaryUv.Y();
            const double dirLength = std::hypot(dirU, dirV);
            if (dirLength <= Precision::PConfusion()) {
                continue;
            }

            for (const double inset : insetDistances) {
                const double u = boundaryUv.X() + dirU / dirLength * inset;
                const double v = boundaryUv.Y() + dirV / dirLength * inset;
                if (u < uMin || u > uMax || v < vMin || v > vMax) {
                    continue;
                }
                if (!IsUvInsideFace(hostFace, u, v)) {
                    continue;
                }
                if (IsDuplicateUvSample(candidates, u, v, duplicateTolerance)) {
                    continue;
                }

                candidates.emplace_back(u, v);
                break;
            }
        }
    }

    if (static_cast<int>(candidates.size()) <= maxSamples) {
        return candidates;
    }

    std::vector<gp_Pnt2d> selected;
    selected.reserve(maxSamples);
    for (int i = 0; i < maxSamples; ++i) {
        const size_t index =
            static_cast<size_t>((static_cast<double>(i) + 0.5) * candidates.size() / maxSamples);
        selected.push_back(candidates[std::min(index, candidates.size() - 1)]);
    }
    return selected;
}

void AddPlacementIfValid(
    int hostFaceId,
    const TopoDS_Face& hostFace,
    const gp_Pnt& shapeCenter,
    double rivetRadius,
    double rivetHeight,
    RivetFootprint footprint,
    double u,
    double v,
    int& instanceId,
    std::vector<RivetPlacement>& placements
) {
    if (!IsUvInsideFace(hostFace, u, v)) {
        return;
    }

    gp_Pnt point;
    gp_Dir normal;
    if (!EvaluateFacePointAndNormal(hostFace, u, v, point, normal)) {
        return;
    }
    OrientNormalAwayFromCenter(point, shapeCenter, normal);

    placements.push_back({
        instanceId++,
        hostFaceId,
        u,
        v,
        rivetRadius,
        rivetHeight,
        footprint,
        point,
        normal,
    });
}

std::vector<RivetPlacement> BuildWingRivetPlacements(
    int hostFaceId,
    const TopoDS_Face& hostFace,
    const FaceFeature& hostFeature,
    const gp_Pnt& shapeCenter,
    int startingInstanceId
) {
    double uMin = 0.0;
    double uMax = 0.0;
    double vMin = 0.0;
    double vMax = 0.0;
    BRepTools::UVBounds(hostFace, uMin, uMax, vMin, vMax);

    const std::vector<double> uFractions = {0.22, 0.38, 0.54, 0.70};
    const std::vector<double> vFractions = {0.38, 0.58};
    const double faceScale = std::sqrt(std::max(hostFeature.area, 1.0));
    const double rivetRadius = std::clamp(faceScale * 0.0012, 0.003, 0.16);
    const double rivetHeight = std::max(rivetRadius * 0.65, 0.002);

    std::vector<RivetPlacement> placements;
    int instanceId = startingInstanceId;
    const std::vector<gp_Pnt2d> trimInsetSamples =
        BuildTrimInsetUvSamples(hostFace, uMin, uMax, vMin, vMax, 8);
    for (const auto& sample : trimInsetSamples) {
        const RivetFootprint footprint = SelectRivetFootprint(instanceId, startingInstanceId);
        AddPlacementIfValid(
            hostFaceId,
            hostFace,
            shapeCenter,
            rivetRadius,
            rivetHeight,
            footprint,
            sample.X(),
            sample.Y(),
            instanceId,
            placements
        );
    }
    if (!placements.empty()) {
        return placements;
    }

    for (const double vFraction : vFractions) {
        for (const double uFraction : uFractions) {
            const double u = uMin + (uMax - uMin) * uFraction;
            const double v = vMin + (vMax - vMin) * vFraction;
            const RivetFootprint footprint = SelectRivetFootprint(instanceId, startingInstanceId);
            AddPlacementIfValid(
                hostFaceId,
                hostFace,
                shapeCenter,
                rivetRadius,
                rivetHeight,
                footprint,
                u,
                v,
                instanceId,
                placements
            );
        }
    }

    return placements;
}

bool BuildStarDecalWire(
    const TopoDS_Face& hostFace,
    double centerU,
    double centerV,
    double outerRadius,
    TopoDS_Wire& starWire
) {
    constexpr double kPi = 3.14159265358979323846;
    BRepBuilderAPI_MakePolygon polygon;
    for (int vertexIndex = 0; vertexIndex < 10; ++vertexIndex) {
        const double radius = vertexIndex % 2 == 0 ? outerRadius : outerRadius * 0.42;
        const double angle = kPi * 0.5 + vertexIndex * kPi / 5.0;
        const double u = centerU + std::cos(angle) * radius;
        const double v = centerV + std::sin(angle) * radius;
        if (!IsUvInsideFace(hostFace, u, v)) {
            return false;
        }

        gp_Pnt point;
        gp_Dir normal;
        if (!EvaluateFacePointAndNormal(hostFace, u, v, point, normal)) {
            return false;
        }
        polygon.Add(point);
    }
    polygon.Close();
    if (!polygon.IsDone()) {
        return false;
    }
    starWire = polygon.Wire();
    return !starWire.IsNull();
}

bool SplitFaceWithStarDecal(
    const TopoDS_Shape& shape,
    const TopoDS_Face& hostFace,
    TopoDS_Shape& result,
    TopoDS_Shape& decalFace
) {
    double uMin = 0.0;
    double uMax = 0.0;
    double vMin = 0.0;
    double vMax = 0.0;
    BRepTools::UVBounds(hostFace, uMin, uMax, vMin, vMax);
    const double uvScale = std::min(uMax - uMin, vMax - vMin);
    if (uvScale <= Precision::PConfusion()) {
        return false;
    }

    const std::vector<gp_Pnt2d> centers = {
        {uMin + (uMax - uMin) * 0.50, vMin + (vMax - vMin) * 0.50},
        {uMin + (uMax - uMin) * 0.64, vMin + (vMax - vMin) * 0.40},
        {uMin + (uMax - uMin) * 0.46, vMin + (vMax - vMin) * 0.58},
        {uMin + (uMax - uMin) * 0.30, vMin + (vMax - vMin) * 0.42},
    };
    for (const double radiusScale : {0.440, 0.350, 0.300, 0.220, 0.160, 0.120, 0.085, 0.055, 0.025, 0.010}) {
        for (const auto& center : centers) {
            TopoDS_Wire starWire;
            if (!BuildStarDecalWire(hostFace, center.X(), center.Y(), uvScale * radiusScale, starWire)) {
                continue;
            }

            BRepFeat_SplitShape split(shape);
            split.Add(starWire, hostFace);
            split.Build();
            if (!split.IsDone() || split.Shape().IsNull() || !IsShapeValid(split.Shape())) {
                continue;
            }
            result = split.Shape();
            const auto splitFeatures = ExtractFeatures(result);
            TopTools_IndexedMapOfShape faceMap;
            TopExp::MapShapes(result, TopAbs_FACE, faceMap);
            double smallestArea = std::numeric_limits<double>::infinity();
            for (TopTools_ListIteratorOfListOfShape it(split.Modified(hostFace)); it.More(); it.Next()) {
                if (it.Value().ShapeType() != TopAbs_FACE) {
                    continue;
                }
                const int faceId = faceMap.FindIndex(it.Value());
                if (faceId <= 0 || faceId > static_cast<int>(splitFeatures.size())) {
                    continue;
                }
                const double area = splitFeatures[static_cast<size_t>(faceId - 1)].area;
                if (area < smallestArea) {
                    smallestArea = area;
                    decalFace = it.Value();
                }
            }
            if (decalFace.IsNull()) {
                continue;
            }
            std::cout << ">>> Star decal radius scale: " << radiusScale << std::endl;
            return true;
        }
    }
    return false;
}

bool AddStarDecalToFuselageOrTail(
    TopoDS_Shape& shape,
    int startingInstanceId,
    std::vector<std::pair<int, std::vector<TopoDS_Shape>>>& decalFacesByInstance,
    std::vector<LabelsInstanceEntry>& labelInstances,
    int manualHostFaceId
) {
    std::vector<WingHostFace> hostFaces;
    if (manualHostFaceId > 0) {
        WingHostFace manualHost;
        if (!BuildManualDecalHostFace(shape, manualHostFaceId, manualHost)) {
            return false;
        }
        hostFaces.push_back(manualHost);
        std::cout << ">>> Using manually selected star decal host face: " << manualHostFaceId << std::endl;
    } else {
        hostFaces = SelectDecalHostFaces(shape);
    }
    for (const auto& hostFaceInfo : hostFaces) {
        const TopoDS_Face hostFace = GetFaceById(shape, hostFaceInfo.faceId);
        if (hostFace.IsNull()) {
            continue;
        }

        TopoDS_Shape splitShape;
        TopoDS_Shape decalFace;
        if (!SplitFaceWithStarDecal(shape, hostFace, splitShape, decalFace)) {
            std::cout << ">>> Could not place a star decal on candidate face "
                      << hostFaceInfo.faceId << std::endl;
            continue;
        }
        shape = splitShape;
        decalFacesByInstance.push_back({startingInstanceId, {decalFace}});
        labelInstances.push_back({
            startingInstanceId,
            "decal",
            hostFaceInfo.faceId,
            1.0,
            0.0,
        });
        std::cout << ">>> Star decal host face: " << hostFaceInfo.faceId
                  << " (fuselage/tail candidate)" << std::endl;
        return true;
    }
    return false;
}

TopoDS_Shape MakeRoundRivetSolid(const RivetPlacement& placement) {
    const double embedDepth = std::max(placement.radius * 0.35, 0.01);
    gp_Vec normalVec(placement.normal);
    const gp_Pnt axisOrigin = placement.basePoint.Translated(-normalVec * embedDepth);
    gp_Ax2 axis(axisOrigin, placement.normal);
    return BRepPrimAPI_MakeCylinder(axis, placement.radius, placement.height + embedDepth).Shape();
}

TopoDS_Shape MakeHexagonRivetSolid(const RivetPlacement& placement) {
    const double embedDepth = std::max(placement.radius * 0.35, 0.01);
    const double prismHeight = placement.height + embedDepth;
    const gp_Vec normalVec(placement.normal);
    const gp_Pnt baseCenter = placement.basePoint.Translated(-normalVec * embedDepth);

    gp_Vec reference(1.0, 0.0, 0.0);
    gp_Vec tangentU = normalVec.Crossed(reference);
    if (tangentU.SquareMagnitude() <= Precision::SquareConfusion()) {
        reference = gp_Vec(0.0, 1.0, 0.0);
        tangentU = normalVec.Crossed(reference);
    }
    tangentU.Normalize();
    gp_Vec tangentV = normalVec.Crossed(tangentU);
    tangentV.Normalize();

    BRepBuilderAPI_MakePolygon polygon;
    constexpr double kPi = 3.14159265358979323846;
    for (int vertexIndex = 0; vertexIndex < 6; ++vertexIndex) {
        const double angle = kPi / 6.0 + vertexIndex * 2.0 * kPi / 6.0;
        gp_Vec offset =
            tangentU * (std::cos(angle) * placement.radius) +
            tangentV * (std::sin(angle) * placement.radius);
        polygon.Add(baseCenter.Translated(offset));
    }
    polygon.Close();

    TopoDS_Face baseFace = BRepBuilderAPI_MakeFace(polygon.Wire()).Face();
    return BRepPrimAPI_MakePrism(baseFace, normalVec * prismHeight).Shape();
}

TopoDS_Shape MakeRivetSolid(const RivetPlacement& placement) {
    if (placement.footprint == RivetFootprint::Hexagon) {
        return MakeHexagonRivetSolid(placement);
    }
    return MakeRoundRivetSolid(placement);
}

std::vector<TopoDS_Shape> CollectGeneratedRivetFaces(
    BRepAlgoAPI_Fuse& fuse,
    const TopoDS_Shape& rivetSolid
) {
    std::vector<TopoDS_Shape> generatedFaces;
    TopExp_Explorer faceExplorer(rivetSolid, TopAbs_FACE);
    for (; faceExplorer.More(); faceExplorer.Next()) {
        const TopoDS_Shape& rivetFace = faceExplorer.Current();
        const TopTools_ListOfShape& modified = fuse.Modified(rivetFace);
        for (TopTools_ListIteratorOfListOfShape it(modified); it.More(); it.Next()) {
            if (it.Value().ShapeType() == TopAbs_FACE) {
                generatedFaces.push_back(it.Value());
            }
        }

        const TopTools_ListOfShape& generated = fuse.Generated(rivetFace);
        for (TopTools_ListIteratorOfListOfShape it(generated); it.More(); it.Next()) {
            if (it.Value().ShapeType() == TopAbs_FACE) {
                generatedFaces.push_back(it.Value());
            }
        }
    }
    return generatedFaces;
}

// Shape validation and output helpers.
bool IsShapeValid(const TopoDS_Shape& shape) {
    if (shape.IsNull()) {
        return false;
    }

    BRepCheck_Analyzer analyzer(shape);
    return analyzer.IsValid();
}

std::string BuildOutputStepPath(const std::string& inputFile, const fs::path& outputDir) {
    const fs::path inputPath(inputFile);
    return (outputDir / (inputPath.stem().string() + "_wing_rivets.step")).string();
}

std::string BuildOutputLabelsPath(const std::string& inputFile, const fs::path& outputDir) {
    const fs::path inputPath(inputFile);
    return (outputDir / (inputPath.stem().string() + "_wing_rivets.labels.json")).string();
}

std::string BuildRivetOnlyStepPath(const std::string& inputFile, const fs::path& outputDir) {
    const fs::path inputPath(inputFile);
    return (outputDir / (inputPath.stem().string() + "_rivet_only.step")).string();
}

fs::path BuildWingRivetStepsDir(const std::string& inputFile) {
    fs::path parentPath = fs::path(inputFile).parent_path();
    if (parentPath.filename() == "source") {
        parentPath = parentPath.parent_path();
    }
    return parentPath / "step";
}

fs::path BuildWingRivetLabelsDir(const std::string& inputFile) {
    fs::path parentPath = fs::path(inputFile).parent_path();
    if (parentPath.filename() == "source") {
        parentPath = parentPath.parent_path();
    }
    return parentPath / "label";
}

void WriteLabelsJson(
    const std::string& inputFile,
    const std::string& outputStepFile,
    const std::string& outputLabelsFile,
    const std::vector<LabelsFaceEntry>& faceEntries,
    const std::vector<LabelsInstanceEntry>& instances
) {
    std::ofstream jsonFile(outputLabelsFile);
    jsonFile << std::fixed << std::setprecision(6);
    jsonFile << "{\n";
    jsonFile << "  \"model_id\": \"" << EscapeJson(fs::path(outputStepFile).stem().string()) << "\",\n";
    jsonFile << "  \"base_model\": \"" << EscapeJson(fs::path(inputFile).filename().string()) << "\",\n";
    jsonFile << "  \"output_step\": \"" << EscapeJson(outputStepFile) << "\",\n";
    jsonFile << "  \"faces\": [\n";

    for (size_t index = 0; index < faceEntries.size(); ++index) {
        const auto& entry = faceEntries[index];
        jsonFile << "    {\"face_id\": " << entry.faceId
                 << ", \"semantic\": \"" << entry.semantic
                 << "\", \"instance_id\": " << entry.instanceId
                 << ", \"operation\": \"" << entry.operation << "\"}";
        if (index + 1 != faceEntries.size()) {
            jsonFile << ",";
        }
        jsonFile << "\n";
    }

    jsonFile << "  ],\n";
    jsonFile << "  \"instances\": [\n";
    for (size_t index = 0; index < instances.size(); ++index) {
        const auto& instance = instances[index];
        jsonFile << "    {\"instance_id\": " << instance.instanceId
                 << ", \"type\": \"" << instance.type
                 << "\", \"host_face\": " << instance.hostFaceId
                 << ", \"inverse_op\": {\"kind\": \""
                 << (instance.type == "decal" ? "remove_decal_boundary" : "remove_protrusion")
                 << "\", \"radius\": "
                 << instance.radius << ", \"height\": " << instance.height << "}}";
        if (index + 1 != instances.size()) {
            jsonFile << ",";
        }
        jsonFile << "\n";
    }
    jsonFile << "  ]\n";
    jsonFile << "}\n";
}
}

// Public command entry points.
int RunWingRivetInjectionImpl(
    const std::string& inputFile,
    const std::string& outputStepFile,
    const std::string& outputLabelsFile
) {
    std::cout << ">>> Injecting wing rivets into: " << inputFile << std::endl;

    TopoDS_Shape originalShape;
    if (!LoadShapeFromStep(inputFile, originalShape)) {
        std::cout << ">>> Failed to load STEP model." << std::endl;
        return 1;
    }

    const auto originalFeatures = ExtractFeatures(originalShape);
    if (originalFeatures.empty()) {
        std::cout << ">>> Failed to extract original face features." << std::endl;
        return 1;
    }
    const gp_Pnt shapeCenter = ComputeShapeCenter(originalShape);

    std::vector<WingHostFace> hostFaces = SelectWingHostFaces(originalFeatures, true);
    if (hostFaces.empty()) {
        hostFaces = SelectWingHostFaces(originalFeatures, false);
    }
    if (hostFaces.empty()) {
        std::cout << ">>> Could not find suitable wing faces for rivet injection." << std::endl;
        return 1;
    }

    std::vector<RivetPlacement> placements;
    int nextInstanceId = 1;
    for (const auto& hostFaceInfo : hostFaces) {
        const TopoDS_Face hostFace = GetFaceById(originalShape, hostFaceInfo.faceId);
        if (hostFace.IsNull()) {
            continue;
        }

        const auto facePlacements = BuildWingRivetPlacements(
            hostFaceInfo.faceId,
            hostFace,
            hostFaceInfo.feature,
            shapeCenter,
            nextInstanceId
        );
        nextInstanceId += static_cast<int>(facePlacements.size());
        placements.insert(placements.end(), facePlacements.begin(), facePlacements.end());
        std::cout << ">>> Wing host face " << hostFaceInfo.faceId
                  << " centerZ=" << hostFaceInfo.feature.centerZ
                  << " area=" << hostFaceInfo.feature.area
                  << " placements=" << facePlacements.size() << std::endl;
    }

    if (placements.empty()) {
        std::cout << ">>> No valid rivet placements found on the selected wing faces." << std::endl;
        return 1;
    }

    std::cout << ">>> Selected " << hostFaces.size()
              << " wing host faces with " << placements.size() << " rivet placements." << std::endl;

    const InjectionTarget injectionTarget = BuildInjectionTarget(originalShape, hostFaces);
    TopoDS_Shape currentShape = injectionTarget.workingShape;
    std::vector<std::pair<int, std::vector<TopoDS_Shape>>> rivetFacesByInstance;
    std::vector<LabelsInstanceEntry> labelInstances;
    std::vector<TopoDS_Shape> acceptedRivetShapes;
    std::vector<RivetPlacement> acceptedPlacements;

    std::cout << ">>> Injection target type=" << ShapeTypeName(currentShape.ShapeType())
              << " subshape_mode=" << (injectionTarget.isSubshapeMode ? "true" : "false")
              << " per_solid_mode=" << (injectionTarget.isPerSolidMode ? "true" : "false")
              << " valid=" << (IsShapeValid(currentShape) ? "true" : "false")
              << " solids=" << CountSubShapes(currentShape, TopAbs_SOLID)
              << " shells=" << CountSubShapes(currentShape, TopAbs_SHELL)
              << " faces=" << CountSubShapes(currentShape, TopAbs_FACE)
              << std::endl;

    std::map<int, TopoDS_Shape> workingSolidByIndex = injectionTarget.workingSolidByIndex;
    for (const auto& placement : placements) {
        TopoDS_Shape* fuseTarget = &currentShape;
        int ownerSolidIndex = -1;
        if (injectionTarget.isPerSolidMode) {
            const auto ownerIt = injectionTarget.solidIndexByHostFaceId.find(placement.hostFaceId);
            if (ownerIt == injectionTarget.solidIndexByHostFaceId.end()) {
                std::cout << ">>> Skipping rivet " << placement.instanceId
                          << " because host face has no owning solid." << std::endl;
                continue;
            }

            ownerSolidIndex = ownerIt->second;
            auto workingSolidIt = workingSolidByIndex.find(ownerSolidIndex);
            if (workingSolidIt == workingSolidByIndex.end() || workingSolidIt->second.IsNull()) {
                std::cout << ">>> Skipping rivet " << placement.instanceId
                          << " because owning solid is unavailable." << std::endl;
                continue;
            }
            fuseTarget = &workingSolidIt->second;
        }

        const TopoDS_Shape rivetSolid = MakeRivetSolid(placement);
        BRepAlgoAPI_Fuse fuse(*fuseTarget, rivetSolid);
        fuse.SetFuzzyValue(1.0e-6);
        fuse.Build();
        if (!fuse.IsDone()) {
            std::cout << ">>> Skipping rivet " << placement.instanceId
                      << " because fuse operation failed." << std::endl;
            std::cout << ">>>   host_face=" << placement.hostFaceId
                      << " shape=" << RivetFootprintName(placement.footprint)
                      << " radius=" << placement.radius
                      << " height=" << placement.height
                      << " base=(" << placement.basePoint.X()
                      << ", " << placement.basePoint.Y()
                      << ", " << placement.basePoint.Z() << ")"
                      << " normal=(" << placement.normal.X()
                      << ", " << placement.normal.Y()
                      << ", " << placement.normal.Z() << ")"
                      << std::endl;
            std::cout << ">>>   current_shape type=" << ShapeTypeName(fuseTarget->ShapeType())
                      << " valid=" << (IsShapeValid(*fuseTarget) ? "true" : "false")
                      << " solids=" << CountSubShapes(*fuseTarget, TopAbs_SOLID)
                      << " shells=" << CountSubShapes(*fuseTarget, TopAbs_SHELL)
                      << " faces=" << CountSubShapes(*fuseTarget, TopAbs_FACE)
                      << " owner_solid=" << ownerSolidIndex
                      << std::endl;
            std::cout << ">>>   rivet_shape type=" << ShapeTypeName(rivetSolid.ShapeType())
                      << " valid=" << (IsShapeValid(rivetSolid) ? "true" : "false")
                      << " solids=" << CountSubShapes(rivetSolid, TopAbs_SOLID)
                      << " shells=" << CountSubShapes(rivetSolid, TopAbs_SHELL)
                      << " faces=" << CountSubShapes(rivetSolid, TopAbs_FACE)
                      << std::endl;
            if (fuse.HasErrors()) {
                std::ostringstream errorStream;
                fuse.DumpErrors(errorStream);
                std::cout << ">>>   OCC fuse errors:\n" << errorStream.str();
            }
            if (fuse.HasWarnings()) {
                std::ostringstream warningStream;
                fuse.DumpWarnings(warningStream);
                std::cout << ">>>   OCC fuse warnings:\n" << warningStream.str();
            }
            continue;
        }

        const TopoDS_Shape candidateShape = fuse.Shape();
        if (!IsShapeValid(candidateShape)) {
            std::cout << ">>> Skipping rivet " << placement.instanceId
                      << " shape=" << RivetFootprintName(placement.footprint)
                      << " because fused model became invalid." << std::endl;
            continue;
        }

        if (injectionTarget.isPerSolidMode) {
            workingSolidByIndex[ownerSolidIndex] = candidateShape;
        } else {
            currentShape = candidateShape;
        }
        acceptedRivetShapes.push_back(rivetSolid);
        acceptedPlacements.push_back(placement);
        rivetFacesByInstance.push_back({placement.instanceId, CollectGeneratedRivetFaces(fuse, rivetSolid)});
        labelInstances.push_back({
            placement.instanceId,
            "rivet",
            placement.hostFaceId,
            placement.radius,
            placement.height,
        });
    }

    if (injectionTarget.isPerSolidMode) {
        currentShape = RebuildShapeWithSolidReplacements(
            originalShape,
            injectionTarget.originalSolidByIndex,
            workingSolidByIndex
        );
        if (!IsShapeValid(currentShape)) {
            std::cout << ">>> Failed to rebuild final assembly after per-solid rivet fusion."
                      << std::endl;
            return 1;
        }
    }

    if (labelInstances.empty()) {
        std::cout << ">>> No rivets were successfully fused into the model." << std::endl;
        return 1;
    }

    TopoDS_Shape finalOutputShape = currentShape;
    if (injectionTarget.isSubshapeMode && !injectionTarget.isPerSolidMode) {
        finalOutputShape = RebuildShapeWithReplacement(
            originalShape,
            injectionTarget.originalSubshape,
            currentShape
        );
        if (!IsShapeValid(finalOutputShape)) {
            std::cout << ">>> Failed to rebuild final assembly after rivet fusion." << std::endl;
            return 1;
        }
    }

    auto rivetSignatures = BuildRivetFaceSignatures(finalOutputShape, rivetFacesByInstance);
    if (rivetSignatures.empty()) {
        std::cout << ">>> Failed to collect rivet face signatures before STEP export." << std::endl;
        return 1;
    }

    if (!SaveShapeToStep(finalOutputShape, outputStepFile)) {
        std::cout << ">>> Failed to export modified STEP model." << std::endl;
        return 1;
    }

    if (!acceptedRivetShapes.empty()) {
        TopoDS_Compound rivetCompound;
        BRep_Builder builder;
        builder.MakeCompound(rivetCompound);
        for (const auto& rivetShape : acceptedRivetShapes) {
            if (!rivetShape.IsNull()) {
                builder.Add(rivetCompound, rivetShape);
            }
        }

        const std::string rivetOnlyStepFile =
            BuildRivetOnlyStepPath(inputFile, fs::path(outputStepFile).parent_path());
        if (!SaveShapeToStep(rivetCompound, rivetOnlyStepFile)) {
            std::cout << ">>> Failed to export rivet-only STEP model." << std::endl;
            return 1;
        }
        std::cout << ">>> Output rivet-only STEP: " << rivetOnlyStepFile << std::endl;
    }

    TopoDS_Shape reloadedOutputShape;
    if (!LoadShapeFromStep(outputStepFile, reloadedOutputShape)) {
        std::cout << ">>> Failed to reload exported STEP model for label alignment." << std::endl;
        return 1;
    }

    const auto finalFeatures = ExtractFeatures(reloadedOutputShape);
    auto matchedRivetFaceIdsByInstance =
        MatchRivetFacesAfterReload(rivetSignatures, finalFeatures);
    AugmentRivetFaceMatchesFromPlacements(
        acceptedPlacements,
        finalFeatures,
        matchedRivetFaceIdsByInstance
    );

    std::vector<LabelsFaceEntry> faceEntries;
    faceEntries.reserve(finalFeatures.size());
    for (const auto& feature : finalFeatures) {
        faceEntries.push_back({feature.id, "background", -1, "keep"});
    }

    for (const auto& [instanceId, finalFaceIds] : matchedRivetFaceIdsByInstance) {
        for (const int finalFaceId : finalFaceIds) {
            auto& entry = faceEntries[static_cast<size_t>(finalFaceId - 1)];
            const auto instanceIt = std::find_if(
                labelInstances.begin(), labelInstances.end(),
                [instanceId](const LabelsInstanceEntry& instance) {
                    return instance.instanceId == instanceId;
                }
            );
            const bool isDecal = instanceIt != labelInstances.end() && instanceIt->type == "decal";
            entry.semantic = isDecal ? "decal" : "rivet";
            entry.instanceId = instanceId;
            entry.operation = isDecal ? "remove_decal_boundary" : "remove_protrusion";
        }
    }

    int matchedInstanceCount = 0;
    for (const auto& instance : labelInstances) {
        const auto it = matchedRivetFaceIdsByInstance.find(instance.instanceId);
        if (it != matchedRivetFaceIdsByInstance.end() && !it->second.empty()) {
            matchedInstanceCount++;
        }
    }
    if (matchedInstanceCount != static_cast<int>(labelInstances.size())) {
        std::cout << ">>> Failed to align every rivet instance after STEP reload. matched="
                  << matchedInstanceCount << " expected=" << labelInstances.size() << std::endl;
        return 1;
    }

    WriteLabelsJson(inputFile, outputStepFile, outputLabelsFile, faceEntries, labelInstances);

    std::cout << ">>> Wing rivet injection complete." << std::endl;
    std::cout << ">>> Output STEP: " << outputStepFile << std::endl;
    std::cout << ">>> Output labels: " << outputLabelsFile << std::endl;
    std::cout << ">>> Injected rivet count: " << labelInstances.size() << std::endl;
    int roundRivetCount = 0;
    int hexagonRivetCount = 0;
    for (const auto& placement : acceptedPlacements) {
        if (placement.footprint == RivetFootprint::Hexagon) {
            hexagonRivetCount++;
        } else {
            roundRivetCount++;
        }
    }
    std::cout << ">>> Round rivets: " << roundRivetCount
              << ", hexagon rivets: " << hexagonRivetCount << std::endl;
    return 0;
}

int RunWingRivetInjection(const std::string& inputFile) {
    const fs::path outputStepDir = BuildWingRivetStepsDir(inputFile);
    const fs::path outputLabelsDir = BuildWingRivetLabelsDir(inputFile);
    fs::create_directories(outputStepDir);
    fs::create_directories(outputLabelsDir);
    return RunWingRivetInjectionImpl(
        inputFile,
        BuildOutputStepPath(inputFile, outputStepDir),
        BuildOutputLabelsPath(inputFile, outputLabelsDir)
    );
}

int RunStarDecalInjection(const std::string& inputFile, int hostFaceId) {
    const fs::path inputPath = fs::absolute(fs::path(inputFile));
    const fs::path planeModelDir = fs::current_path() / "data" / "plane_model";
    const fs::path expectedInputDir = planeModelDir / "after_rivet";
    const fs::path outputDir = planeModelDir / "after_two";
    if (inputPath.parent_path().lexically_normal() != expectedInputDir.lexically_normal()) {
        std::cout << ">>> Star decals only accept models from: " << expectedInputDir << std::endl;
        return 1;
    }

    const fs::path inputLabelsFile = planeModelDir / "label" /
        (inputPath.stem().string() + ".labels.json");
    LabelsData inputLabels;
    if (!ParseLabelsJson(inputLabelsFile, inputLabels) || !ValidateRivetLabels(inputLabels)) {
        std::cout << ">>> Could not load valid rivet labels: " << inputLabelsFile << std::endl;
        return 1;
    }

    TopoDS_Shape inputShape;
    if (!LoadShapeFromStep(inputFile, inputShape)) {
        std::cout << ">>> Failed to load rivet-injected STEP model." << std::endl;
        return 1;
    }
    const auto inputFeatures = ExtractFeatures(inputShape);
    if (!ValidateLabelFaceIds(inputLabels, static_cast<int>(inputFeatures.size()))) {
        std::cout << ">>> Input STEP and label face IDs are inconsistent." << std::endl;
        return 1;
    }

    std::vector<std::pair<int, std::vector<TopoDS_Shape>>> rivetFacesByInstance;
    for (const auto& face : inputLabels.faces) {
        if (face.semantic == "rivet") {
            rivetFacesByInstance.push_back({face.instanceId, {GetFaceById(inputShape, face.faceId)}});
        }
    }
    const auto rivetSignatures = BuildRivetFaceSignatures(inputShape, rivetFacesByInstance);
    if (rivetSignatures.empty()) {
        std::cout << ">>> Could not recover rivet faces from input labels." << std::endl;
        return 1;
    }

    int nextInstanceId = 1;
    for (const auto& instance : inputLabels.instances) {
        nextInstanceId = std::max(nextInstanceId, instance.instanceId + 1);
    }
    TopoDS_Shape outputShape = inputShape;
    std::vector<std::pair<int, std::vector<TopoDS_Shape>>> decalFacesByInstance;
    std::vector<LabelsInstanceEntry> outputInstances = inputLabels.instances;
    if (!AddStarDecalToFuselageOrTail(
            outputShape,
            nextInstanceId,
            decalFacesByInstance,
            outputInstances,
            hostFaceId
        )) {
        std::cout << ">>> Failed to add star decals." << std::endl;
        return 1;
    }

    const auto decalSignatures = BuildRivetFaceSignatures(outputShape, decalFacesByInstance);
    if (decalSignatures.empty()) {
        std::cout << ">>> Failed to recover star decal faces before STEP export." << std::endl;
        return 1;
    }

    fs::create_directories(outputDir);
    const std::string outputStem = inputPath.stem().string() + "_decals";
    const fs::path outputStepFile = outputDir / (outputStem + ".step");
    const fs::path outputLabelsFile = outputDir / (outputStem + ".labels.json");
    if (!SaveShapeToStep(outputShape, outputStepFile.string())) {
        std::cout << ">>> Failed to export STEP with star decals." << std::endl;
        return 1;
    }

    TopoDS_Shape reloadedShape;
    if (!LoadShapeFromStep(outputStepFile.string(), reloadedShape)) {
        std::cout << ">>> Failed to reload STEP with star decals." << std::endl;
        return 1;
    }
    const auto outputFeatures = ExtractFeatures(reloadedShape);
    auto matchedFaces = MatchRivetFacesAfterReload(rivetSignatures, outputFeatures);
    const auto matchedDecalFaces = MatchRivetFacesAfterReload(decalSignatures, outputFeatures);
    for (const auto& [instanceId, faceIds] : matchedDecalFaces) {
        matchedFaces[instanceId] = faceIds;
    }

    std::vector<LabelsFaceEntry> outputFaces;
    outputFaces.reserve(outputFeatures.size());
    for (const auto& feature : outputFeatures) {
        outputFaces.push_back({feature.id, "background", -1, "keep"});
    }
    for (const auto& [instanceId, faceIds] : matchedFaces) {
        const auto instanceIt = std::find_if(
            outputInstances.begin(), outputInstances.end(),
            [instanceId](const LabelsInstanceEntry& instance) { return instance.instanceId == instanceId; }
        );
        if (instanceIt == outputInstances.end() || faceIds.empty()) {
            continue;
        }
        const bool isDecal = instanceIt->type == "decal";
        for (const int faceId : faceIds) {
            auto& face = outputFaces[static_cast<size_t>(faceId - 1)];
            face.semantic = isDecal ? "decal" : "rivet";
            face.instanceId = instanceId;
            face.operation = isDecal ? "remove_decal_boundary" : "remove_protrusion";
        }
    }
    if (!ValidateRivetLabels({outputFaces, outputInstances})) {
        std::cout << ">>> Generated decal labels are invalid." << std::endl;
        return 1;
    }
    WriteLabelsJson(inputFile, outputStepFile.string(), outputLabelsFile.string(), outputFaces, outputInstances);
    std::cout << ">>> Output STEP: " << outputStepFile << std::endl;
    std::cout << ">>> Output labels: " << outputLabelsFile << std::endl;
    return 0;
}

int RunBatchWingRivetInjection(const std::string& inputDir) {
    const fs::path inputPath(inputDir);
    if (!fs::exists(inputPath) || !fs::is_directory(inputPath)) {
        std::cout << ">>> Input directory does not exist: " << inputDir << std::endl;
        return 1;
    }

    std::cout << ">>> Batch injecting wing rivets in directory: " << inputDir << std::endl;

    fs::path outputBaseDir = inputPath;
    if (inputPath.filename() == "source") {
        outputBaseDir = inputPath.parent_path();
    }
    const fs::path outputStepDir = outputBaseDir / "step";
    const fs::path outputLabelsDir = outputBaseDir / "label";
    fs::create_directories(outputStepDir);
    fs::create_directories(outputLabelsDir);

    int totalFiles = 0;
    int successCount = 0;
    int failureCount = 0;

    for (const auto& entry : fs::directory_iterator(inputPath)) {
        if (!entry.is_regular_file() || !IsStepFile(entry.path()) || IsGeneratedWingRivetStep(entry.path())) {
            continue;
        }

        totalFiles++;
        std::cout << ">>> [" << totalFiles << "] Processing: " << entry.path().filename().string() << std::endl;
        const int result = RunWingRivetInjectionImpl(
            entry.path().string(),
            BuildOutputStepPath(entry.path().string(), outputStepDir),
            BuildOutputLabelsPath(entry.path().string(), outputLabelsDir)
        );
        if (result == 0) {
            successCount++;
        } else {
            failureCount++;
            std::cout << ">>> Failed for file: " << entry.path().filename().string() << std::endl;
        }
    }

    std::cout << ">>> Batch wing rivet injection complete." << std::endl;
    std::cout << ">>> Total STEP files: " << totalFiles << std::endl;
    std::cout << ">>> Success count: " << successCount << std::endl;
    std::cout << ">>> Failure count: " << failureCount << std::endl;
    std::cout << ">>> Output STEP dir: " << outputStepDir.string() << std::endl;
    std::cout << ">>> Output labels dir: " << outputLabelsDir.string() << std::endl;

    return failureCount == 0 ? 0 : 2;
}

int RunWingRivetDatasetValidation(const std::string& inputDir) {
    const fs::path inputPath(inputDir);
    if (!fs::exists(inputPath) || !fs::is_directory(inputPath)) {
        std::cout << ">>> Input directory does not exist: " << inputDir << std::endl;
        return 1;
    }

    fs::path outputBaseDir = inputPath;
    if (inputPath.filename() == "source") {
        outputBaseDir = inputPath.parent_path();
    }
    const fs::path outputStepDir = outputBaseDir / "step";
    const fs::path outputLabelsDir = outputBaseDir / "label";
    if (!fs::exists(outputStepDir) || !fs::is_directory(outputStepDir) ||
        !fs::exists(outputLabelsDir) || !fs::is_directory(outputLabelsDir)) {
        std::cout << ">>> Missing step or label directory for: "
                  << inputDir << std::endl;
        return 1;
    }

    const fs::path statsPath = inputPath / "wing_rivet_dataset_stats.csv";
    std::ofstream statsFile(statsPath);
    statsFile
        << "model_name,step_path,labels_path,status,t1_face_id_stable,t2_injection_valid,"
        << "t3_graph_stable,face_count,label_face_count,rivet_face_count,background_face_count,"
        << "instance_count,shape_valid,duplicate_face_key_count,graph_mismatch_count,message\n";

    int total = 0;
    int passCount = 0;
    int failCount = 0;
    int totalFaces = 0;
    int totalRivetFaces = 0;
    int totalInstances = 0;

    std::cout << ">>> Validating wing-rivet dataset: " << inputDir << std::endl;
    for (const auto& entry : fs::directory_iterator(outputLabelsDir)) {
        if (!entry.is_regular_file() || entry.path().extension() != ".json") {
            continue;
        }

        const std::string suffix = "_wing_rivets.labels";
        std::string stem = entry.path().stem().string();
        if (stem.size() <= suffix.size() ||
            stem.substr(stem.size() - suffix.size()) != suffix) {
            continue;
        }
        const std::string modelStem = stem.substr(0, stem.size() - suffix.size());
        const fs::path stepPath = outputStepDir / (modelStem + "_wing_rivets.step");

        DatasetValidationResult result = ValidateWingRivetSample(entry.path(), stepPath);
        const bool pass = result.t1 && result.t2 && result.t3;
        total++;
        if (pass) {
            passCount++;
        } else {
            failCount++;
        }
        totalFaces += result.faceCount;
        totalRivetFaces += result.rivetFaceCount;
        totalInstances += result.instanceCount;

        statsFile
            << CsvEscape(modelStem) << ","
            << CsvEscape(result.stepPath) << ","
            << CsvEscape(result.labelsPath) << ","
            << (pass ? "PASS" : "FAIL") << ","
            << (result.t1 ? "PASS" : "FAIL") << ","
            << (result.t2 ? "PASS" : "FAIL") << ","
            << (result.t3 ? "PASS" : "FAIL") << ","
            << result.faceCount << ","
            << result.labelFaceCount << ","
            << result.rivetFaceCount << ","
            << result.backgroundFaceCount << ","
            << result.instanceCount << ","
            << (result.shapeValid ? "true" : "false") << ","
            << result.duplicateFaceKeyCount << ","
            << result.graphMismatchCount << ","
            << CsvEscape(result.message) << "\n";

        std::cout << ">>> [" << (pass ? "PASS" : "FAIL") << "] " << modelStem
                  << " faces=" << result.faceCount
                  << " rivet_faces=" << result.rivetFaceCount
                  << " instances=" << result.instanceCount
                  << " t1=" << (result.t1 ? "PASS" : "FAIL")
                  << " t2=" << (result.t2 ? "PASS" : "FAIL")
                  << " t3=" << (result.t3 ? "PASS" : "FAIL")
                  << " message=" << result.message
                  << std::endl;
    }

    std::cout << ">>> Dataset validation complete." << std::endl;
    std::cout << ">>> Total samples: " << total << std::endl;
    std::cout << ">>> Pass count: " << passCount << std::endl;
    std::cout << ">>> Failure count: " << failCount << std::endl;
    std::cout << ">>> Total faces: " << totalFaces << std::endl;
    std::cout << ">>> Total rivet faces: " << totalRivetFaces << std::endl;
    std::cout << ">>> Total rivet instances: " << totalInstances << std::endl;
    std::cout << ">>> Stats CSV: " << statsPath.string() << std::endl;

    return failCount == 0 ? 0 : 2;
}
