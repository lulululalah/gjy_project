#include "FeatureInjector.h"

#include "FeatureExtractor.h"

#include <BRepAlgoAPI_Fuse.hxx>
#include <BRep_Builder.hxx>
#include <BRepCheck_Analyzer.hxx>
#include <BRepClass_FaceClassifier.hxx>
#include <BRepLProp_SLProps.hxx>
#include <BRepPrimAPI_MakeCylinder.hxx>
#include <BRepTools_ReShape.hxx>
#include <BRepTools.hxx>
#include <BRepAdaptor_Surface.hxx>
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
#include <TopoDS_Face.hxx>
#include <gp_Ax2.hxx>
#include <gp_Dir.hxx>
#include <gp_Pnt.hxx>
#include <gp_Pnt2d.hxx>
#include <gp_Vec.hxx>

#include <filesystem>
#include <fstream>
#include <cctype>
#include <iomanip>
#include <iostream>
#include <algorithm>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {
struct RivetPlacement {
    int instanceId = -1;
    int hostFaceId = -1;
    double u = 0.0;
    double v = 0.0;
    double radius = 0.0;
    double height = 0.0;
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

struct InjectionTarget {
    TopoDS_Shape originalSubshape;
    TopoDS_Shape workingShape;
    bool isSubshapeMode = false;
};

bool IsShapeValid(const TopoDS_Shape& shape);

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
    const bool isWingSkinSurface = feature.surfaceType == GeomAbs_Plane;
    const bool isWingLikeOrientation =
        std::abs(feature.normalY) >= 0.95 &&
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
    return feature.area * (1.0 + std::abs(feature.centerZ)) * std::max(verticalPreference, 0.1);
}

std::vector<WingHostFace> SelectWingHostFaces(
    const std::vector<FaceFeature>& features,
    bool usePrimaryCandidates
) {
    WingHostFace positiveWing;
    WingHostFace negativeWing;
    for (const auto& feature : features) {
        const bool isCandidate = usePrimaryCandidates
            ? IsPrimaryWingSkinCandidate(feature)
            : IsFallbackWingCandidate(feature);
        if (!isCandidate) {
            continue;
        }

        const double score = ComputeWingScore(feature);
        WingHostFace* bucket = feature.centerZ >= 0.0 ? &positiveWing : &negativeWing;
        if (score > bucket->score) {
            bucket->faceId = feature.id;
            bucket->score = score;
            bucket->feature = feature;
        }
    }

    std::vector<WingHostFace> hostFaces;
    if (positiveWing.faceId > 0) {
        hostFaces.push_back(positiveWing);
    }
    if (negativeWing.faceId > 0) {
        hostFaces.push_back(negativeWing);
    }
    return hostFaces;
}

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

InjectionTarget BuildInjectionTarget(
    const TopoDS_Shape& originalShape,
    const std::vector<WingHostFace>& hostFaces
) {
    InjectionTarget target;
    if (hostFaces.empty()) {
        target.workingShape = originalShape;
        return target;
    }

    int solidIndex = -1;
    for (const auto& hostFace : hostFaces) {
        const int currentIndex = FindOwningSolidIndex(originalShape, hostFace.faceId);
        if (currentIndex <= 0) {
            solidIndex = -1;
            break;
        }

        if (solidIndex < 0) {
            solidIndex = currentIndex;
        } else if (solidIndex != currentIndex) {
            solidIndex = -1;
            break;
        }
    }

    if (solidIndex > 0) {
        target.originalSubshape = GetSolidByIndex(originalShape, solidIndex);
        target.workingShape = target.originalSubshape;
        target.isSubshapeMode = true;
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

std::vector<RivetPlacement> BuildWingRivetPlacements(
    int hostFaceId,
    const TopoDS_Face& hostFace,
    const FaceFeature& hostFeature,
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
    const double rivetRadius = std::clamp(faceScale * 0.02, 0.03, faceScale * 0.08);
    const double rivetHeight = std::max(rivetRadius * 0.65, 0.02);

    std::vector<RivetPlacement> placements;
    int instanceId = startingInstanceId;
    for (const double vFraction : vFractions) {
        for (const double uFraction : uFractions) {
            const double u = uMin + (uMax - uMin) * uFraction;
            const double v = vMin + (vMax - vMin) * vFraction;
            if (!IsUvInsideFace(hostFace, u, v)) {
                continue;
            }

            gp_Pnt point;
            gp_Dir normal;
            if (!EvaluateFacePointAndNormal(hostFace, u, v, point, normal)) {
                continue;
            }

            placements.push_back({
                instanceId++,
                hostFaceId,
                u,
                v,
                rivetRadius,
                rivetHeight,
                point,
                normal,
            });
        }
    }

    return placements;
}

TopoDS_Shape MakeRivetSolid(const RivetPlacement& placement) {
    const double embedDepth = std::max(placement.radius * 0.35, 0.01);
    gp_Vec normalVec(placement.normal);
    const gp_Pnt axisOrigin = placement.basePoint.Translated(-normalVec * embedDepth);
    gp_Ax2 axis(axisOrigin, placement.normal);
    return BRepPrimAPI_MakeCylinder(axis, placement.radius, placement.height + embedDepth).Shape();
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
    return fs::path(inputFile).parent_path() / "wing_rivet_steps";
}

fs::path BuildWingRivetLabelsDir(const std::string& inputFile) {
    return fs::path(inputFile).parent_path() / "wing_rivet_labels";
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
                 << ", \"inverse_op\": {\"kind\": \"remove_protrusion\", \"radius\": "
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

    std::cout << ">>> Injection target type=" << ShapeTypeName(currentShape.ShapeType())
              << " subshape_mode=" << (injectionTarget.isSubshapeMode ? "true" : "false")
              << " valid=" << (IsShapeValid(currentShape) ? "true" : "false")
              << " solids=" << CountSubShapes(currentShape, TopAbs_SOLID)
              << " shells=" << CountSubShapes(currentShape, TopAbs_SHELL)
              << " faces=" << CountSubShapes(currentShape, TopAbs_FACE)
              << std::endl;

    for (const auto& placement : placements) {
        const TopoDS_Shape rivetSolid = MakeRivetSolid(placement);
        BRepAlgoAPI_Fuse fuse(currentShape, rivetSolid);
        fuse.SetFuzzyValue(1.0e-6);
        fuse.Build();
        if (!fuse.IsDone()) {
            std::cout << ">>> Skipping rivet " << placement.instanceId
                      << " because fuse operation failed." << std::endl;
            std::cout << ">>>   host_face=" << placement.hostFaceId
                      << " radius=" << placement.radius
                      << " height=" << placement.height
                      << " base=(" << placement.basePoint.X()
                      << ", " << placement.basePoint.Y()
                      << ", " << placement.basePoint.Z() << ")"
                      << " normal=(" << placement.normal.X()
                      << ", " << placement.normal.Y()
                      << ", " << placement.normal.Z() << ")"
                      << std::endl;
            std::cout << ">>>   current_shape type=" << ShapeTypeName(currentShape.ShapeType())
                      << " valid=" << (IsShapeValid(currentShape) ? "true" : "false")
                      << " solids=" << CountSubShapes(currentShape, TopAbs_SOLID)
                      << " shells=" << CountSubShapes(currentShape, TopAbs_SHELL)
                      << " faces=" << CountSubShapes(currentShape, TopAbs_FACE)
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
                      << " because fused model became invalid." << std::endl;
            continue;
        }

        currentShape = candidateShape;
        acceptedRivetShapes.push_back(rivetSolid);
        rivetFacesByInstance.push_back({placement.instanceId, CollectGeneratedRivetFaces(fuse, rivetSolid)});
        labelInstances.push_back({
            placement.instanceId,
            "rivet",
            placement.hostFaceId,
            placement.radius,
            placement.height,
        });
    }

    if (labelInstances.empty()) {
        std::cout << ">>> No rivets were successfully fused into the model." << std::endl;
        return 1;
    }

    TopoDS_Shape finalOutputShape = currentShape;
    if (injectionTarget.isSubshapeMode) {
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

    const auto finalFeatures = ExtractFeatures(finalOutputShape);
    TopTools_IndexedMapOfShape finalFaceMap;
    TopExp::MapShapes(finalOutputShape, TopAbs_FACE, finalFaceMap);

    std::vector<LabelsFaceEntry> faceEntries;
    faceEntries.reserve(finalFeatures.size());
    for (const auto& feature : finalFeatures) {
        faceEntries.push_back({feature.id, "background", -1, "keep"});
    }

    for (const auto& [instanceId, rivetFaces] : rivetFacesByInstance) {
        std::set<int> finalFaceIds;
        for (const auto& rivetFace : rivetFaces) {
            const int finalFaceId = finalFaceMap.FindIndex(rivetFace);
            if (finalFaceId > 0) {
                finalFaceIds.insert(finalFaceId);
            }
        }

        for (const int finalFaceId : finalFaceIds) {
            auto& entry = faceEntries[static_cast<size_t>(finalFaceId - 1)];
            entry.semantic = "rivet";
            entry.instanceId = instanceId;
            entry.operation = "remove_protrusion";
        }
    }

    WriteLabelsJson(inputFile, outputStepFile, outputLabelsFile, faceEntries, labelInstances);

    std::cout << ">>> Wing rivet injection complete." << std::endl;
    std::cout << ">>> Output STEP: " << outputStepFile << std::endl;
    std::cout << ">>> Output labels: " << outputLabelsFile << std::endl;
    std::cout << ">>> Injected rivet count: " << labelInstances.size() << std::endl;
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

int RunBatchWingRivetInjection(const std::string& inputDir) {
    const fs::path inputPath(inputDir);
    if (!fs::exists(inputPath) || !fs::is_directory(inputPath)) {
        std::cout << ">>> Input directory does not exist: " << inputDir << std::endl;
        return 1;
    }

    std::cout << ">>> Batch injecting wing rivets in directory: " << inputDir << std::endl;

    const fs::path outputStepDir = inputPath / "wing_rivet_steps";
    const fs::path outputLabelsDir = inputPath / "wing_rivet_labels";
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
