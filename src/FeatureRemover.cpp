#include "FeatureRemover.h"

#include <BRepAlgoAPI_Defeaturing.hxx>
#include <BRepBuilderAPI_MakeEdge.hxx>
#include <BRepBuilderAPI_MakeFace.hxx>
#include <BRepBuilderAPI_Sewing.hxx>
#include <BRepBndLib.hxx>
#include <BRep_Builder.hxx>
#include <BRepCheck_Analyzer.hxx>
#include <BRepGProp.hxx>
#include <BRepLib.hxx>
#include <BRepTools.hxx>
#include <BRepTools_ReShape.hxx>
#include <BRep_Tool.hxx>
#include <Bnd_Box.hxx>
#include <GProp_GProps.hxx>
#include <GCE2d_MakeSegment.hxx>
#include <IFSelect_ReturnStatus.hxx>
#include <ShapeFix_Shape.hxx>
#include <ShapeAnalysis_Surface.hxx>
#include <ShapeUpgrade_RemoveInternalWires.hxx>
#include <ShapeUpgrade_UnifySameDomain.hxx>
#include <Standard_Failure.hxx>
#include <STEPControl_Reader.hxx>
#include <STEPControl_Writer.hxx>
#include <TopAbs_ShapeEnum.hxx>
#include <TopExp.hxx>
#include <TopExp_Explorer.hxx>
#include <TopoDS.hxx>
#include <TopoDS_Edge.hxx>
#include <TopoDS_Shape.hxx>
#include <TopoDS_Vertex.hxx>
#include <TopTools_IndexedMapOfShape.hxx>
#include <TopTools_IndexedDataMapOfShapeListOfShape.hxx>
#include <TopTools_ListOfShape.hxx>
#include <TopTools_SequenceOfShape.hxx>
#include <Geom_Surface.hxx>
#include <gp_Pnt.hxx>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace
{
    std::vector<std::string> SplitCsvRow(const std::string& row)
    {
        std::vector<std::string> fields;
        std::string field;
        bool inQuotes = false;

        for (std::size_t index = 0; index < row.size(); ++index)
        {
            const char ch = row[index];
            if (ch == '"')
            {
                if (inQuotes && index + 1 < row.size() && row[index + 1] == '"')
                {
                    field.push_back('"');
                    ++index;
                }
                else
                {
                    inQuotes = !inQuotes;
                }
            }
            else if (ch == ',' && !inQuotes)
            {
                fields.push_back(field);
                field.clear();
            }
            else
            {
                field.push_back(ch);
            }
        }
        fields.push_back(field);
        return fields;
    }

    bool LoadStep(const std::string& inputFile, TopoDS_Shape& shape)
    {
        STEPControl_Reader reader;
        if (reader.ReadFile(inputFile.c_str()) != IFSelect_RetDone)
        {
            return false;
        }
        reader.TransferRoots();
        shape = reader.OneShape();
        return !shape.IsNull();
    }

    bool SaveStep(const TopoDS_Shape& shape, const std::string& outputFile)
    {
        STEPControl_Writer writer;
        if (writer.Transfer(shape, STEPControl_AsIs) != IFSelect_RetDone)
        {
            return false;
        }
        return writer.Write(outputFile.c_str()) == IFSelect_RetDone;
    }

    struct PredictionSelection
    {
        std::set<int> allFaceIds;
        std::set<int> rivetFaceIds;
        std::set<int> surfaceFeatureFaceIds;
    };

    PredictionSelection ReadPredictedFaceIds(const std::string& predictionsFile)
    {
        std::ifstream input(predictionsFile);
        if (!input)
        {
            throw std::runtime_error("Cannot open prediction CSV: " + predictionsFile);
        }

        std::string line;
        if (!std::getline(input, line))
        {
            throw std::runtime_error("Prediction CSV is empty: " + predictionsFile);
        }

        const std::vector<std::string> header = SplitCsvRow(line);
        int faceIdColumn = -1;
        int predictionColumn = -1;
        for (std::size_t index = 0; index < header.size(); ++index)
        {
            if (header[index] == "face_id")
            {
                faceIdColumn = static_cast<int>(index);
            }
            else if (header[index] == "pred_label")
            {
                predictionColumn = static_cast<int>(index);
            }
        }
        if (faceIdColumn < 0 || predictionColumn < 0)
        {
            throw std::runtime_error("Prediction CSV must contain face_id and pred_label columns.");
        }

        PredictionSelection selection;
        int lineNumber = 1;
        while (std::getline(input, line))
        {
            ++lineNumber;
            if (line.empty())
            {
                continue;
            }
            const std::vector<std::string> fields = SplitCsvRow(line);
            const int requiredColumn = faceIdColumn > predictionColumn ? faceIdColumn : predictionColumn;
            if (static_cast<int>(fields.size()) <= requiredColumn)
            {
                throw std::runtime_error("Malformed prediction CSV row " + std::to_string(lineNumber) + ".");
            }
            try
            {
                const int faceId = std::stoi(fields[faceIdColumn]);
                if (!selection.allFaceIds.insert(faceId).second)
                {
                    throw std::runtime_error("Duplicate face_id in prediction CSV row " + std::to_string(lineNumber) + ".");
                }
                const int predictedLabel = std::stoi(fields[predictionColumn]);
                if (predictedLabel == 1)
                {
                    selection.rivetFaceIds.insert(faceId);
                }
                else if (predictedLabel == 2)
                {
                    selection.surfaceFeatureFaceIds.insert(faceId);
                }
            }
            catch (const std::exception&)
            {
                throw std::runtime_error("Invalid numeric value in prediction CSV row " + std::to_string(lineNumber) + ".");
            }
        }
        return selection;
    }

    int CountFaces(const TopoDS_Shape& shape)
    {
        TopTools_IndexedMapOfShape faces;
        TopExp::MapShapes(shape, TopAbs_FACE, faces);
        return faces.Extent();
    }

    int CountFreeEdges(const TopoDS_Shape& shape)
    {
        TopTools_IndexedMapOfShape edgeMap;
        TopExp::MapShapes(shape, TopAbs_EDGE, edgeMap);
        TopTools_IndexedDataMapOfShapeListOfShape edgeFaces;
        TopExp::MapShapesAndAncestors(
            shape, TopAbs_EDGE, TopAbs_FACE, edgeFaces);
        int freeEdgeCount = 0;
        for (int edgeId = 1; edgeId <= edgeMap.Extent(); ++edgeId)
        {
            const int ancestorIndex = edgeFaces.FindIndex(edgeMap.FindKey(edgeId));
            if (ancestorIndex > 0 &&
                edgeFaces.FindFromIndex(ancestorIndex).Extent() == 1)
            {
                ++freeEdgeCount;
            }
        }
        return freeEdgeCount;
    }

    double FaceArea(const TopoDS_Face& face)
    {
        GProp_GProps properties;
        BRepGProp::SurfaceProperties(face, properties);
        return properties.Mass();
    }

    struct WireSignature
    {
        gp_Pnt center;
        double length;
    };

    struct EdgeSignature
    {
        gp_Pnt center;
        double length;
    };

    struct FaceSignature
    {
        gp_Pnt center;
        double area;
    };

    struct RemovalSelection
    {
        std::set<int> faceIds;
        std::vector<std::set<int>> faceGroups;
        std::map<int, std::vector<WireSignature>> contactWiresBySupportFace;
    };

    WireSignature GetWireSignature(const TopoDS_Wire& wire)
    {
        GProp_GProps properties;
        BRepGProp::LinearProperties(wire, properties);
        return {properties.CentreOfMass(), properties.Mass()};
    }

    std::vector<EdgeSignature> GetFaceEdgeSignatures(
        const TopTools_IndexedMapOfShape& faceMap,
        const std::set<int>& faceIds)
    {
        TopTools_IndexedMapOfShape edges;
        for (const int faceId : faceIds)
        {
            for (TopExp_Explorer explorer(faceMap.FindKey(faceId), TopAbs_EDGE);
                 explorer.More(); explorer.Next())
            {
                edges.Add(explorer.Current());
            }
        }
        std::vector<EdgeSignature> signatures;
        signatures.reserve(edges.Extent());
        for (int edgeId = 1; edgeId <= edges.Extent(); ++edgeId)
        {
            GProp_GProps properties;
            BRepGProp::LinearProperties(edges.FindKey(edgeId), properties);
            signatures.push_back({properties.CentreOfMass(), properties.Mass()});
        }
        return signatures;
    }

    bool ContainsMatchingEdge(
        const TopoDS_Shape& shape,
        const std::vector<EdgeSignature>& targets)
    {
        for (TopExp_Explorer explorer(shape, TopAbs_EDGE);
             explorer.More(); explorer.Next())
        {
            GProp_GProps properties;
            BRepGProp::LinearProperties(explorer.Current(), properties);
            const EdgeSignature candidate{properties.CentreOfMass(), properties.Mass()};
            for (const EdgeSignature& target : targets)
            {
                const double lengthScale = std::max({candidate.length, target.length, 1.0e-9});
                if (std::abs(candidate.length - target.length) / lengthScale <= 1.0e-6 &&
                    candidate.center.Distance(target.center) <=
                        std::max(lengthScale * 1.0e-6, 1.0e-7))
                {
                    return true;
                }
            }
        }
        return false;
    }

    FaceSignature GetFaceSignature(const TopoDS_Face& face)
    {
        GProp_GProps properties;
        BRepGProp::SurfaceProperties(face, properties);
        return {properties.CentreOfMass(), properties.Mass()};
    }

    std::vector<FaceSignature> GetFaceSignatures(
        const TopTools_IndexedMapOfShape& faceMap,
        const std::set<int>& faceIds)
    {
        std::vector<FaceSignature> signatures;
        signatures.reserve(faceIds.size());
        for (const int faceId : faceIds)
        {
            signatures.push_back(GetFaceSignature(
                TopoDS::Face(faceMap.FindKey(faceId))));
        }
        return signatures;
    }

    RemovalSelection ExpandSmallConnectedFaces(
        const TopoDS_Shape& shape,
        const TopTools_IndexedMapOfShape& faceMap,
        const std::set<int>& seedFaceIds,
        const double areaMultiplier = 4.0,
        const std::size_t maximumFacesPerComponent = 256)
    {
        RemovalSelection result;
        if (seedFaceIds.empty())
        {
            return result;
        }

        double maximumSeedArea = 0.0;
        for (const int faceId : seedFaceIds)
        {
            maximumSeedArea = std::max(
                maximumSeedArea,
                FaceArea(TopoDS::Face(faceMap.FindKey(faceId))));
        }
        // A rivet can be split into several small side/bottom faces.  Do not
        // cross into a large aircraft panel while collecting that local patch.
        const double maximumConnectedFaceArea = std::max(
            maximumSeedArea * areaMultiplier, 1.0e-9);

        TopTools_IndexedDataMapOfShapeListOfShape edgeFaces;
        TopExp::MapShapesAndAncestors(shape, TopAbs_EDGE, TopAbs_FACE, edgeFaces);

        result.faceIds = seedFaceIds;
        std::set<int> supportFaceIds;
        std::vector<int> pending(seedFaceIds.begin(), seedFaceIds.end());
        for (std::size_t cursor = 0; cursor < pending.size(); ++cursor)
        {
            const TopoDS_Face currentFace = TopoDS::Face(faceMap.FindKey(pending[cursor]));
            for (TopExp_Explorer edgeExplorer(currentFace, TopAbs_EDGE);
                 edgeExplorer.More(); edgeExplorer.Next())
            {
                const TopoDS_Shape edge = edgeExplorer.Current();
                const int edgeIndex = edgeFaces.FindIndex(edge);
                if (edgeIndex <= 0)
                {
                    continue;
                }
                const TopTools_ListOfShape& adjacentFaces = edgeFaces.FindFromIndex(edgeIndex);
                for (TopTools_ListIteratorOfListOfShape faceIterator(adjacentFaces);
                     faceIterator.More(); faceIterator.Next())
                {
                    const int adjacentId = faceMap.FindIndex(faceIterator.Value());
                    if (adjacentId <= 0 || result.faceIds.count(adjacentId) != 0)
                    {
                        continue;
                    }
                    const double adjacentArea = FaceArea(TopoDS::Face(faceIterator.Value()));
                    if (adjacentArea <= maximumConnectedFaceArea &&
                        result.faceIds.size() < maximumFacesPerComponent)
                    {
                        result.faceIds.insert(adjacentId);
                        pending.push_back(adjacentId);
                    }
                    else
                    {
                        supportFaceIds.insert(adjacentId);
                    }
                }
            }
        }
        // Record only the host-face inner wires that actually touch the
        // selected rivet component.  Those loops are the Boolean contact
        // boundaries to be filled; unrelated panel openings must be kept.
        for (const int supportFaceId : supportFaceIds)
        {
            const TopoDS_Face supportFace = TopoDS::Face(faceMap.FindKey(supportFaceId));
            const TopoDS_Wire outerWire = BRepTools::OuterWire(supportFace);
            for (TopExp_Explorer wireExplorer(supportFace, TopAbs_WIRE);
                 wireExplorer.More(); wireExplorer.Next())
            {
                const TopoDS_Wire wire = TopoDS::Wire(wireExplorer.Current());
                if (wire.IsSame(outerWire))
                {
                    continue;
                }
                bool touchesSelectedFace = false;
                for (TopExp_Explorer edgeExplorer(wire, TopAbs_EDGE);
                     edgeExplorer.More() && !touchesSelectedFace; edgeExplorer.Next())
                {
                    const int edgeIndex = edgeFaces.FindIndex(edgeExplorer.Current());
                    if (edgeIndex <= 0)
                    {
                        continue;
                    }
                    const TopTools_ListOfShape& adjacentFaces = edgeFaces.FindFromIndex(edgeIndex);
                    for (TopTools_ListIteratorOfListOfShape faceIterator(adjacentFaces);
                         faceIterator.More(); faceIterator.Next())
                    {
                        const int adjacentId = faceMap.FindIndex(faceIterator.Value());
                        if (result.faceIds.count(adjacentId) != 0)
                        {
                            touchesSelectedFace = true;
                            break;
                        }
                    }
                }
                if (touchesSelectedFace)
                {
                    result.contactWiresBySupportFace[supportFaceId].push_back(
                        GetWireSignature(wire));
                }
            }
        }

        std::set<int> grouped;
        for (const int startFaceId : result.faceIds)
        {
            if (grouped.count(startFaceId) != 0)
            {
                continue;
            }
            std::set<int> group;
            std::vector<int> groupPending{startFaceId};
            grouped.insert(startFaceId);
            for (std::size_t cursor = 0; cursor < groupPending.size(); ++cursor)
            {
                const int currentId = groupPending[cursor];
                group.insert(currentId);
                const TopoDS_Face currentFace = TopoDS::Face(faceMap.FindKey(currentId));
                for (TopExp_Explorer edgeExplorer(currentFace, TopAbs_EDGE);
                     edgeExplorer.More(); edgeExplorer.Next())
                {
                    const int edgeIndex = edgeFaces.FindIndex(edgeExplorer.Current());
                    if (edgeIndex <= 0)
                    {
                        continue;
                    }
                    const TopTools_ListOfShape& adjacentFaces = edgeFaces.FindFromIndex(edgeIndex);
                    for (TopTools_ListIteratorOfListOfShape iterator(adjacentFaces);
                         iterator.More(); iterator.Next())
                    {
                        const int adjacentId = faceMap.FindIndex(iterator.Value());
                        if (result.faceIds.count(adjacentId) != 0 &&
                            grouped.insert(adjacentId).second)
                        {
                            groupPending.push_back(adjacentId);
                        }
                    }
                }
            }
            result.faceGroups.push_back(std::move(group));
        }
        return result;
    }

    bool SameContactWire(const WireSignature& candidate, const WireSignature& target)
    {
        const double lengthScale = std::max({candidate.length, target.length, 1.0e-9});
        const double relativeLengthError = std::abs(candidate.length - target.length) / lengthScale;
        const double centerTolerance = std::max(lengthScale * 0.1, 1.0e-6);
        return relativeLengthError < 0.15 &&
            candidate.center.Distance(target.center) < centerTolerance;
    }

    bool SameCoincidentWire(const WireSignature& candidate, const WireSignature& target)
    {
        const double lengthScale = std::max({candidate.length, target.length, 1.0e-9});
        return std::abs(candidate.length - target.length) / lengthScale < 0.02 &&
            candidate.center.Distance(target.center) <
                std::max(lengthScale * 0.01, 1.0e-6);
    }

    int FillContactWiresOnSupportFaces(
        TopoDS_Shape& shape,
        const std::map<int, std::vector<WireSignature>>& contactWiresBySupportFace,
        const TopTools_IndexedMapOfShape& originalFaceMap)
    {
        int replacedCount = 0;
        for (const auto& supportEntry : contactWiresBySupportFace)
        {
            const int originalId = supportEntry.first;
            if (originalId < 1 || originalId > originalFaceMap.Extent())
            {
                continue;
            }
            const TopoDS_Face originalFace = TopoDS::Face(originalFaceMap.FindKey(originalId));
            for (const WireSignature& targetWire : supportEntry.second)
            {
                TopTools_IndexedMapOfShape currentFaceMap;
                TopExp::MapShapes(shape, TopAbs_FACE, currentFaceMap);
                int currentId = currentFaceMap.FindIndex(originalFace);
                GProp_GProps originalProperties;
                BRepGProp::SurfaceProperties(originalFace, originalProperties);
                const gp_Pnt originalCenter = originalProperties.CentreOfMass();
                const double originalArea = originalProperties.Mass();
                if (currentId <= 0)
                {
                    double bestScore = std::numeric_limits<double>::max();
                    for (int candidateId = 1; candidateId <= currentFaceMap.Extent(); ++candidateId)
                    {
                        const TopoDS_Face candidate = TopoDS::Face(currentFaceMap.FindKey(candidateId));
                        GProp_GProps candidateProperties;
                        BRepGProp::SurfaceProperties(candidate, candidateProperties);
                        const double areaScale = std::max(originalArea, candidateProperties.Mass());
                        const double areaError = std::abs(originalArea - candidateProperties.Mass()) /
                            std::max(areaScale, 1.0);
                        const double centerError = originalCenter.SquareDistance(
                            candidateProperties.CentreOfMass()) / std::max(originalArea, 1.0);
                        const double score = areaError + centerError;
                        if (score < bestScore)
                        {
                            bestScore = score;
                            currentId = candidateId;
                        }
                    }
                    if (currentId <= 0 || bestScore > 0.05)
                    {
                        continue;
                    }
                }

                const TopoDS_Face currentFace = TopoDS::Face(currentFaceMap.FindKey(currentId));
                const TopoDS_Wire outerWire = BRepTools::OuterWire(currentFace);
                Handle(Geom_Surface) surface = BRep_Tool::Surface(currentFace);
                if (outerWire.IsNull() || surface.IsNull())
                {
                    continue;
                }

                BRepBuilderAPI_MakeFace maker(surface, outerWire, Standard_True);
                bool removedTargetWire = false;
                for (TopExp_Explorer wireExplorer(currentFace, TopAbs_WIRE);
                     wireExplorer.More(); wireExplorer.Next())
                {
                    const TopoDS_Wire wire = TopoDS::Wire(wireExplorer.Current());
                    if (wire.IsSame(outerWire))
                    {
                        continue;
                    }
                    if (!removedTargetWire &&
                        SameContactWire(GetWireSignature(wire), targetWire))
                    {
                        removedTargetWire = true;
                    }
                    else
                    {
                        maker.Add(wire);
                    }
                }
                if (!removedTargetWire || !maker.IsDone())
                {
                    continue;
                }
                TopoDS_Face repairedFace = maker.Face();
                repairedFace.Orientation(currentFace.Orientation());
                BRepTools_ReShape reshaper;
                reshaper.Replace(currentFace, repairedFace);
                TopoDS_Shape trialShape = reshaper.Apply(shape);
                ShapeFix_Shape trialFixer(trialShape);
                trialFixer.Perform();
                if (!trialFixer.Shape().IsNull())
                {
                    trialShape = trialFixer.Shape();
                }
                if (!trialShape.IsNull() && BRepCheck_Analyzer(trialShape).IsValid())
                {
                    shape = trialShape;
                    ++replacedCount;
                }
            }
        }
        return replacedCount;
    }

    double ShapeVolume(const TopoDS_Shape& shape)
    {
        GProp_GProps properties;
        BRepGProp::VolumeProperties(shape, properties);
        return std::abs(properties.Mass());
    }

    bool PreservesVolume(
        const TopoDS_Shape& before,
        const TopoDS_Shape& after,
        const double maximumRelativeChange = 0.01)
    {
        const double beforeVolume = ShapeVolume(before);
        const double afterVolume = ShapeVolume(after);
        if (beforeVolume <= 1.0e-12)
        {
            return true;
        }
        return std::abs(afterVolume - beforeVolume) / beforeVolume <= maximumRelativeChange;
    }

    std::set<int> MatchGroupFaces(
        const TopoDS_Shape& currentShape,
        const TopTools_IndexedMapOfShape& originalFaceMap,
        const std::set<int>& originalFaceIds)
    {
        TopTools_IndexedMapOfShape currentFaceMap;
        TopExp::MapShapes(currentShape, TopAbs_FACE, currentFaceMap);
        std::set<int> matched;
        for (const int originalId : originalFaceIds)
        {
            const TopoDS_Face originalFace = TopoDS::Face(originalFaceMap.FindKey(originalId));
            int currentId = currentFaceMap.FindIndex(originalFace);
            if (currentId <= 0)
            {
                const FaceSignature original = GetFaceSignature(originalFace);
                double bestDistance = std::numeric_limits<double>::max();
                double bestAreaError = std::numeric_limits<double>::max();
                for (int candidateId = 1; candidateId <= currentFaceMap.Extent(); ++candidateId)
                {
                    if (matched.count(candidateId) != 0)
                    {
                        continue;
                    }
                    const FaceSignature candidate = GetFaceSignature(
                        TopoDS::Face(currentFaceMap.FindKey(candidateId)));
                    const double areaScale = std::max({original.area, candidate.area, 1.0e-12});
                    const double areaError = std::abs(original.area - candidate.area) / areaScale;
                    const double distance = original.center.Distance(candidate.center);
                    if (areaError < bestAreaError ||
                        (std::abs(areaError - bestAreaError) < 1.0e-12 && distance < bestDistance))
                    {
                        bestAreaError = areaError;
                        bestDistance = distance;
                        currentId = candidateId;
                    }
                }
                constexpr double pi = 3.14159265358979323846;
                const double distanceTolerance = std::max(
                    std::sqrt(original.area / pi) * 0.25, 1.0e-6);
                if (currentId <= 0 || bestAreaError > 0.05 || bestDistance > distanceTolerance)
                {
                    continue;
                }
            }
            matched.insert(currentId);
        }
        return matched;
    }

    std::set<int> FindOriginalFacesStillPresent(
        const TopoDS_Shape& currentShape,
        const TopTools_IndexedMapOfShape& originalFaceMap,
        const std::set<int>& originalFaceIds)
    {
        TopTools_IndexedMapOfShape currentFaceMap;
        TopExp::MapShapes(currentShape, TopAbs_FACE, currentFaceMap);
        std::set<int> currentIds;
        for (const int originalId : originalFaceIds)
        {
            const int currentId = currentFaceMap.FindIndex(
                originalFaceMap.FindKey(originalId));
            if (currentId > 0)
            {
                currentIds.insert(currentId);
            }
        }
        return currentIds;
    }

    std::set<int> FindRemainingOriginalFaceIds(
        const TopoDS_Shape& currentShape,
        const TopTools_IndexedMapOfShape& originalFaceMap,
        const std::set<int>& originalFaceIds)
    {
        TopTools_IndexedMapOfShape currentFaceMap;
        TopExp::MapShapes(currentShape, TopAbs_FACE, currentFaceMap);
        std::set<int> remainingOriginalIds;
        for (const int originalId : originalFaceIds)
        {
            if (currentFaceMap.FindIndex(originalFaceMap.FindKey(originalId)) > 0)
            {
                remainingOriginalIds.insert(originalId);
            }
        }
        return remainingOriginalIds;
    }

    bool IsNearRemovedRivet(
        const FaceSignature& candidate,
        const std::vector<FaceSignature>& removedFaceSignatures)
    {
        constexpr double pi = 3.14159265358979323846;
        for (const FaceSignature& removed : removedFaceSignatures)
        {
            if (candidate.area > removed.area * 4.0)
            {
                continue;
            }
            const double radiusScale = std::sqrt(
                std::max(candidate.area, removed.area) / pi);
            const double tolerance = std::max(radiusScale * 1.5, 1.0e-6);
            if (candidate.center.Distance(removed.center) <= tolerance)
            {
                return true;
            }
        }
        return false;
    }

    int RemoveResidualRivetComponents(
        TopoDS_Shape& shape,
        const std::vector<FaceSignature>& removedFaceSignatures)
    {
        TopTools_IndexedMapOfShape faceMap;
        TopExp::MapShapes(shape, TopAbs_FACE, faceMap);
        TopTools_IndexedDataMapOfShapeListOfShape faceSolids;
        TopExp::MapShapesAndAncestors(shape, TopAbs_FACE, TopAbs_SOLID, faceSolids);

        TopTools_IndexedMapOfShape componentsToRemove;
        constexpr int maximumResidualComponentFaces = 16;
        int candidateFaceCount = 0;
        for (int faceId = 1; faceId <= faceMap.Extent(); ++faceId)
        {
            const TopoDS_Face face = TopoDS::Face(faceMap.FindKey(faceId));
            if (!IsNearRemovedRivet(GetFaceSignature(face), removedFaceSignatures))
            {
                continue;
            }
            ++candidateFaceCount;

            bool foundSmallSolid = false;
            const int solidIndex = faceSolids.FindIndex(face);
            if (solidIndex > 0)
            {
                const TopTools_ListOfShape& solids = faceSolids.FindFromIndex(solidIndex);
                for (TopTools_ListIteratorOfListOfShape iterator(solids);
                     iterator.More(); iterator.Next())
                {
                    if (CountFaces(iterator.Value()) <= maximumResidualComponentFaces)
                    {
                        componentsToRemove.Add(iterator.Value());
                        foundSmallSolid = true;
                    }
                }
            }
            if (foundSmallSolid)
            {
                continue;
            }

            // Some Boolean remnants are attached to a large shell in memory
            // and become standalone one-face shells only after STEP export.
            // Their geometry still matches the removed rivet location, so
            // remove the face occurrence itself instead of its large shell.
            componentsToRemove.Add(face);
        }

        if (componentsToRemove.IsEmpty())
        {
            std::cout << "Residual candidate faces: " << candidateFaceCount << std::endl;
            return 0;
        }
        int acceptedCount = 0;
        int rejectedCount = 0;
        for (int index = 1; index <= componentsToRemove.Extent(); ++index)
        {
            const int faceCountBefore = CountFaces(shape);
            BRepTools_ReShape reshaper;
            reshaper.Remove(componentsToRemove.FindKey(index));
            TopoDS_Shape trialShape = reshaper.Apply(shape);
            ShapeFix_Shape trialFixer(trialShape);
            trialFixer.Perform();
            trialShape = trialFixer.Shape();
            if (!trialShape.IsNull() &&
                CountFaces(trialShape) < faceCountBefore &&
                PreservesVolume(shape, trialShape, 0.005) &&
                BRepCheck_Analyzer(trialShape).IsValid())
            {
                shape = trialShape;
                ++acceptedCount;
            }
            else
            {
                ++rejectedCount;
            }
        }
        std::cout << "Residual candidate faces: " << candidateFaceCount << std::endl;
        std::cout << "Residual components rejected by BRep validation: "
                  << rejectedCount << std::endl;
        return acceptedCount;
    }

    std::vector<std::set<int>> BuildSelectedFaceGroups(
        const TopoDS_Shape& shape,
        const TopTools_IndexedMapOfShape& faceMap,
        const std::set<int>& selectedFaceIds)
    {
        TopTools_IndexedDataMapOfShapeListOfShape edgeFaces;
        TopExp::MapShapesAndAncestors(shape, TopAbs_EDGE, TopAbs_FACE, edgeFaces);
        std::set<int> grouped;
        std::vector<std::set<int>> groups;
        for (const int startFaceId : selectedFaceIds)
        {
            if (!grouped.insert(startFaceId).second)
            {
                continue;
            }
            std::set<int> group;
            std::vector<int> pending{startFaceId};
            for (std::size_t cursor = 0; cursor < pending.size(); ++cursor)
            {
                const int currentId = pending[cursor];
                group.insert(currentId);
                const TopoDS_Face face = TopoDS::Face(faceMap.FindKey(currentId));
                for (TopExp_Explorer edgeExplorer(face, TopAbs_EDGE);
                     edgeExplorer.More(); edgeExplorer.Next())
                {
                    const int edgeIndex = edgeFaces.FindIndex(edgeExplorer.Current());
                    if (edgeIndex <= 0)
                    {
                        continue;
                    }
                    const TopTools_ListOfShape& adjacentFaces = edgeFaces.FindFromIndex(edgeIndex);
                    for (TopTools_ListIteratorOfListOfShape iterator(adjacentFaces);
                         iterator.More(); iterator.Next())
                    {
                        const int adjacentId = faceMap.FindIndex(iterator.Value());
                        if (selectedFaceIds.count(adjacentId) != 0 &&
                            grouped.insert(adjacentId).second)
                        {
                            pending.push_back(adjacentId);
                        }
                    }
                }
            }
            groups.push_back(std::move(group));
        }
        return groups;
    }

    bool TryDirectFaceRemoval(
        TopoDS_Shape& shape,
        const std::set<int>& matchedFaceIds)
    {
        TopTools_IndexedMapOfShape currentFaceMap;
        TopExp::MapShapes(shape, TopAbs_FACE, currentFaceMap);
        const std::vector<EdgeSignature> removedBoundary =
            GetFaceEdgeSignatures(currentFaceMap, matchedFaceIds);
        BRepTools_ReShape reshaper;
        for (const int faceId : matchedFaceIds)
        {
            reshaper.Remove(currentFaceMap.FindKey(faceId));
        }
        TopoDS_Shape trialShape = reshaper.Apply(shape);
        if (!trialShape.IsNull() &&
            BRepCheck_Analyzer(trialShape).IsValid() &&
            !ContainsMatchingEdge(trialShape, removedBoundary))
        {
            shape = trialShape;
            return true;
        }
        ShapeFix_Shape fixer(trialShape);
        fixer.Perform();
        if (!fixer.Shape().IsNull())
        {
            trialShape = fixer.Shape();
        }
        if (trialShape.IsNull() ||
            !BRepCheck_Analyzer(trialShape).IsValid() ||
            ContainsMatchingEdge(trialShape, removedBoundary))
        {
            return false;
        }
        shape = trialShape;
        return true;
    }

    bool TryScopedDefeaturing(
        TopoDS_Shape& shape,
        const std::set<int>& faceIds)
    {
        if (faceIds.empty())
        {
            return false;
        }
        TopTools_IndexedMapOfShape faceMap;
        TopExp::MapShapes(shape, TopAbs_FACE, faceMap);
        const std::vector<EdgeSignature> removedBoundary =
            GetFaceEdgeSignatures(faceMap, faceIds);
        TopTools_IndexedDataMapOfShapeListOfShape faceSolids;
        TopExp::MapShapesAndAncestors(shape, TopAbs_FACE, TopAbs_SOLID, faceSolids);
        TopTools_IndexedDataMapOfShapeListOfShape faceShells;
        TopExp::MapShapesAndAncestors(shape, TopAbs_FACE, TopAbs_SHELL, faceShells);

        TopoDS_Shape component;
        for (const int faceId : faceIds)
        {
            const TopoDS_Shape face = faceMap.FindKey(faceId);
            TopoDS_Shape candidate;
            const int solidIndex = faceSolids.FindIndex(face);
            if (solidIndex > 0 && !faceSolids.FindFromIndex(solidIndex).IsEmpty())
            {
                candidate = faceSolids.FindFromIndex(solidIndex).First();
            }
            else
            {
                const int shellIndex = faceShells.FindIndex(face);
                if (shellIndex > 0 && !faceShells.FindFromIndex(shellIndex).IsEmpty())
                {
                    candidate = faceShells.FindFromIndex(shellIndex).First();
                }
            }
            if (candidate.IsNull() || (!component.IsNull() && !component.IsSame(candidate)))
            {
                return false;
            }
            component = candidate;
        }
        if (component.IsNull())
        {
            return false;
        }

        TopTools_ListOfShape facesToRemove;
        for (const int faceId : faceIds)
        {
            facesToRemove.Append(faceMap.FindKey(faceId));
        }
        BRepAlgoAPI_Defeaturing defeaturing;
        defeaturing.SetShape(component);
        defeaturing.AddFacesToRemove(facesToRemove);
        defeaturing.SetRunParallel(Standard_True);
        defeaturing.Build();
        if (!defeaturing.IsDone() ||
            defeaturing.Shape().IsNull() ||
            !BRepCheck_Analyzer(defeaturing.Shape()).IsValid())
        {
            return false;
        }
        BRepTools_ReShape reshaper;
        reshaper.Replace(component, defeaturing.Shape());
        const TopoDS_Shape trialShape = reshaper.Apply(shape);
        if (trialShape.IsNull() ||
            !BRepCheck_Analyzer(trialShape).IsValid() ||
            ContainsMatchingEdge(trialShape, removedBoundary))
        {
            return false;
        }
        TopTools_IndexedMapOfShape trialFaceMap;
        TopExp::MapShapes(trialShape, TopAbs_FACE, trialFaceMap);
        for (const int faceId : faceIds)
        {
            if (trialFaceMap.FindIndex(faceMap.FindKey(faceId)) > 0)
            {
                return false;
            }
        }
        shape = trialShape;
        return true;
    }

    int RemoveSurfacePatchesAndFillHosts(
        TopoDS_Shape& shape,
        const TopTools_IndexedMapOfShape& faceMap,
        const std::set<int>& selectedFaceIds,
        const bool allowInvalidResult = false,
        const bool allowSplitBoundaryMatch = false,
        const std::map<int, std::set<int>>* removalExpansion = nullptr,
        const std::map<int, TopoDS_Shape>* removalContainers = nullptr,
        const bool preserveValidSharedEdges = false,
        const bool fillHostWithSurfacePatches = false)
    {
        TopTools_IndexedDataMapOfShapeListOfShape edgeFaces;
        TopExp::MapShapesAndAncestors(shape, TopAbs_EDGE, TopAbs_FACE, edgeFaces);
        std::vector<std::pair<std::set<int>, WireSignature>> selectedBoundaries;
        const std::vector<std::set<int>> selectedGroups = BuildSelectedFaceGroups(
            shape, faceMap, selectedFaceIds);
        for (const std::set<int>& selectedGroup : selectedGroups)
        {
            TopTools_IndexedMapOfShape boundaryEdges;
            for (const int selectedFaceId : selectedGroup)
            {
                for (TopExp_Explorer edgeExplorer(
                         faceMap.FindKey(selectedFaceId), TopAbs_EDGE);
                     edgeExplorer.More(); edgeExplorer.Next())
                {
                    const TopoDS_Shape edge = edgeExplorer.Current();
                    const int edgeIndex = edgeFaces.FindIndex(edge);
                    int selectedAncestorCount = 0;
                    if (edgeIndex > 0)
                    {
                        const TopTools_ListOfShape& ancestors =
                            edgeFaces.FindFromIndex(edgeIndex);
                        for (TopTools_ListIteratorOfListOfShape iterator(ancestors);
                             iterator.More(); iterator.Next())
                        {
                            const int adjacentId = faceMap.FindIndex(iterator.Value());
                            if (selectedGroup.count(adjacentId) != 0)
                            {
                                ++selectedAncestorCount;
                            }
                        }
                    }
                    if (selectedAncestorCount == 1)
                    {
                        boundaryEdges.Add(edge);
                    }
                }
            }
            GProp_GProps boundaryProperties;
            for (int edgeId = 1; edgeId <= boundaryEdges.Extent(); ++edgeId)
            {
                GProp_GProps edgeProperties;
                BRepGProp::LinearProperties(boundaryEdges.FindKey(edgeId), edgeProperties);
                boundaryProperties.Add(edgeProperties);
            }
            if (boundaryProperties.Mass() > 0.0)
            {
                selectedBoundaries.emplace_back(
                    selectedGroup,
                    WireSignature{
                        boundaryProperties.CentreOfMass(), boundaryProperties.Mass()});
            }
        }
        std::map<int, std::set<int>> removableWireIndexes;
        std::set<int> patchFaceIds;
        TopTools_IndexedMapOfShape patchContainers;
        const auto addPatchFace = [&](const int selectedId)
        {
            patchFaceIds.insert(selectedId);
            if (removalExpansion != nullptr)
            {
                const auto expansion = removalExpansion->find(selectedId);
                if (expansion != removalExpansion->end())
                {
                    patchFaceIds.insert(
                        expansion->second.begin(), expansion->second.end());
                }
            }
            if (removalContainers != nullptr)
            {
                const auto container = removalContainers->find(selectedId);
                if (container != removalContainers->end())
                {
                    patchContainers.Add(container->second);
                }
            }
        };
        for (int supportId = 1; supportId <= faceMap.Extent(); ++supportId)
        {
            if (selectedFaceIds.count(supportId) != 0)
            {
                continue;
            }
            TopoDS_Face supportFace = TopoDS::Face(faceMap.FindKey(supportId));
            const TopoDS_Wire outerWire = BRepTools::OuterWire(supportFace);
            int wireIndex = 0;
            for (TopExp_Explorer wireExplorer(supportFace, TopAbs_WIRE);
                 wireExplorer.More(); wireExplorer.Next(), ++wireIndex)
            {
                const TopoDS_Wire wire = TopoDS::Wire(wireExplorer.Current());
                if (wire.IsSame(outerWire))
                {
                    continue;
                }
                bool touchesSelectedFace = false;
                for (TopExp_Explorer edgeExplorer(wire, TopAbs_EDGE);
                     edgeExplorer.More() && !touchesSelectedFace; edgeExplorer.Next())
                {
                    const int edgeIndex = edgeFaces.FindIndex(edgeExplorer.Current());
                    if (edgeIndex <= 0)
                    {
                        continue;
                    }
                    const TopTools_ListOfShape& adjacentFaces = edgeFaces.FindFromIndex(edgeIndex);
                    for (TopTools_ListIteratorOfListOfShape faceIterator(adjacentFaces);
                         faceIterator.More(); faceIterator.Next())
                    {
                        const int adjacentId = faceMap.FindIndex(faceIterator.Value());
                        if (selectedFaceIds.count(adjacentId) != 0)
                        {
                            touchesSelectedFace = true;
                            addPatchFace(adjacentId);
                            break;
                        }
                    }
                }
                if (!touchesSelectedFace)
                {
                    const WireSignature innerWireSignature = GetWireSignature(wire);
                    for (const auto& [selectedGroup, selectedWireSignature] :
                         selectedBoundaries)
                    {
                        const double lengthScale = std::max(
                            {innerWireSignature.length,
                             selectedWireSignature.length,
                             1.0e-9});
                        const double lengthRatio = innerWireSignature.length /
                            std::max(selectedWireSignature.length, 1.0e-9);
                        const bool splitBoundaryMatch = allowSplitBoundaryMatch &&
                            lengthRatio >= 0.30 && lengthRatio <= 3.40 &&
                            innerWireSignature.center.Distance(
                                selectedWireSignature.center) <=
                                std::max(lengthScale * 0.18, 1.0e-6);
                        if (SameCoincidentWire(
                                innerWireSignature, selectedWireSignature) ||
                            splitBoundaryMatch)
                        {
                            touchesSelectedFace = true;
                            for (const int selectedId : selectedGroup)
                            {
                                addPatchFace(selectedId);
                            }
                        }
                    }
                }
                if (touchesSelectedFace)
                {
                    removableWireIndexes[supportId].insert(wireIndex);
                }
            }
        }
        if (removableWireIndexes.empty())
        {
            return 0;
        }

        const std::vector<EdgeSignature> removedBoundary =
            GetFaceEdgeSignatures(faceMap, patchFaceIds);
        BRepTools_ReShape reshaper;
        std::set<int> containerFaceIds;
        for (int containerId = 1;
             containerId <= patchContainers.Extent(); ++containerId)
        {
            const TopoDS_Shape container = patchContainers.FindKey(containerId);
            reshaper.Remove(container);
            for (TopExp_Explorer explorer(container, TopAbs_FACE);
                 explorer.More(); explorer.Next())
            {
                const int faceId = faceMap.FindIndex(explorer.Current());
                if (faceId > 0)
                {
                    containerFaceIds.insert(faceId);
                }
            }
        }
        for (const int faceId : patchFaceIds)
        {
            if (containerFaceIds.count(faceId) == 0)
            {
                reshaper.Remove(faceMap.FindKey(faceId));
            }
        }
        int removedWireCount = 0;
        TopTools_SequenceOfShape targetedInternalWires;
        for (const auto& [supportId, indexes] : removableWireIndexes)
        {
            if (fillHostWithSurfacePatches)
            {
                std::cout << "Matched embedded-window host F" << supportId
                          << " wire indexes:";
                for (const int index : indexes)
                {
                    std::cout << ' ' << index;
                }
                std::cout << std::endl;
            }
            TopoDS_Face supportFace = TopoDS::Face(faceMap.FindKey(supportId));
            const TopoDS_Wire outerWire = BRepTools::OuterWire(supportFace);
            const Handle(Geom_Surface) surface = BRep_Tool::Surface(supportFace);
            if (outerWire.IsNull() || surface.IsNull())
            {
                continue;
            }
            if (fillHostWithSurfacePatches)
            {
                int wireIndex = 0;
                for (TopExp_Explorer wireExplorer(supportFace, TopAbs_WIRE);
                     wireExplorer.More(); wireExplorer.Next(), ++wireIndex)
                {
                    const TopoDS_Wire wire = TopoDS::Wire(wireExplorer.Current());
                    if (wire.IsSame(outerWire) || indexes.count(wireIndex) == 0)
                    {
                        continue;
                    }
                    targetedInternalWires.Append(wire);
                    ++removedWireCount;
                }
                continue;
            }
            if (preserveValidSharedEdges)
            {
                int wireIndex = 0;
                for (TopExp_Explorer wireExplorer(supportFace, TopAbs_WIRE);
                     wireExplorer.More(); wireExplorer.Next(), ++wireIndex)
                {
                    const TopoDS_Wire wire = TopoDS::Wire(wireExplorer.Current());
                    if (!wire.IsSame(outerWire) &&
                        indexes.count(wireIndex) != 0)
                    {
                        BRep_Builder directBuilder;
                        directBuilder.Remove(supportFace, wire);
                        ++removedWireCount;
                    }
                }
                continue;
            }
            BRepBuilderAPI_MakeFace maker(surface, outerWire, Standard_True);
            int wireIndex = 0;
            for (TopExp_Explorer wireExplorer(supportFace, TopAbs_WIRE);
                 wireExplorer.More(); wireExplorer.Next(), ++wireIndex)
            {
                const TopoDS_Wire wire = TopoDS::Wire(wireExplorer.Current());
                if (wire.IsSame(outerWire))
                {
                    continue;
                }
                if (indexes.count(wireIndex) != 0)
                {
                    ++removedWireCount;
                }
                else
                {
                    maker.Add(wire);
                }
            }
            if (!maker.IsDone())
            {
                return 0;
            }
            TopoDS_Face repairedFace = maker.Face();
            repairedFace.Orientation(supportFace.Orientation());
            reshaper.Replace(supportFace, repairedFace);
        }

        TopoDS_Shape trialShape;
        if (fillHostWithSurfacePatches && !targetedInternalWires.IsEmpty())
        {
            std::vector<WireSignature> targetWireSignatures;
            for (int index = 1; index <= targetedInternalWires.Length(); ++index)
            {
                targetWireSignatures.push_back(GetWireSignature(
                    TopoDS::Wire(targetedInternalWires.Value(index))));
            }
            trialShape = reshaper.Apply(shape);
            TopTools_SequenceOfShape remappedInternalWires;
            for (TopExp_Explorer faceExplorer(trialShape, TopAbs_FACE);
                 faceExplorer.More(); faceExplorer.Next())
            {
                const TopoDS_Face currentFace = TopoDS::Face(faceExplorer.Current());
                const TopoDS_Wire currentOuterWire =
                    BRepTools::OuterWire(currentFace);
                for (TopExp_Explorer wireExplorer(currentFace, TopAbs_WIRE);
                     wireExplorer.More(); wireExplorer.Next())
                {
                    const TopoDS_Wire currentWire =
                        TopoDS::Wire(wireExplorer.Current());
                    if (currentWire.IsSame(currentOuterWire))
                    {
                        continue;
                    }
                    const WireSignature currentSignature =
                        GetWireSignature(currentWire);
                    if (std::any_of(
                            targetWireSignatures.begin(),
                            targetWireSignatures.end(),
                            [&](const WireSignature& target)
                            {
                                return SameCoincidentWire(
                                    currentSignature, target);
                            }))
                    {
                        remappedInternalWires.Append(currentWire);
                    }
                }
            }
            if (remappedInternalWires.IsEmpty())
            {
                return 0;
            }
            ShapeUpgrade_RemoveInternalWires wireRemover(trialShape);
            wireRemover.MinArea() = std::numeric_limits<double>::max();
            wireRemover.RemoveFaceMode() = Standard_False;
            if (!wireRemover.Perform(remappedInternalWires) ||
                wireRemover.GetResult().IsNull())
            {
                return 0;
            }
            trialShape = wireRemover.GetResult();
        }
        else
        {
            trialShape = reshaper.Apply(shape);
        }
        const bool rawTrialIsValid = !trialShape.IsNull() &&
            BRepCheck_Analyzer(trialShape).IsValid();
        if (!preserveValidSharedEdges || !rawTrialIsValid)
        {
            ShapeFix_Shape fixer(trialShape);
            fixer.Perform();
            if (!fixer.Shape().IsNull())
            {
                trialShape = fixer.Shape();
            }
        }
        const bool validTrial = !trialShape.IsNull() &&
            BRepCheck_Analyzer(trialShape).IsValid();
        const bool usableTrial = allowInvalidResult ? !trialShape.IsNull() : validTrial;
        const bool boundaryRemains = usableTrial &&
            ContainsMatchingEdge(trialShape, removedBoundary);
        const bool patchedFacesRemain = usableTrial &&
            !FindOriginalFacesStillPresent(trialShape, faceMap, patchFaceIds).empty();
        if (!usableTrial || patchedFacesRemain)
        {
            std::cout << "Surface host-loop repair rejected: valid=" << validTrial
                      << ", boundary_remaining=" << boundaryRemains
                      << ", patched_faces_remaining=" << patchedFacesRemain
                      << ", matched_patches=" << patchFaceIds.size()
                      << ", matched_loops=" << removedWireCount << std::endl;
            return 0;
        }
        shape = trialShape;
        return removedWireCount;
    }

    int RemoveFullySelectedFreeShells(
        TopoDS_Shape& shape,
        const std::set<int>& selectedCurrentFaceIds)
    {
        TopTools_IndexedMapOfShape faceMap;
        TopExp::MapShapes(shape, TopAbs_FACE, faceMap);
        TopTools_IndexedMapOfShape shellMap;
        TopExp::MapShapes(shape, TopAbs_SHELL, shellMap);
        TopTools_IndexedMapOfShape solidMap;
        TopExp::MapShapes(shape, TopAbs_SOLID, solidMap);
        TopTools_IndexedDataMapOfShapeListOfShape faceShells;
        TopExp::MapShapesAndAncestors(shape, TopAbs_FACE, TopAbs_SHELL, faceShells);
        TopTools_IndexedDataMapOfShapeListOfShape faceSolids;
        TopExp::MapShapesAndAncestors(shape, TopAbs_FACE, TopAbs_SOLID, faceSolids);

        std::set<int> removableSolidIds;
        std::set<int> removableShellIds;
        for (const int faceId : selectedCurrentFaceIds)
        {
            const TopoDS_Shape face = faceMap.FindKey(faceId);
            const int solidAncestorIndex = faceSolids.FindIndex(face);
            if (solidAncestorIndex > 0 &&
                !faceSolids.FindFromIndex(solidAncestorIndex).IsEmpty())
            {
                const int solidId = solidMap.FindIndex(
                    faceSolids.FindFromIndex(solidAncestorIndex).First());
                bool fullySelected = solidId > 0;
                if (fullySelected)
                {
                    for (TopExp_Explorer explorer(solidMap.FindKey(solidId), TopAbs_FACE);
                         explorer.More(); explorer.Next())
                    {
                        const int solidFaceId = faceMap.FindIndex(explorer.Current());
                        if (selectedCurrentFaceIds.count(solidFaceId) == 0)
                        {
                            fullySelected = false;
                            break;
                        }
                    }
                }
                if (fullySelected)
                {
                    removableSolidIds.insert(solidId);
                    continue;
                }
            }
            const int shellAncestorIndex = faceShells.FindIndex(face);
            if (shellAncestorIndex <= 0)
            {
                continue;
            }
            const TopTools_ListOfShape& ancestors =
                faceShells.FindFromIndex(shellAncestorIndex);
            if (ancestors.IsEmpty())
            {
                continue;
            }
            const int shellId = shellMap.FindIndex(ancestors.First());
            if (shellId <= 0)
            {
                continue;
            }
            bool fullySelected = true;
            for (TopExp_Explorer explorer(shellMap.FindKey(shellId), TopAbs_FACE);
                 explorer.More(); explorer.Next())
            {
                const int shellFaceId = faceMap.FindIndex(explorer.Current());
                if (selectedCurrentFaceIds.count(shellFaceId) == 0)
                {
                    fullySelected = false;
                    break;
                }
            }
            if (fullySelected)
            {
                removableShellIds.insert(shellId);
            }
        }
        if (removableSolidIds.empty() && removableShellIds.empty())
        {
            return 0;
        }

        BRepTools_ReShape reshaper;
        for (const int solidId : removableSolidIds)
        {
            reshaper.Remove(solidMap.FindKey(solidId));
        }
        for (const int shellId : removableShellIds)
        {
            reshaper.Remove(shellMap.FindKey(shellId));
        }
        TopoDS_Shape trialShape = reshaper.Apply(shape);
        ShapeFix_Shape fixer(trialShape);
        fixer.Perform();
        if (!fixer.Shape().IsNull())
        {
            trialShape = fixer.Shape();
        }
        if (trialShape.IsNull() ||
            !BRepCheck_Analyzer(trialShape).IsValid())
        {
            return 0;
        }
        shape = trialShape;
        return static_cast<int>(removableSolidIds.size() + removableShellIds.size());
    }

    int StraightenRepeatedSharedSkinEdges(TopoDS_Shape& shape)
    {
        TopTools_IndexedMapOfShape faceMap;
        TopExp::MapShapes(shape, TopAbs_FACE, faceMap);
        TopTools_IndexedMapOfShape edgeMap;
        TopExp::MapShapes(shape, TopAbs_EDGE, edgeMap);
        TopTools_IndexedDataMapOfShapeListOfShape edgeFaces;
        TopExp::MapShapesAndAncestors(
            shape, TopAbs_EDGE, TopAbs_FACE, edgeFaces);

        std::map<std::pair<int, int>, std::vector<int>> sharedEdgesByFacePair;
        for (int edgeId = 1; edgeId <= edgeMap.Extent(); ++edgeId)
        {
            const int ancestorIndex = edgeFaces.FindIndex(edgeMap.FindKey(edgeId));
            if (ancestorIndex <= 0)
            {
                continue;
            }
            const TopTools_ListOfShape& ancestors =
                edgeFaces.FindFromIndex(ancestorIndex);
            if (ancestors.Extent() != 2)
            {
                continue;
            }
            TopTools_ListIteratorOfListOfShape iterator(ancestors);
            const int firstFaceId = faceMap.FindIndex(iterator.Value());
            iterator.Next();
            const int secondFaceId = faceMap.FindIndex(iterator.Value());
            if (firstFaceId <= 0 || secondFaceId <= 0)
            {
                continue;
            }
            sharedEdgesByFacePair[
                std::minmax(firstFaceId, secondFaceId)].push_back(edgeId);
        }

        struct SharedSkinEdge
        {
            TopoDS_Edge edge;
            int firstFaceId;
            int secondFaceId;
        };
        std::vector<SharedSkinEdge> targetEdges;
        for (const auto& [facePair, edgeIds] : sharedEdgesByFacePair)
        {
            if (edgeIds.size() < 8)
            {
                continue;
            }
            for (const int edgeId : edgeIds)
            {
                const TopoDS_Edge edge = TopoDS::Edge(edgeMap.FindKey(edgeId));
                TopoDS_Vertex firstVertex;
                TopoDS_Vertex lastVertex;
                TopExp::Vertices(edge, firstVertex, lastVertex);
                if (firstVertex.IsNull() || lastVertex.IsNull())
                {
                    continue;
                }
                GProp_GProps properties;
                BRepGProp::LinearProperties(edge, properties);
                const double edgeLength = properties.Mass();
                const double chordLength = BRep_Tool::Pnt(firstVertex).Distance(
                    BRep_Tool::Pnt(lastVertex));
                if (edgeLength <= 10.0 && chordLength >= 0.5)
                {
                    targetEdges.push_back(
                        {edge, facePair.first, facePair.second});
                }
            }
        }
        if (targetEdges.empty())
        {
            return 0;
        }

        const auto makeSurfaceConnection =
            [&faceMap](const SharedSkinEdge& target, TopoDS_Edge& newEdge)
        {
            try
            {
            TopoDS_Vertex firstVertex;
            TopoDS_Vertex lastVertex;
            TopExp::Vertices(target.edge, firstVertex, lastVertex);
            const gp_Pnt firstPoint = BRep_Tool::Pnt(firstVertex);
            const gp_Pnt lastPoint = BRep_Tool::Pnt(lastVertex);
            const TopoDS_Face firstFace = TopoDS::Face(
                faceMap.FindKey(target.firstFaceId));
            const TopoDS_Face secondFace = TopoDS::Face(
                faceMap.FindKey(target.secondFaceId));
            const Handle(Geom_Surface) firstSurface =
                BRep_Tool::Surface(firstFace);
            const Handle(Geom_Surface) secondSurface =
                BRep_Tool::Surface(secondFace);
            if (firstSurface.IsNull() || secondSurface.IsNull())
            {
                return false;
            }

            ShapeAnalysis_Surface firstAnalysis(firstSurface);
            ShapeAnalysis_Surface secondAnalysis(secondSurface);
            const gp_Pnt2d firstUv1 = firstAnalysis.ValueOfUV(
                firstPoint, 1.0e-5);
            const gp_Pnt2d firstUv2 = firstAnalysis.ValueOfUV(
                lastPoint, 1.0e-5);
            const gp_Pnt2d secondUv1 = secondAnalysis.ValueOfUV(
                firstPoint, 1.0e-5);
            const gp_Pnt2d secondUv2 = secondAnalysis.ValueOfUV(
                lastPoint, 1.0e-5);
            const Handle(Geom2d_TrimmedCurve) firstCurve =
                GCE2d_MakeSegment(firstUv1, firstUv2).Value();
            const Handle(Geom2d_TrimmedCurve) secondCurve =
                GCE2d_MakeSegment(secondUv1, secondUv2).Value();
            BRepBuilderAPI_MakeEdge edgeMaker(
                firstCurve, firstSurface, firstVertex, lastVertex);
            if (!edgeMaker.IsDone())
            {
                return false;
            }
            newEdge = edgeMaker.Edge();
            if (!BRepLib::BuildCurve3d(newEdge, 1.0e-4))
            {
                return false;
            }
            BRep_Builder builder;
            builder.UpdateEdge(newEdge, secondCurve, secondFace, 1.0e-4);
            BRepLib::SameParameter(newEdge, 1.0e-4);
            return true;
            }
            catch (const Standard_Failure&)
            {
                return false;
            }
        };

        int batchReplacementCount = 0;
        try
        {
            BRepTools_ReShape batchReplacer;
            for (const SharedSkinEdge& target : targetEdges)
            {
                TopoDS_Edge newEdge;
                if (makeSurfaceConnection(target, newEdge))
                {
                    batchReplacer.Replace(target.edge, newEdge);
                    ++batchReplacementCount;
                }
            }
            TopoDS_Shape trialShape = batchReplacer.Apply(shape);
            ShapeFix_Shape batchFixer(trialShape);
            batchFixer.Perform();
            if (!batchFixer.Shape().IsNull())
            {
                trialShape = batchFixer.Shape();
            }
            if (!trialShape.IsNull() && BRepCheck_Analyzer(trialShape).IsValid())
            {
                shape = trialShape;
                return batchReplacementCount;
            }
        }
        catch (const Standard_Failure&)
        {
            batchReplacementCount = 0;
        }

        int acceptedCount = 0;
        for (const SharedSkinEdge& target : targetEdges)
        {
            try
            {
            TopTools_IndexedMapOfShape currentEdges;
            TopExp::MapShapes(shape, TopAbs_EDGE, currentEdges);
            if (currentEdges.FindIndex(target.edge) <= 0)
            {
                continue;
            }
            TopoDS_Edge newEdge;
            if (!makeSurfaceConnection(target, newEdge))
            {
                continue;
            }
            BRepTools_ReShape replacer;
            replacer.Replace(target.edge, newEdge);
            TopoDS_Shape edgeTrial = replacer.Apply(shape);
            ShapeFix_Shape edgeFixer(edgeTrial);
            edgeFixer.Perform();
            if (!edgeFixer.Shape().IsNull())
            {
                edgeTrial = edgeFixer.Shape();
            }
            if (!edgeTrial.IsNull() && BRepCheck_Analyzer(edgeTrial).IsValid())
            {
                shape = edgeTrial;
                ++acceptedCount;
            }
            }
            catch (const Standard_Failure&)
            {
                continue;
            }
        }
        return acceptedCount;
    }

}

int RunPredictedRivetRemoval(
    const std::string& inputFile,
    const std::string& predictionsFile,
    const std::string& outputFile)
{
    try
    {
        if (std::filesystem::weakly_canonical(inputFile) ==
            std::filesystem::weakly_canonical(outputFile))
        {
            std::cerr << "Input and output STEP paths must be different." << std::endl;
            return 1;
        }

        const PredictionSelection predictions = ReadPredictedFaceIds(predictionsFile);
        if (predictions.rivetFaceIds.empty())
        {
            std::cerr << "No post-processed rivet predictions (pred_label=1) were found." << std::endl;
            return 1;
        }

        TopoDS_Shape inputShape;
        if (!LoadStep(inputFile, inputShape))
        {
            std::cerr << "Failed to read STEP file: " << inputFile << std::endl;
            return 1;
        }

        TopTools_IndexedMapOfShape faceMap;
        TopExp::MapShapes(inputShape, TopAbs_FACE, faceMap);
        if (predictions.allFaceIds.size() != static_cast<std::size_t>(faceMap.Extent()))
        {
            std::cerr << "Prediction CSV covers " << predictions.allFaceIds.size()
                      << " faces, but the STEP contains " << faceMap.Extent()
                      << ". Refusing to delete with mismatched data." << std::endl;
            return 1;
        }
        for (int faceId = 1; faceId <= faceMap.Extent(); ++faceId)
        {
            if (predictions.allFaceIds.count(faceId) == 0)
            {
                std::cerr << "Prediction CSV does not contain STEP face ID " << faceId
                          << ". Refusing to delete with incomplete data." << std::endl;
                return 1;
            }
        }

        const RemovalSelection removal = ExpandSmallConnectedFaces(
            inputShape, faceMap, predictions.rivetFaceIds);
        const std::vector<FaceSignature> removedFaceSignatures =
            GetFaceSignatures(faceMap, removal.faceIds);
        const bool inputIsValid = BRepCheck_Analyzer(inputShape).IsValid();

        TopoDS_Shape repairedShape = inputShape;
        int acceptedDefeaturingGroups = 0;
        int rejectedDefeaturingGroups = 0;
        for (const std::set<int>& originalGroup : removal.faceGroups)
        {
            const std::set<int> matchedIds = MatchGroupFaces(
                repairedShape, faceMap, originalGroup);
            if (matchedIds.empty())
            {
                ++rejectedDefeaturingGroups;
                continue;
            }
            TopTools_IndexedMapOfShape currentFaceMap;
            TopExp::MapShapes(repairedShape, TopAbs_FACE, currentFaceMap);
            TopTools_ListOfShape groupFaces;
            for (const int currentId : matchedIds)
            {
                groupFaces.Append(TopoDS::Face(currentFaceMap.FindKey(currentId)));
            }

            BRepAlgoAPI_Defeaturing defeaturing;
            defeaturing.SetShape(repairedShape);
            defeaturing.AddFacesToRemove(groupFaces);
            defeaturing.SetRunParallel(Standard_True);
            defeaturing.Build();
            if (defeaturing.IsDone() &&
                !defeaturing.Shape().IsNull() &&
                BRepCheck_Analyzer(defeaturing.Shape()).IsValid() &&
                PreservesVolume(repairedShape, defeaturing.Shape()))
            {
                repairedShape = defeaturing.Shape();
                ++acceptedDefeaturingGroups;
            }
            else
            {
                ++rejectedDefeaturingGroups;
            }
        }
        std::cout << "Rivet groups accepted: " << acceptedDefeaturingGroups
                  << ", rejected: " << rejectedDefeaturingGroups << std::endl;
        std::cout << "BRep valid after defeaturing: "
                  << (BRepCheck_Analyzer(repairedShape).IsValid() ? "yes" : "no")
                  << std::endl;
        const int removedResidualComponentCount = RemoveResidualRivetComponents(
            repairedShape, removedFaceSignatures);

        // Remove standalone Boolean remnants before rebuilding the host face.
        // Otherwise those overlapping faces can make an otherwise correct
        // contact-loop fill temporarily non-manifold and force a rollback.
        const TopoDS_Shape shapeBeforeContactFill = repairedShape;
        int filledContactWireCount = FillContactWiresOnSupportFaces(
            repairedShape, removal.contactWiresBySupportFace, faceMap);
        if (!BRepCheck_Analyzer(repairedShape).IsValid() ||
            !PreservesVolume(shapeBeforeContactFill, repairedShape))
        {
            repairedShape = shapeBeforeContactFill;
            filledContactWireCount = 0;
            std::cout << "Contact-loop fill rolled back because it produced an invalid BRep."
                      << std::endl;
        }

        // Replacing a perforated host face with its continuous underlying
        // surface can leave same-domain seam edges.  Merge them so a filled
        // rivet contact does not remain visible as a circular outline.
        const TopoDS_Shape shapeBeforeUnify = repairedShape;
        ShapeUpgrade_UnifySameDomain unify(repairedShape, Standard_True, Standard_True);
        unify.Build();
        if (!unify.Shape().IsNull() &&
            BRepCheck_Analyzer(unify.Shape()).IsValid() &&
            PreservesVolume(shapeBeforeUnify, unify.Shape()))
        {
            repairedShape = unify.Shape();
        }
        else
        {
            repairedShape = shapeBeforeUnify;
            std::cout << "Same-domain unification rolled back because it produced an invalid BRep."
                      << std::endl;
        }

        std::cout << "Boolean contact loops detected: ";
        std::size_t detectedContactWireCount = 0;
        for (const auto& entry : removal.contactWiresBySupportFace)
        {
            detectedContactWireCount += entry.second.size();
        }
        std::cout << detectedContactWireCount << std::endl;
        std::cout << "Boolean contact loops filled: " << filledContactWireCount << std::endl;
        std::cout << "Residual rivet components removed: "
                  << removedResidualComponentCount << std::endl;
        const TopoDS_Shape shapeBeforeFinalFix = repairedShape;
        ShapeFix_Shape finalFixer(repairedShape);
        finalFixer.Perform();
        if (!finalFixer.Shape().IsNull() &&
            BRepCheck_Analyzer(finalFixer.Shape()).IsValid() &&
            PreservesVolume(shapeBeforeFinalFix, finalFixer.Shape()))
        {
            repairedShape = finalFixer.Shape();
        }
        else
        {
            repairedShape = shapeBeforeFinalFix;
            std::cout << "Final ShapeFix rolled back because it produced an invalid BRep."
                      << std::endl;
        }
        if (repairedShape.IsNull() || !BRepCheck_Analyzer(repairedShape).IsValid())
        {
            std::cerr << "The repaired result is not a valid BRep; no STEP file was written." << std::endl;
            return 1;
        }

        const std::filesystem::path outputPath(outputFile);
        if (outputPath.has_parent_path())
        {
            std::filesystem::create_directories(outputPath.parent_path());
        }
        if (!SaveStep(repairedShape, outputFile))
        {
            std::cerr << "Failed to write repaired STEP file: " << outputFile << std::endl;
            return 1;
        }

        std::cout << "Predicted rivet faces selected: " << predictions.rivetFaceIds.size() << std::endl;
        std::cout << "Small connected faces selected: " << removal.faceIds.size() << std::endl;
        std::cout << "Input BRep valid: " << (inputIsValid ? "yes" : "no") << std::endl;
        std::cout << "Output BRep valid: yes" << std::endl;
        std::cout << "Face count: " << faceMap.Extent() << " -> " << CountFaces(repairedShape) << std::endl;
        std::cout << "Repaired STEP: " << outputFile << std::endl;
        return 0;
    }
    catch (const Standard_Failure& error)
    {
        std::cerr << "Predicted rivet removal failed in OpenCASCADE: "
                  << error.GetMessageString() << std::endl;
        return 1;
    }
    catch (const std::exception& error)
    {
        std::cerr << "Predicted rivet removal failed: " << error.what() << std::endl;
        return 1;
    }
}


int RunPredictedSurfaceFeatureRemoval(
    const std::string& inputFile,
    const std::string& predictionsFile,
    const std::string& outputFile)
{
    try
    {
        if (std::filesystem::weakly_canonical(inputFile) ==
            std::filesystem::weakly_canonical(outputFile))
        {
            std::cerr << "Input and output STEP paths must be different." << std::endl;
            return 1;
        }

        const PredictionSelection predictions = ReadPredictedFaceIds(predictionsFile);
        if (predictions.surfaceFeatureFaceIds.empty())
        {
            std::cerr << "No post-processed surface-feature predictions (pred_label=2) were found."
                      << std::endl;
            return 1;
        }

        TopoDS_Shape inputShape;
        if (!LoadStep(inputFile, inputShape))
        {
            std::cerr << "Failed to read STEP file: " << inputFile << std::endl;
            return 1;
        }
        TopTools_IndexedMapOfShape originalFaceMap;
        TopExp::MapShapes(inputShape, TopAbs_FACE, originalFaceMap);
        if (predictions.allFaceIds.size() !=
            static_cast<std::size_t>(originalFaceMap.Extent()))
        {
            std::cerr << "Prediction CSV covers " << predictions.allFaceIds.size()
                      << " faces, but the STEP contains " << originalFaceMap.Extent()
                      << ". Refusing to delete with mismatched data." << std::endl;
            return 1;
        }
        for (int faceId = 1; faceId <= originalFaceMap.Extent(); ++faceId)
        {
            if (predictions.allFaceIds.count(faceId) == 0)
            {
                std::cerr << "Prediction CSV does not contain STEP face ID " << faceId
                          << ". Refusing to delete with incomplete data." << std::endl;
                return 1;
            }
        }
        const bool inputIsValid = BRepCheck_Analyzer(inputShape).IsValid();
        if (!inputIsValid)
        {
            std::cout << "Warning: input STEP is not a valid BRep; using raw face removal "
                      << "without topology healing." << std::endl;
        }

        const RemovalSelection expandedSurfaceSelection = ExpandSmallConnectedFaces(
            inputShape, originalFaceMap, predictions.surfaceFeatureFaceIds,
            2.0, 32);
        const std::set<int>& surfaceFaceIds = expandedSurfaceSelection.faceIds;
        TopoDS_Shape repairedShape = inputShape;

        if (!inputIsValid)
        {
            BRepTools_ReShape reshaper;
            for (const int faceId : surfaceFaceIds)
            {
                reshaper.Remove(originalFaceMap.FindKey(faceId));
            }
            repairedShape = reshaper.Apply(inputShape);
            if (repairedShape.IsNull() ||
                CountFaces(repairedShape) >= originalFaceMap.Extent())
            {
                std::cerr << "Raw surface-feature removal did not reduce the face count; "
                          << "no STEP file was written." << std::endl;
                return 1;
            }
            const std::filesystem::path outputPath(outputFile);
            if (outputPath.has_parent_path())
            {
                std::filesystem::create_directories(outputPath.parent_path());
            }
            if (!SaveStep(repairedShape, outputFile))
            {
                std::cerr << "Failed to write raw surface-feature removal STEP: "
                          << outputFile << std::endl;
                return 1;
            }
            std::cout << "Predicted surface-feature faces selected: "
                      << predictions.surfaceFeatureFaceIds.size() << std::endl;
            std::cout << "Raw faces removed: "
                      << originalFaceMap.Extent() - CountFaces(repairedShape) << std::endl;
            std::cout << "Output BRep valid: no (input was already invalid)" << std::endl;
            std::cout << "Face count: " << originalFaceMap.Extent()
                      << " -> " << CountFaces(repairedShape) << std::endl;
            std::cout << "Surface-feature removal STEP: " << outputFile << std::endl;
            return 0;
        }
        const std::vector<std::set<int>> groups = BuildSelectedFaceGroups(
            inputShape, originalFaceMap, surfaceFaceIds);
        int removedFreeShellCount = 0;
        int directRemovalAccepted = 0;
        int scopedDefeaturingAccepted = 0;
        int rejectedGroups = 0;
        const bool useLargeBatchRemoval = surfaceFaceIds.size() > 500;

        const int filledSurfaceWireCount = RemoveSurfacePatchesAndFillHosts(
            repairedShape, originalFaceMap, surfaceFaceIds);

        std::set<int> remainingAfterMerge = FindOriginalFacesStillPresent(
            repairedShape, originalFaceMap, surfaceFaceIds);
        if (!remainingAfterMerge.empty())
        {
            removedFreeShellCount = RemoveFullySelectedFreeShells(
                repairedShape, remainingAfterMerge);
            remainingAfterMerge = FindOriginalFacesStillPresent(
                repairedShape, originalFaceMap, surfaceFaceIds);
        }
        if (!remainingAfterMerge.empty())
        {
            // Large decal/window-heavy models are prohibitively expensive when
            // every face is ShapeFix'ed independently.  Try one validated
            // batch operation; smaller models retain the original per-group
            // fallback below.
            if (useLargeBatchRemoval)
            {
                if (TryDirectFaceRemoval(repairedShape, remainingAfterMerge))
                {
                    ++directRemovalAccepted;
                }
                else if (TryScopedDefeaturing(repairedShape, remainingAfterMerge))
                {
                    ++scopedDefeaturingAccepted;
                }
                remainingAfterMerge = FindOriginalFacesStillPresent(
                    repairedShape, originalFaceMap, surfaceFaceIds);
            }
        }
        if (!remainingAfterMerge.empty() && !useLargeBatchRemoval)
        {
            for (const std::set<int>& originalGroup : groups)
            {
                const std::set<int> currentIds = FindOriginalFacesStillPresent(
                    repairedShape, originalFaceMap, originalGroup);
                if (currentIds.empty())
                {
                    continue;
                }
                if (TryDirectFaceRemoval(repairedShape, currentIds))
                {
                    ++directRemovalAccepted;
                }
                else
                {
                    ++rejectedGroups;
                }
            }
            remainingAfterMerge = FindOriginalFacesStillPresent(
                repairedShape, originalFaceMap, surfaceFaceIds);
        }
        if (!remainingAfterMerge.empty() && !useLargeBatchRemoval)
        {
            for (const std::set<int>& originalGroup : groups)
            {
                const std::set<int> currentIds = FindOriginalFacesStillPresent(
                    repairedShape, originalFaceMap, originalGroup);
                if (!currentIds.empty() && TryScopedDefeaturing(repairedShape, currentIds))
                {
                    ++scopedDefeaturingAccepted;
                }
            }
        }

        const TopoDS_Shape shapeBeforeFinalFix = repairedShape;
        ShapeFix_Shape finalFixer(repairedShape);
        finalFixer.Perform();
        if (!finalFixer.Shape().IsNull() &&
            BRepCheck_Analyzer(finalFixer.Shape()).IsValid() &&
            PreservesVolume(shapeBeforeFinalFix, finalFixer.Shape()))
        {
            repairedShape = finalFixer.Shape();
        }
        if (repairedShape.IsNull() || !BRepCheck_Analyzer(repairedShape).IsValid())
        {
            std::cerr << "The surface-feature removal result is not a valid BRep; "
                      << "no STEP file was written." << std::endl;
            return 1;
        }

        // Only heal seams when a host inner loop was actually detected and
        // rebuilt.  Models without that topology keep the ordinary local
        // deletion path and are not subjected to global surface merging.
        if (filledSurfaceWireCount > 0)
        {
            const TopoDS_Shape shapeBeforeUnify = repairedShape;
            ShapeUpgrade_UnifySameDomain unify(
                repairedShape, Standard_True, Standard_True);
            unify.Build();
            if (!unify.Shape().IsNull() &&
                BRepCheck_Analyzer(unify.Shape()).IsValid())
            {
                repairedShape = unify.Shape();
            }

            BRepBuilderAPI_Sewing sewing(1.0e-4, Standard_True, Standard_True,
                                         Standard_True, Standard_False);
            sewing.Add(repairedShape);
            sewing.Perform();
            if (!sewing.SewedShape().IsNull() &&
                BRepCheck_Analyzer(sewing.SewedShape()).IsValid())
            {
                repairedShape = sewing.SewedShape();
                ShapeUpgrade_UnifySameDomain sewnUnify(
                    repairedShape, Standard_True, Standard_True);
                sewnUnify.Build();
                if (!sewnUnify.Shape().IsNull() &&
                    BRepCheck_Analyzer(sewnUnify.Shape()).IsValid())
                {
                    repairedShape = sewnUnify.Shape();
                }
            }
        }

        const std::set<int> remainingSurfaceFaces = FindRemainingOriginalFaceIds(
            repairedShape, originalFaceMap, surfaceFaceIds);
        if (!remainingSurfaceFaces.empty())
        {
            std::cout << "Surface-feature faces skipped because no valid repair was found:"
                      << std::endl;
            std::cout << "Skipped original F-numbers:";
            for (const int faceId : remainingSurfaceFaces)
            {
                std::cout << ' ' << faceId;
            }
            std::cout << std::endl;
        }
        const std::filesystem::path outputPath(outputFile);
        if (outputPath.has_parent_path())
        {
            std::filesystem::create_directories(outputPath.parent_path());
        }
        if (!SaveStep(repairedShape, outputFile))
        {
            std::cerr << "Failed to write surface-feature removal STEP: "
                      << outputFile << std::endl;
            return 1;
        }

        std::cout << "Predicted surface-feature faces selected: "
                  << predictions.surfaceFeatureFaceIds.size() << std::endl;
        std::cout << "Surface-feature groups: " << groups.size() << std::endl;
        std::cout << "Surface-feature host loops filled: "
                  << filledSurfaceWireCount << std::endl;
        std::cout << "Fully selected free shells removed: "
                  << removedFreeShellCount << std::endl;
        std::cout << "Groups removed directly: " << directRemovalAccepted << std::endl;
        std::cout << "Groups removed by scoped defeaturing: "
                  << scopedDefeaturingAccepted << std::endl;
        std::cout << "Groups rejected: " << rejectedGroups << std::endl;
        std::cout << "Predicted faces skipped: "
                  << remainingSurfaceFaces.size() << std::endl;
        std::cout << "Output BRep valid: yes" << std::endl;
        std::cout << "Face count: " << originalFaceMap.Extent()
                  << " -> " << CountFaces(repairedShape) << std::endl;
        std::cout << "Surface-feature removal STEP: " << outputFile << std::endl;
        return 0;
    }
    catch (const Standard_Failure& error)
    {
        std::cerr << "Predicted surface-feature removal failed in OpenCASCADE: "
                  << error.GetMessageString() << std::endl;
        return 1;
    }
    catch (const std::exception& error)
    {
        std::cerr << "Predicted surface-feature removal failed: "
                  << error.what() << std::endl;
        return 1;
    }
}

int RunInvalidSurfaceHostRebuild(
    const std::string& inputFile,
    const std::string& predictionsFile,
    const std::string& outputFile)
{
    try
    {
        if (std::filesystem::weakly_canonical(inputFile) ==
            std::filesystem::weakly_canonical(outputFile))
        {
            std::cerr << "Input and output STEP paths must be different." << std::endl;
            return 1;
        }

        const PredictionSelection predictions = ReadPredictedFaceIds(predictionsFile);
        std::set<int> selectedFaceIds = predictions.surfaceFeatureFaceIds;
        selectedFaceIds.insert(
            predictions.rivetFaceIds.begin(), predictions.rivetFaceIds.end());
        if (selectedFaceIds.empty())
        {
            std::cerr << "No non-background predictions were found." << std::endl;
            return 1;
        }

        TopoDS_Shape inputShape;
        if (!LoadStep(inputFile, inputShape))
        {
            std::cerr << "Failed to read STEP file: " << inputFile << std::endl;
            return 1;
        }
        TopTools_IndexedMapOfShape originalFaceMap;
        TopExp::MapShapes(inputShape, TopAbs_FACE, originalFaceMap);
        if (predictions.allFaceIds.size() !=
            static_cast<std::size_t>(originalFaceMap.Extent()))
        {
            std::cerr << "Prediction CSV covers " << predictions.allFaceIds.size()
                      << " faces, but the STEP contains " << originalFaceMap.Extent()
                      << ". Refusing to rebuild with mismatched data." << std::endl;
            return 1;
        }
        if (BRepCheck_Analyzer(inputShape).IsValid())
        {
            std::cerr << "This dedicated rebuild mode is only for an already-invalid "
                      << "source BRep. Use the standard removal mode for valid models."
                      << std::endl;
            return 1;
        }

        TopoDS_Shape rebuiltShape = inputShape;
        const int filledHostLoopCount = RemoveSurfacePatchesAndFillHosts(
            rebuiltShape, originalFaceMap, selectedFaceIds, true);
        if (filledHostLoopCount <= 0 || rebuiltShape.IsNull())
        {
            std::cerr << "No predicted window host loops could be rebuilt; "
                      << "no candidate STEP was written." << std::endl;
            return 1;
        }

        // Remove only selected faces that are still present after the host
        // faces have been rebuilt.  Do not run global same-domain unification:
        // it is both expensive and unsafe on an already-invalid assembly.
        TopTools_IndexedMapOfShape rebuiltFaceMap;
        TopExp::MapShapes(rebuiltShape, TopAbs_FACE, rebuiltFaceMap);
        BRepTools_ReShape remainingRemover;
        int remainingSelectedFaceCount = 0;
        for (const int originalFaceId : selectedFaceIds)
        {
            const int rebuiltFaceId = rebuiltFaceMap.FindIndex(
                originalFaceMap.FindKey(originalFaceId));
            if (rebuiltFaceId > 0)
            {
                remainingRemover.Remove(rebuiltFaceMap.FindKey(rebuiltFaceId));
                ++remainingSelectedFaceCount;
            }
        }
        if (remainingSelectedFaceCount > 0)
        {
            const TopoDS_Shape removedShape = remainingRemover.Apply(rebuiltShape);
            if (!removedShape.IsNull())
            {
                rebuiltShape = removedShape;
            }
        }

        const std::filesystem::path outputPath(outputFile);
        if (outputPath.has_parent_path())
        {
            std::filesystem::create_directories(outputPath.parent_path());
        }
        if (!SaveStep(rebuiltShape, outputFile))
        {
            std::cerr << "Failed to write invalid-BRep host rebuild candidate: "
                      << outputFile << std::endl;
            return 1;
        }

        std::cout << "Invalid-BRep predicted faces selected: "
                  << selectedFaceIds.size() << std::endl;
        std::cout << "Window host loops rebuilt: " << filledHostLoopCount << std::endl;
        std::cout << "Remaining selected faces removed: "
                  << remainingSelectedFaceCount << std::endl;
        std::cout << "Output BRep valid: "
                  << (BRepCheck_Analyzer(rebuiltShape).IsValid() ? "yes" : "no")
                  << " (diagnostic only)" << std::endl;
        std::cout << "Face count: " << originalFaceMap.Extent()
                  << " -> " << CountFaces(rebuiltShape) << std::endl;
        std::cout << "Invalid-BRep host rebuild candidate: " << outputFile << std::endl;
        return 0;
    }
    catch (const Standard_Failure& error)
    {
        std::cerr << "Invalid-BRep host rebuild failed in OpenCASCADE: "
                  << error.GetMessageString() << std::endl;
        return 1;
    }
    catch (const std::exception& error)
    {
        std::cerr << "Invalid-BRep host rebuild failed: "
                  << error.what() << std::endl;
        return 1;
    }
}

int RunSplitWindowSkinRebuild(
    const std::string& inputFile,
    const std::string& predictionsFile,
    const std::string& outputFile)
{
    try
    {
        if (std::filesystem::weakly_canonical(inputFile) ==
            std::filesystem::weakly_canonical(outputFile))
        {
            std::cerr << "Input and output STEP paths must be different." << std::endl;
            return 1;
        }

        const PredictionSelection predictions = ReadPredictedFaceIds(predictionsFile);
        if (predictions.surfaceFeatureFaceIds.empty())
        {
            std::cerr << "No surface-feature predictions were found." << std::endl;
            return 1;
        }

        TopoDS_Shape inputShape;
        if (!LoadStep(inputFile, inputShape))
        {
            std::cerr << "Failed to read STEP file: " << inputFile << std::endl;
            return 1;
        }
        TopTools_IndexedMapOfShape originalFaceMap;
        TopExp::MapShapes(inputShape, TopAbs_FACE, originalFaceMap);
        if (predictions.allFaceIds.size() !=
            static_cast<std::size_t>(originalFaceMap.Extent()))
        {
            std::cerr << "Prediction CSV covers " << predictions.allFaceIds.size()
                      << " faces, but the STEP contains " << originalFaceMap.Extent()
                      << ". Refusing to rebuild with mismatched data." << std::endl;
            return 1;
        }

        const bool inputIsValid = BRepCheck_Analyzer(inputShape).IsValid();
        const int inputFreeEdgeCount = CountFreeEdges(inputShape);
        TopoDS_Shape rebuiltShape = inputShape;
        const int filledHostLoopCount = RemoveSurfacePatchesAndFillHosts(
            rebuiltShape, originalFaceMap,
            predictions.surfaceFeatureFaceIds, !inputIsValid, true);
        if (filledHostLoopCount <= 0 || rebuiltShape.IsNull())
        {
            std::cerr << "No split-window host loops could be rebuilt; "
                      << "no candidate STEP was written." << std::endl;
            return 1;
        }

        int directRemovalAccepted = 0;
        int scopedRemovalAccepted = 0;
        std::set<int> removableWindowFaceIds;
        for (const int faceId : predictions.surfaceFeatureFaceIds)
        {
            const TopoDS_Face face = TopoDS::Face(originalFaceMap.FindKey(faceId));
            const TopoDS_Wire outerWire = BRepTools::OuterWire(face);
            int innerWireCount = 0;
            for (TopExp_Explorer wireExplorer(face, TopAbs_WIRE);
                 wireExplorer.More(); wireExplorer.Next())
            {
                if (!wireExplorer.Current().IsSame(outerWire))
                {
                    ++innerWireCount;
                }
            }
            if (innerWireCount == 0)
            {
                removableWindowFaceIds.insert(faceId);
            }
        }
        const std::vector<std::set<int>> groups = BuildSelectedFaceGroups(
            inputShape, originalFaceMap, removableWindowFaceIds);
        constexpr std::size_t windowGroupsPerBatch = 8;
        for (std::size_t batchStart = 0;
             batchStart < groups.size();
             batchStart += windowGroupsPerBatch)
        {
            const TopoDS_Shape shapeBeforeBatch = rebuiltShape;
            std::set<int> batchFaceIds;
            const std::size_t batchEnd = std::min(
                groups.size(), batchStart + windowGroupsPerBatch);
            for (std::size_t groupIndex = batchStart;
                 groupIndex < batchEnd; ++groupIndex)
            {
                const std::set<int> currentIds = MatchGroupFaces(
                    rebuiltShape, originalFaceMap, groups[groupIndex]);
                batchFaceIds.insert(currentIds.begin(), currentIds.end());
            }
            bool acceptedBatch = !batchFaceIds.empty() &&
                TryDirectFaceRemoval(rebuiltShape, batchFaceIds);
            if (acceptedBatch &&
                CountFreeEdges(rebuiltShape) <= inputFreeEdgeCount)
            {
                ++directRemovalAccepted;
                continue;
            }
            rebuiltShape = shapeBeforeBatch;

            // A mixed batch can contain one window whose host has not been
            // reconstructed. Retry each group, but keep only closed-shell
            // results so no real opening is introduced.
            for (std::size_t groupIndex = batchStart;
                 groupIndex < batchEnd; ++groupIndex)
            {
                const std::set<int> currentIds = MatchGroupFaces(
                    rebuiltShape, originalFaceMap, groups[groupIndex]);
                if (currentIds.empty())
                {
                    continue;
                }
                const TopoDS_Shape shapeBeforeGroup = rebuiltShape;
                if (TryDirectFaceRemoval(rebuiltShape, currentIds) &&
                    CountFreeEdges(rebuiltShape) <= inputFreeEdgeCount)
                {
                    ++directRemovalAccepted;
                }
                else
                {
                    rebuiltShape = shapeBeforeGroup;
                }
            }
        }

        const TopoDS_Shape shapeBeforeEdgeStraightening = rebuiltShape;
        int straightenedSharedEdgeCount =
            StraightenRepeatedSharedSkinEdges(rebuiltShape);
        if (CountFreeEdges(rebuiltShape) > inputFreeEdgeCount)
        {
            rebuiltShape = shapeBeforeEdgeStraightening;
            straightenedSharedEdgeCount = 0;
        }

        // DC-10 represents the visible window outlines as shared edges between
        // adjacent B-spline skin patches.  After the window faces and host
        // loops are removed, concatenate compatible B-spline patches so those
        // shared trim edges disappear from the fuselage surface.
        const TopoDS_Shape shapeBeforeSkinMerge = rebuiltShape;
        try
        {
            ShapeUpgrade_UnifySameDomain skinMerge(
                rebuiltShape, Standard_True, Standard_True, Standard_True);
            skinMerge.SetLinearTolerance(1.0e-4);
            skinMerge.SetAngularTolerance(1.0e-3);
            skinMerge.Build();
            if (!skinMerge.Shape().IsNull() &&
                (!inputIsValid || BRepCheck_Analyzer(skinMerge.Shape()).IsValid()) &&
                CountFreeEdges(skinMerge.Shape()) <= inputFreeEdgeCount)
            {
                rebuiltShape = skinMerge.Shape();
            }
            else
            {
                rebuiltShape = shapeBeforeSkinMerge;
            }
        }
        catch (const Standard_Failure&)
        {
            rebuiltShape = shapeBeforeSkinMerge;
        }

        if (rebuiltShape.IsNull() ||
            (inputIsValid && !BRepCheck_Analyzer(rebuiltShape).IsValid()) ||
            CountFreeEdges(rebuiltShape) > inputFreeEdgeCount)
        {
            std::cerr << "Split-window skin rebuild produced an unusable result; "
                      << "no candidate STEP was written." << std::endl;
            return 1;
        }

        const std::set<int> remainingFaceIds = FindRemainingOriginalFaceIds(
            rebuiltShape, originalFaceMap, predictions.surfaceFeatureFaceIds);
        const std::filesystem::path outputPath(outputFile);
        if (outputPath.has_parent_path())
        {
            std::filesystem::create_directories(outputPath.parent_path());
        }
        if (!SaveStep(rebuiltShape, outputFile))
        {
            std::cerr << "Failed to write split-window skin rebuild candidate: "
                      << outputFile << std::endl;
            return 1;
        }

        std::cout << "Split-window predicted faces selected: "
                  << predictions.surfaceFeatureFaceIds.size() << std::endl;
        std::cout << "Single-boundary window faces eligible for removal: "
                  << removableWindowFaceIds.size() << std::endl;
        std::cout << "Split-window host loops rebuilt: "
                  << filledHostLoopCount << std::endl;
        std::cout << "Remaining groups removed directly: "
                  << directRemovalAccepted << std::endl;
        std::cout << "Remaining groups removed by scoped defeaturing: "
                  << scopedRemovalAccepted << std::endl;
        std::cout << "Window detour edges replaced by vertex connections: "
                  << straightenedSharedEdgeCount << std::endl;
        std::cout << "Predicted faces still present: "
                  << remainingFaceIds.size() << std::endl;
        std::cout << "Output BRep valid: "
                  << (BRepCheck_Analyzer(rebuiltShape).IsValid() ? "yes" : "no")
                  << std::endl;
        std::cout << "Free edges: " << inputFreeEdgeCount
                  << " -> " << CountFreeEdges(rebuiltShape) << std::endl;
        std::cout << "Face count: " << originalFaceMap.Extent()
                  << " -> " << CountFaces(rebuiltShape) << std::endl;
        std::cout << "Split-window skin rebuild candidate: "
                  << outputFile << std::endl;
        return 0;
    }
    catch (const Standard_Failure& error)
    {
        std::cerr << "Split-window skin rebuild failed in OpenCASCADE: "
                  << error.GetMessageString() << std::endl;
        return 1;
    }
    catch (const std::exception& error)
    {
        std::cerr << "Split-window skin rebuild failed: "
                  << error.what() << std::endl;
        return 1;
    }
}

int RunBridgeSplitWindowFace(
    const std::string& inputFile,
    const int windowFaceId,
    const std::string& outputFile)
{
    try
    {
        TopoDS_Shape inputShape;
        if (!LoadStep(inputFile, inputShape))
        {
            std::cerr << "Failed to read STEP file: " << inputFile << std::endl;
            return 1;
        }

        TopTools_IndexedMapOfShape faceMap;
        TopExp::MapShapes(inputShape, TopAbs_FACE, faceMap);
        if (windowFaceId <= 0 || windowFaceId > faceMap.Extent())
        {
            std::cerr << "Window face ID is outside the STEP face range." << std::endl;
            return 1;
        }
        const TopoDS_Face windowFace = TopoDS::Face(
            faceMap.FindKey(windowFaceId));

        TopTools_IndexedDataMapOfShapeListOfShape edgeFaces;
        TopExp::MapShapesAndAncestors(
            inputShape, TopAbs_EDGE, TopAbs_FACE, edgeFaces);
        std::vector<TopoDS_Edge> windowEdges;
        std::vector<TopoDS_Face> hostFaces;
        for (TopExp_Explorer explorer(windowFace, TopAbs_EDGE);
             explorer.More(); explorer.Next())
        {
            const TopoDS_Edge edge = TopoDS::Edge(explorer.Current());
            const int ancestorIndex = edgeFaces.FindIndex(edge);
            if (ancestorIndex <= 0)
            {
                continue;
            }
            TopoDS_Face hostFace;
            for (TopTools_ListIteratorOfListOfShape iterator(
                     edgeFaces.FindFromIndex(ancestorIndex));
                 iterator.More(); iterator.Next())
            {
                if (!iterator.Value().IsSame(windowFace))
                {
                    hostFace = TopoDS::Face(iterator.Value());
                    break;
                }
            }
            if (!hostFace.IsNull())
            {
                windowEdges.push_back(edge);
                hostFaces.push_back(hostFace);
            }
        }
        if (windowEdges.size() != 2 || hostFaces.size() != 2 ||
            hostFaces[0].IsSame(hostFaces[1]))
        {
            std::cerr << "Selected face is not a two-edge split window between "
                      << "two different skin faces." << std::endl;
            return 1;
        }

        TopoDS_Vertex firstVertex;
        TopoDS_Vertex lastVertex;
        TopExp::Vertices(windowEdges.front(), firstVertex, lastVertex);
        if (firstVertex.IsNull() || lastVertex.IsNull())
        {
            std::cerr << "Window endpoints could not be resolved." << std::endl;
            return 1;
        }

        const Handle(Geom_Surface) firstSurface = BRep_Tool::Surface(hostFaces[0]);
        const Handle(Geom_Surface) secondSurface = BRep_Tool::Surface(hostFaces[1]);
        ShapeAnalysis_Surface firstAnalysis(firstSurface);
        ShapeAnalysis_Surface secondAnalysis(secondSurface);
        const gp_Pnt firstPoint = BRep_Tool::Pnt(firstVertex);
        const gp_Pnt lastPoint = BRep_Tool::Pnt(lastVertex);
        const Handle(Geom2d_TrimmedCurve) firstCurve = GCE2d_MakeSegment(
            firstAnalysis.ValueOfUV(firstPoint, 1.0e-5),
            firstAnalysis.ValueOfUV(lastPoint, 1.0e-5)).Value();
        const Handle(Geom2d_TrimmedCurve) secondCurve = GCE2d_MakeSegment(
            secondAnalysis.ValueOfUV(firstPoint, 1.0e-5),
            secondAnalysis.ValueOfUV(lastPoint, 1.0e-5)).Value();

        BRepBuilderAPI_MakeEdge edgeMaker(
            firstCurve, firstSurface, firstVertex, lastVertex);
        if (!edgeMaker.IsDone())
        {
            std::cerr << "Failed to build the replacement skin edge." << std::endl;
            return 1;
        }
        TopoDS_Edge bridgeEdge = edgeMaker.Edge();
        BRep_Builder builder;
        builder.UpdateEdge(bridgeEdge, secondCurve, hostFaces[1], 1.0e-4);
        BRepLib::BuildCurve3d(bridgeEdge, 1.0e-4);
        BRepLib::SameParameter(bridgeEdge, 1.0e-4);

        BRepTools_ReShape reshaper;
        reshaper.Replace(windowEdges[0], bridgeEdge);
        reshaper.Replace(windowEdges[1], bridgeEdge);
        reshaper.Remove(windowFace);
        TopoDS_Shape candidate = reshaper.Apply(inputShape);
        ShapeFix_Shape fixer(candidate);
        fixer.Perform();
        if (!fixer.Shape().IsNull())
        {
            candidate = fixer.Shape();
        }

        const int inputFreeEdges = CountFreeEdges(inputShape);
        const int outputFreeEdges = CountFreeEdges(candidate);
        if (candidate.IsNull() || !BRepCheck_Analyzer(candidate).IsValid() ||
            outputFreeEdges > inputFreeEdges)
        {
            std::cerr << "Single-window bridge failed topology validation; "
                      << "no candidate STEP was written. Free edges: "
                      << inputFreeEdges << " -> " << outputFreeEdges << std::endl;
            return 1;
        }

        const std::filesystem::path outputPath(outputFile);
        if (outputPath.has_parent_path())
        {
            std::filesystem::create_directories(outputPath.parent_path());
        }
        if (!SaveStep(candidate, outputFile))
        {
            std::cerr << "Failed to write single-window bridge candidate." << std::endl;
            return 1;
        }
        std::cout << "Bridged split window face: F" << windowFaceId << std::endl;
        std::cout << "Host faces: F" << faceMap.FindIndex(hostFaces[0])
                  << " and F" << faceMap.FindIndex(hostFaces[1]) << std::endl;
        std::cout << "BRep valid: yes" << std::endl;
        std::cout << "Free edges: " << inputFreeEdges
                  << " -> " << outputFreeEdges << std::endl;
        std::cout << "Face count: " << faceMap.Extent()
                  << " -> " << CountFaces(candidate) << std::endl;
        std::cout << "Single-window candidate: " << outputFile << std::endl;
        return 0;
    }
    catch (const Standard_Failure& error)
    {
        std::cerr << "Single-window bridge failed in OpenCASCADE: "
                  << error.GetMessageString() << std::endl;
        return 1;
    }
}

int RunEmbeddedWindowHostRebuild(
    const std::string& inputFile,
    const std::string& predictionsFile,
    const std::string& outputFile,
    const std::string& rebuildProfile)
{
    try
    {
        std::string effectiveProfile = rebuildProfile;
        const std::string inputModelName =
            std::filesystem::path(inputFile).filename().string();
        if (effectiveProfile == "auto")
        {
            if (inputModelName.find("Airplane_body") == 0)
            {
                effectiveProfile = "airplane-body";
            }
            else if (inputModelName.find("AULIRA_2") == 0)
            {
                effectiveProfile = "aulira";
            }
            else if (inputModelName.find("Airbus") == 0)
            {
                effectiveProfile = "airbus";
            }
            else if (inputModelName.find("Gulfstream_G280") == 0)
            {
                effectiveProfile = "gulfstream-g280";
            }
            else
            {
                effectiveProfile = "generic";
            }
        }
        const bool rebuildGulfstreamTailDecals =
            effectiveProfile == "gulfstream-g280";
        const bool rebuildAirbusWindows =
            effectiveProfile == "airbus";
        const bool usePredictedWindowFaceAnchors =
            effectiveProfile == "aulira" || rebuildAirbusWindows ||
            rebuildGulfstreamTailDecals;
        if (effectiveProfile != "generic" &&
            effectiveProfile != "airplane-body" &&
            effectiveProfile != "aulira" &&
            !rebuildAirbusWindows &&
            !rebuildGulfstreamTailDecals)
        {
            std::cerr << "Unknown embedded-window rebuild profile: "
                      << effectiveProfile << std::endl;
            return 1;
        }
        const bool profileMatchesInput =
            effectiveProfile == "generic" ||
            (effectiveProfile == "airplane-body" &&
             inputModelName.find("Airplane_body") == 0) ||
            (effectiveProfile == "aulira" &&
             inputModelName.find("AULIRA_2") == 0) ||
            (rebuildAirbusWindows &&
             inputModelName.find("Airbus") == 0) ||
            (rebuildGulfstreamTailDecals &&
             inputModelName.find("Gulfstream_G280") == 0);
        if (!profileMatchesInput)
        {
            std::cerr << "Rebuild profile '" << effectiveProfile
                      << "' does not match input model: " << inputFile
                      << std::endl;
            return 1;
        }
        std::cout << "Embedded-window rebuild profile: "
                  << effectiveProfile;
        if (rebuildProfile == "auto")
        {
            std::cout << " (auto-selected)";
        }
        std::cout << std::endl;
        if (std::filesystem::weakly_canonical(inputFile) ==
            std::filesystem::weakly_canonical(outputFile))
        {
            std::cerr << "Input and output STEP paths must be different." << std::endl;
            return 1;
        }

        const PredictionSelection predictions = ReadPredictedFaceIds(predictionsFile);
        if (predictions.surfaceFeatureFaceIds.empty())
        {
            std::cerr << "No surface-feature predictions were found." << std::endl;
            return 1;
        }

        TopoDS_Shape inputShape;
        if (!LoadStep(inputFile, inputShape))
        {
            std::cerr << "Failed to read STEP file: " << inputFile << std::endl;
            return 1;
        }
        if (!BRepCheck_Analyzer(inputShape).IsValid())
        {
            std::cerr << "Embedded-window host rebuild requires a valid input BRep."
                      << std::endl;
            return 1;
        }

        TopTools_IndexedMapOfShape originalFaceMap;
        TopExp::MapShapes(inputShape, TopAbs_FACE, originalFaceMap);
        if (predictions.allFaceIds.size() !=
            static_cast<std::size_t>(originalFaceMap.Extent()))
        {
            std::cerr << "Prediction CSV and STEP face counts do not match."
                      << std::endl;
            return 1;
        }

        TopTools_IndexedDataMapOfShapeListOfShape originalEdgeFaces;
        TopExp::MapShapesAndAncestors(
            inputShape, TopAbs_EDGE, TopAbs_FACE, originalEdgeFaces);
        std::set<int> selectedSurfaceFeatureFaceIds =
            predictions.surfaceFeatureFaceIds;

        TopTools_IndexedMapOfShape solidMap;
        TopExp::MapShapes(inputShape, TopAbs_SOLID, solidMap);
        TopTools_IndexedDataMapOfShapeListOfShape faceSolids;
        TopExp::MapShapesAndAncestors(
            inputShape, TopAbs_FACE, TopAbs_SOLID, faceSolids);
        std::map<int, std::set<int>> removalExpansion;
        std::map<int, TopoDS_Shape> removalContainers;
        std::set<int> embeddedWindowSeedFaceIds;
        constexpr int maximumEmbeddedWindowSolidFaces = 16;
        for (const int selectedId : selectedSurfaceFeatureFaceIds)
        {
            const TopoDS_Shape selectedFace = originalFaceMap.FindKey(selectedId);
            const int ancestorIndex = faceSolids.FindIndex(selectedFace);
            if (ancestorIndex <= 0 ||
                faceSolids.FindFromIndex(ancestorIndex).IsEmpty())
            {
                continue;
            }
            const TopoDS_Shape solid =
                faceSolids.FindFromIndex(ancestorIndex).First();
            std::set<int> solidFaceIds;
            for (TopExp_Explorer explorer(solid, TopAbs_FACE);
                 explorer.More(); explorer.Next())
            {
                const int faceId = originalFaceMap.FindIndex(explorer.Current());
                if (faceId > 0)
                {
                    solidFaceIds.insert(faceId);
                }
            }
            if (!solidFaceIds.empty() &&
                solidFaceIds.size() <= maximumEmbeddedWindowSolidFaces)
            {
                removalExpansion[selectedId] = std::move(solidFaceIds);
                removalContainers[selectedId] = solid;
                embeddedWindowSeedFaceIds.insert(selectedId);
            }
        }
        if (embeddedWindowSeedFaceIds.empty())
        {
            std::cerr << "No predicted faces belong to a small embedded-window solid."
                      << std::endl;
            return 1;
        }

        // The visible window glass can be a face of the main fuselage solid,
        // while its frame is represented by separate small solids.  Locate
        // repeated, same-area predictions on that main solid and let OCC
        // defeaturing reconstruct the supporting fuselage skin before the
        // frame components are removed.
        std::vector<std::pair<int, double>> mainSolidPredictions;
        for (const int selectedId : selectedSurfaceFeatureFaceIds)
        {
            const TopoDS_Shape selectedFace = originalFaceMap.FindKey(selectedId);
            const int ancestorIndex = faceSolids.FindIndex(selectedFace);
            if (ancestorIndex <= 0 ||
                faceSolids.FindFromIndex(ancestorIndex).IsEmpty())
            {
                continue;
            }
            const TopoDS_Shape solid =
                faceSolids.FindFromIndex(ancestorIndex).First();
            if (CountFaces(solid) <= maximumEmbeddedWindowSolidFaces)
            {
                continue;
            }
            GProp_GProps properties;
            BRepGProp::SurfaceProperties(selectedFace, properties);
            mainSolidPredictions.emplace_back(selectedId, properties.Mass());
        }

        std::vector<std::set<int>> repeatedMainSkinGroups;
        std::vector<bool> grouped(mainSolidPredictions.size(), false);
        for (std::size_t firstIndex = 0;
             firstIndex < mainSolidPredictions.size(); ++firstIndex)
        {
            if (grouped[firstIndex])
            {
                continue;
            }
            std::set<int> group;
            const double referenceArea = mainSolidPredictions[firstIndex].second;
            for (std::size_t candidateIndex = firstIndex;
                 candidateIndex < mainSolidPredictions.size(); ++candidateIndex)
            {
                const double candidateArea = mainSolidPredictions[candidateIndex].second;
                const double tolerance = std::max(referenceArea * 0.01, 1.0e-6);
                if (std::abs(candidateArea - referenceArea) <= tolerance)
                {
                    group.insert(mainSolidPredictions[candidateIndex].first);
                    grouped[candidateIndex] = true;
                }
            }
            if (group.size() >= 3)
            {
                repeatedMainSkinGroups.push_back(std::move(group));
            }
        }

        std::set<int> mainSkinFacesToRemove;
        for (const std::set<int>& originalGroup : repeatedMainSkinGroups)
        {
            std::vector<int> pending(originalGroup.begin(), originalGroup.end());
            std::set<int> visited;
            while (!pending.empty())
            {
                const int currentFaceId = pending.back();
                pending.pop_back();
                if (!visited.insert(currentFaceId).second)
                {
                    continue;
                }
                const TopoDS_Face currentFace = TopoDS::Face(
                    originalFaceMap.FindKey(currentFaceId));
                GProp_GProps properties;
                BRepGProp::SurfaceProperties(currentFace, properties);
                if (properties.Mass() > 150000.0)
                {
                    continue;
                }
                mainSkinFacesToRemove.insert(currentFaceId);
                for (TopExp_Explorer edgeExplorer(currentFace, TopAbs_EDGE);
                     edgeExplorer.More(); edgeExplorer.Next())
                {
                    const int edgeIndex = originalEdgeFaces.FindIndex(
                        edgeExplorer.Current());
                    if (edgeIndex <= 0)
                    {
                        continue;
                    }
                    for (TopTools_ListIteratorOfListOfShape iterator(
                             originalEdgeFaces.FindFromIndex(edgeIndex));
                         iterator.More(); iterator.Next())
                    {
                        const int neighborId = originalFaceMap.FindIndex(
                            iterator.Value());
                        if (neighborId > 0 && visited.count(neighborId) == 0)
                        {
                            pending.push_back(neighborId);
                        }
                    }
                }
            }
        }

        const std::size_t maximumMainSkinRebuildFaces = std::max<std::size_t>(
            1, static_cast<std::size_t>(originalFaceMap.Extent()) / 10);
        bool useNarrowWindowRemoval = false;
        if (mainSkinFacesToRemove.size() > maximumMainSkinRebuildFaces)
        {
            std::cout << "Broad main-skin reconstruction selected "
                      << mainSkinFacesToRemove.size() << " of "
                      << originalFaceMap.Extent() << " faces; limit is "
                      << maximumMainSkinRebuildFaces
                      << ". Falling back to narrow window removal."
                      << std::endl;
            mainSkinFacesToRemove.clear();
            useNarrowWindowRemoval = true;
        }

        TopoDS_Shape rebuiltShape = inputShape;
        if (!mainSkinFacesToRemove.empty())
        {
            BRepTools_ReShape mainSkinRemover;
            for (const int faceId : mainSkinFacesToRemove)
            {
                mainSkinRemover.Remove(originalFaceMap.FindKey(faceId));
            }
            rebuiltShape = mainSkinRemover.Apply(rebuiltShape);
        }
        const int rebuiltMainSkinFaceCount = static_cast<int>(
            mainSkinFacesToRemove.size());

        // A window can be represented by two nested skins: the visible outer
        // fuselage skin and the inner cabin skin.  Matching only the selected
        // window face leaves the other skin's trimming loop visible.  Match
        // every host inner wire to the centroid of a predicted small-window
        // solid, then remove both the solid and all matched host loops.
        std::vector<gp_Pnt> windowCenters;
        std::vector<gp_Pnt> tailDecalCenters;
        std::vector<TopoDS_Shape> removableWindowSolids;
        for (const auto& entry : removalContainers)
        {
            const TopoDS_Shape& solid = entry.second;
            bool alreadyTracked = false;
            for (const TopoDS_Shape& tracked : removableWindowSolids)
            {
                if (tracked.IsSame(solid))
                {
                    alreadyTracked = true;
                    break;
                }
            }
            if (alreadyTracked)
            {
                continue;
            }
            GProp_GProps properties;
            BRepGProp::SurfaceProperties(solid, properties);
            windowCenters.push_back(properties.CentreOfMass());
            removableWindowSolids.push_back(solid);
        }
        if (useNarrowWindowRemoval && usePredictedWindowFaceAnchors)
        {
            // Some aircraft model the window patches as faces of the main
            // fuselage solid instead of separate small solids.  Their
            // predicted face centroids are the only reliable local anchors
            // for rebuilding the carrier skin and removing copied side faces.
            for (const int selectedId : selectedSurfaceFeatureFaceIds)
            {
                const TopoDS_Face selectedFace = TopoDS::Face(
                    originalFaceMap.FindKey(selectedId));
                int selectedEdgeCount = 0;
                for (TopExp_Explorer edgeExplorer(selectedFace, TopAbs_EDGE);
                     edgeExplorer.More(); edgeExplorer.Next())
                {
                    ++selectedEdgeCount;
                }
                GProp_GProps properties;
                BRepGProp::SurfaceProperties(selectedFace, properties);
                if ((!rebuildAirbusWindows && selectedEdgeCount > 2) ||
                    (rebuildAirbusWindows && properties.Mass() > 5.0))
                {
                    continue;
                }
                windowCenters.push_back(properties.CentreOfMass());
            }
        }

        Bnd_Box modelBounds;
        BRepBndLib::Add(inputShape, modelBounds);
        double hostLoopMatchDistance = 300.0;
        double modelDiagonal = 10000.0;
        double modelXMin = 0.0;
        double modelXMax = 0.0;
        if (!modelBounds.IsVoid())
        {
            double yMin = 0.0;
            double zMin = 0.0;
            double yMax = 0.0;
            double zMax = 0.0;
            modelBounds.Get(
                modelXMin, yMin, zMin, modelXMax, yMax, zMax);
            modelDiagonal = std::sqrt(
                (modelXMax - modelXMin) * (modelXMax - modelXMin) +
                (yMax - yMin) * (yMax - yMin) +
                (zMax - zMin) * (zMax - zMin));
            hostLoopMatchDistance = std::min(
                300.0, std::max(1.0e-3, modelDiagonal * 0.03));
        }
        if (useNarrowWindowRemoval && rebuildGulfstreamTailDecals &&
            modelXMax > modelXMin)
        {
            // Lettering on the vertical tail consists of complex predicted
            // faces rather than the one- or two-edge patches used by cabin
            // windows.  Restrict those anchors to the aft 30 percent of the
            // aircraft and match them with a much smaller radius below.
            const double aftThreshold =
                modelXMin + (modelXMax - modelXMin) * 0.70;
            for (const int selectedId : selectedSurfaceFeatureFaceIds)
            {
                const TopoDS_Face selectedFace = TopoDS::Face(
                    originalFaceMap.FindKey(selectedId));
                int selectedEdgeCount = 0;
                for (TopExp_Explorer edgeExplorer(selectedFace, TopAbs_EDGE);
                     edgeExplorer.More(); edgeExplorer.Next())
                {
                    ++selectedEdgeCount;
                }
                if (selectedEdgeCount <= 2)
                {
                    continue;
                }
                GProp_GProps properties;
                BRepGProp::SurfaceProperties(selectedFace, properties);
                const gp_Pnt center = properties.CentreOfMass();
                if (center.X() >= aftThreshold)
                {
                    tailDecalCenters.push_back(center);
                }
            }
        }
        std::cout << "Window-host match distance: "
                  << hostLoopMatchDistance << std::endl;
        std::cout << "Tail decal anchors: "
                  << tailDecalCenters.size() << std::endl;
        BRepTools_ReShape windowRemover;
        int rebuiltLoopCount = 0;
        int rebuiltNarrowHostFaceCount = 0;
        int residualWindowFaceCount = 0;
        TopTools_IndexedMapOfShape currentFaceMap;
        TopExp::MapShapes(rebuiltShape, TopAbs_FACE, currentFaceMap);
        for (int faceIndex = 1; faceIndex <= currentFaceMap.Extent(); ++faceIndex)
        {
            const TopoDS_Face hostFace = TopoDS::Face(
                currentFaceMap.FindKey(faceIndex));
            const TopoDS_Wire outerWire = BRepTools::OuterWire(hostFace);
            int matchedWindowLoopCount = 0;
            for (TopExp_Explorer wireExplorer(hostFace, TopAbs_WIRE);
                 wireExplorer.More(); wireExplorer.Next())
            {
                const TopoDS_Wire candidateWire = TopoDS::Wire(
                    wireExplorer.Current());
                if (candidateWire.IsSame(outerWire))
                {
                    continue;
                }
                GProp_GProps properties;
                BRepGProp::LinearProperties(candidateWire, properties);
                const gp_Pnt loopCenter = properties.CentreOfMass();
                bool isWindowHostLoop = false;
                for (const gp_Pnt& windowCenter : windowCenters)
                {
                    if (loopCenter.Distance(windowCenter) <= hostLoopMatchDistance)
                    {
                        isWindowHostLoop = true;
                        break;
                    }
                }
                if (!isWindowHostLoop)
                {
                    const double tailDecalMatchDistance =
                        hostLoopMatchDistance * 0.2;
                    for (const gp_Pnt& decalCenter : tailDecalCenters)
                    {
                        if (loopCenter.Distance(decalCenter) <=
                            tailDecalMatchDistance)
                        {
                            isWindowHostLoop = true;
                            break;
                        }
                    }
                }
                if (isWindowHostLoop)
                {
                    if (useNarrowWindowRemoval && !rebuildAirbusWindows)
                    {
                        ++matchedWindowLoopCount;
                    }
                    else
                    {
                        windowRemover.Remove(candidateWire);
                        ++rebuiltLoopCount;
                    }
                }
            }
            if (useNarrowWindowRemoval && matchedWindowLoopCount > 0)
            {
                double uMin = 0.0;
                double uMax = 0.0;
                double vMin = 0.0;
                double vMax = 0.0;
                BRepTools::UVBounds(hostFace, uMin, uMax, vMin, vMax);
                const Handle(Geom_Surface) surface = BRep_Tool::Surface(hostFace);
                if (!surface.IsNull())
                {
                    BRepBuilderAPI_MakeFace faceMaker(
                        surface, uMin, uMax, vMin, vMax, 1.0e-6);
                    if (faceMaker.IsDone())
                    {
                        windowRemover.Replace(hostFace, faceMaker.Face());
                        rebuiltLoopCount += matchedWindowLoopCount;
                        ++rebuiltNarrowHostFaceCount;
                    }
                }
            }
        }
        // STEP import may duplicate the window-band faces into another shell,
        // so their original face IDs are no longer recoverable.  Remove any
        // small local face whose centroid coincides with a predicted window;
        // this catches those duplicated F450..F477-style faces generically.
        for (int faceIndex = 1; faceIndex <= currentFaceMap.Extent(); ++faceIndex)
        {
            const TopoDS_Face candidateFace = TopoDS::Face(
                currentFaceMap.FindKey(faceIndex));
            if (rebuildAirbusWindows)
            {
                const int originalFaceId =
                    originalFaceMap.FindIndex(candidateFace);
                if (originalFaceId <= 0 ||
                    selectedSurfaceFeatureFaceIds.count(originalFaceId) == 0)
                {
                    continue;
                }
            }
            GProp_GProps properties;
            BRepGProp::SurfaceProperties(candidateFace, properties);
            const double residualMinimumArea = useNarrowWindowRemoval ?
                1.0e-9 : 10000.0;
            const double residualMaximumArea = rebuildAirbusWindows ? 5.0 :
                (useNarrowWindowRemoval ?
                 modelDiagonal * modelDiagonal * 0.0001 : 150000.0);
            if (properties.Mass() < residualMinimumArea ||
                properties.Mass() > residualMaximumArea)
            {
                continue;
            }
            const gp_Pnt faceCenter = properties.CentreOfMass();
            bool isResidualWindowFace = false;
            for (const gp_Pnt& windowCenter : windowCenters)
            {
                const double residualMatchDistance = useNarrowWindowRemoval ?
                    hostLoopMatchDistance * 0.2 : hostLoopMatchDistance;
                if (faceCenter.Distance(windowCenter) <= residualMatchDistance)
                {
                    isResidualWindowFace = true;
                    break;
                }
            }
            if (!isResidualWindowFace)
            {
                const double residualMatchDistance = useNarrowWindowRemoval ?
                    hostLoopMatchDistance * 0.2 : hostLoopMatchDistance;
                for (const gp_Pnt& decalCenter : tailDecalCenters)
                {
                    if (faceCenter.Distance(decalCenter) <=
                        residualMatchDistance)
                    {
                        isResidualWindowFace = true;
                        break;
                    }
                }
            }
            if (isResidualWindowFace)
            {
                windowRemover.Remove(candidateFace);
                ++residualWindowFaceCount;
            }
        }
        for (const TopoDS_Shape& solid : removableWindowSolids)
        {
            windowRemover.Remove(solid);
        }

        const int inputFreeEdgeCount = CountFreeEdges(inputShape);
        rebuiltShape = windowRemover.Apply(rebuiltShape);
        std::cout << "Free edges after window removal: "
                  << CountFreeEdges(rebuiltShape) << std::endl;

        // F13 is the visible fuselage skin after the window band has been
        // removed.  Its only outer wire still detours around every former
        // window.  Replace that trimmed face with the full parameter domain
        // of its own supporting surface, so the exterior is one smooth skin.
        const int freeEdgesBeforeExteriorSkinRebuild = CountFreeEdges(rebuiltShape);
        std::cout << "Free edges before exterior skin rebuild: "
                  << freeEdgesBeforeExteriorSkinRebuild << std::endl;
        int rebuiltExteriorSkinFaceCount = 0;
        TopTools_IndexedMapOfShape exteriorFaceMap;
        TopExp::MapShapes(rebuiltShape, TopAbs_FACE, exteriorFaceMap);
        BRepTools_ReShape exteriorSkinReplacer;
        for (int faceIndex = 1; faceIndex <= exteriorFaceMap.Extent(); ++faceIndex)
        {
            if (useNarrowWindowRemoval)
            {
                break;
            }
            const TopoDS_Face candidateFace = TopoDS::Face(
                exteriorFaceMap.FindKey(faceIndex));
            int exteriorEdgeCount = 0;
            for (TopExp_Explorer edgeExplorer(candidateFace, TopAbs_EDGE);
                 edgeExplorer.More(); edgeExplorer.Next())
            {
                ++exteriorEdgeCount;
            }
            if (exteriorEdgeCount < 50)
            {
                continue;
            }
            double uMin = 0.0;
            double uMax = 0.0;
            double vMin = 0.0;
            double vMax = 0.0;
            BRepTools::UVBounds(candidateFace, uMin, uMax, vMin, vMax);
            const bool hasFullAngularSpan =
                std::abs((uMax - uMin) - 2.0 * 3.14159265358979323846) < 1.0e-3;
            // Imported B-spline skins may use normalized (0..1) parameters
            // instead of a 0..2pi angular parameter.  Their window-cut
            // carriers are still identifiable by the same large area and
            // unusually dense boundary.
            if (!hasFullAngularSpan && exteriorEdgeCount < 60)
            {
                continue;
            }
            GProp_GProps properties;
            BRepGProp::SurfaceProperties(candidateFace, properties);
            // The area threshold excludes small periodic details.  A full
            // circumference face with dozens of boundary edges is an aircraft
            // skin that has been split around removed windows.
            if (properties.Mass() < 1.0e6)
            {
                continue;
            }
            const Handle(Geom_Surface) surface = BRep_Tool::Surface(candidateFace);
            BRepBuilderAPI_MakeFace faceMaker(
                surface, uMin, uMax, vMin, vMax, 1.0e-6);
            if (faceMaker.IsDone())
            {
                exteriorSkinReplacer.Replace(candidateFace, faceMaker.Face());
                ++rebuiltExteriorSkinFaceCount;
            }
        }
        if (rebuiltExteriorSkinFaceCount > 0)
        {
            const TopoDS_Shape replacement = exteriorSkinReplacer.Apply(rebuiltShape);
            if (!replacement.IsNull())
            {
                rebuiltShape = replacement;
            }
            else
            {
                rebuiltExteriorSkinFaceCount = 0;
            }
        }

        if (!rebuiltShape.IsNull() && !BRepCheck_Analyzer(rebuiltShape).IsValid())
        {
            ShapeFix_Shape fallbackFixer(rebuiltShape);
            fallbackFixer.Perform();
            const TopoDS_Shape fixedShape = fallbackFixer.Shape();
            if (!fixedShape.IsNull() &&
                BRepCheck_Analyzer(fixedShape).IsValid())
            {
                rebuiltShape = fixedShape;
            }
        }

        // Replacing a trimmed skin face with its complete supporting surface
        // creates fresh boundary edges.  Let OCC merge those edges with the
        // adjacent skin edges before the final validation; accept the sewing
        // only when it improves topology rather than hiding a regression.
        if (rebuiltExteriorSkinFaceCount > 0)
        {
            const TopoDS_Shape shapeBeforeSewing = rebuiltShape;
            try
            {
                ShapeFix_Shape fixer(rebuiltShape);
                fixer.Perform();
                const TopoDS_Shape fixedShape = fixer.Shape();
                if (!fixedShape.IsNull() &&
                    BRepCheck_Analyzer(fixedShape).IsValid() &&
                    CountFreeEdges(fixedShape) <= CountFreeEdges(shapeBeforeSewing))
                {
                    rebuiltShape = fixedShape;
                }

                BRepBuilderAPI_Sewing sewing(
                    1.0e-4, Standard_True, Standard_True,
                    Standard_True, Standard_False);
                sewing.Add(rebuiltShape);
                sewing.Perform();
                const TopoDS_Shape sewnShape = sewing.SewedShape();
                if (!sewnShape.IsNull() &&
                    BRepCheck_Analyzer(sewnShape).IsValid() &&
                    CountFreeEdges(sewnShape) <= CountFreeEdges(shapeBeforeSewing))
                {
                    rebuiltShape = sewnShape;
                }
                else
                {
                    rebuiltShape = shapeBeforeSewing;
                }
            }
            catch (const Standard_Failure&)
            {
                rebuiltShape = shapeBeforeSewing;
            }
        }

        int outputFreeEdgeCount = CountFreeEdges(rebuiltShape);
        const bool rebuiltValid = !rebuiltShape.IsNull() &&
            BRepCheck_Analyzer(rebuiltShape).IsValid();
        if (rebuiltLoopCount <= 0 || rebuiltShape.IsNull())
        {
            std::cerr << "Embedded-window surface reconstruction produced no "
                      << "usable shape; no candidate STEP was written. Free edges: "
                      << inputFreeEdgeCount << " -> " << outputFreeEdgeCount
                      << std::endl;
            return 1;
        }

        const TopoDS_Shape shapeBeforeUnify = rebuiltShape;
        ShapeUpgrade_UnifySameDomain unify(
            rebuiltShape, Standard_True, Standard_True, Standard_True);
        unify.Build();
        if (!unify.Shape().IsNull() &&
            BRepCheck_Analyzer(unify.Shape()).IsValid() &&
            CountFreeEdges(unify.Shape()) <= inputFreeEdgeCount)
        {
            rebuiltShape = unify.Shape();
            outputFreeEdgeCount = CountFreeEdges(rebuiltShape);
        }
        else
        {
            rebuiltShape = shapeBeforeUnify;
        }

        const std::filesystem::path outputPath(outputFile);
        if (outputPath.has_parent_path())
        {
            std::filesystem::create_directories(outputPath.parent_path());
        }
        if (!SaveStep(rebuiltShape, outputFile))
        {
            std::cerr << "Failed to write embedded-window repair candidate."
                      << std::endl;
            return 1;
        }

        std::cout << "Embedded-window host loops rebuilt: "
                  << rebuiltLoopCount << std::endl;
        std::cout << "Narrow-mode host skin faces rebuilt: "
                  << rebuiltNarrowHostFaceCount << std::endl;
        std::cout << "Residual window faces removed: "
                  << residualWindowFaceCount << std::endl;
        std::cout << "Repeated main-skin window faces rebuilt: "
                  << rebuiltMainSkinFaceCount << std::endl;
        std::cout << "Full exterior skin faces rebuilt: "
                  << rebuiltExteriorSkinFaceCount << std::endl;
        std::cout << "BRep valid: " << (rebuiltValid ? "yes" : "no") << std::endl;
        std::cout << "Free edges: " << inputFreeEdgeCount
                  << " -> " << outputFreeEdgeCount << std::endl;
        std::cout << "Face count: " << originalFaceMap.Extent()
                  << " -> " << CountFaces(rebuiltShape) << std::endl;
        std::cout << "Embedded-window repair candidate: "
                  << outputFile << std::endl;
        return 0;
    }
    catch (const Standard_Failure& error)
    {
        std::cerr << "Embedded-window repair failed in OpenCASCADE: "
                  << error.GetMessageString() << std::endl;
        return 1;
    }
}

namespace
{
    bool IsStepFile(const std::filesystem::path& path)
    {
        const std::string extension = path.extension().string();
        return extension == ".step" || extension == ".STEP" ||
               extension == ".stp" || extension == ".STP";
    }

    std::filesystem::path FindBatchPrediction(
        const std::filesystem::path& predictionsDir,
        const std::string& stem)
    {
        const std::vector<std::filesystem::path> candidates = {
            predictionsDir / (stem + ".pred.csv"),
            predictionsDir / (stem + "_pred.csv"),
            predictionsDir / (stem + ".csv")};
        for (const auto& candidate : candidates)
        {
            if (std::filesystem::is_regular_file(candidate))
            {
                return candidate;
            }
        }
        return {};
    }
}

int RunBatchEmbeddedWindowHostRebuild(
    const std::string& inputDir,
    const std::string& predictionsDir,
    const std::string& outputDir)
{
    const std::filesystem::path inputPath(inputDir);
    const std::filesystem::path predictionPath(predictionsDir);
    const std::filesystem::path outputPath(outputDir);
    if (!std::filesystem::is_directory(inputPath) ||
        !std::filesystem::is_directory(predictionPath))
    {
        std::cerr << "Batch input and prediction directories must exist." << std::endl;
        return 1;
    }
    std::filesystem::create_directories(outputPath);

    std::vector<std::filesystem::path> inputFiles;
    for (const auto& entry : std::filesystem::directory_iterator(inputPath))
    {
        if (entry.is_regular_file() && IsStepFile(entry.path()))
        {
            inputFiles.push_back(entry.path());
        }
    }
    std::sort(inputFiles.begin(), inputFiles.end());
    if (inputFiles.empty())
    {
        std::cerr << "No STEP files found in batch input directory." << std::endl;
        return 1;
    }

    int failureCount = 0;
    for (const auto& inputFile : inputFiles)
    {
        const auto predictionFile = FindBatchPrediction(
            predictionPath, inputFile.stem().string());
        if (predictionFile.empty())
        {
            std::cerr << "Missing prediction CSV for " << inputFile.filename()
                      << std::endl;
            ++failureCount;
            continue;
        }
        const auto outputFile = outputPath / inputFile.filename();
        std::cout << "[batch] " << inputFile.filename() << std::endl;
        if (RunEmbeddedWindowHostRebuild(
                inputFile.string(), predictionFile.string(),
                outputFile.string(), "auto") != 0)
        {
            ++failureCount;
        }
    }
    std::cout << "Batch embedded-window rebuild complete: "
              << (inputFiles.size() - failureCount) << "/"
              << inputFiles.size() << " succeeded." << std::endl;
    return failureCount == 0 ? 0 : 1;
}
