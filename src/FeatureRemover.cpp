#include "FeatureRemover.h"

#include <BRepAlgoAPI_Defeaturing.hxx>
#include <BRepBuilderAPI_MakeFace.hxx>
#include <BRepCheck_Analyzer.hxx>
#include <BRepGProp.hxx>
#include <BRepTools.hxx>
#include <BRepTools_ReShape.hxx>
#include <BRep_Tool.hxx>
#include <GProp_GProps.hxx>
#include <IFSelect_ReturnStatus.hxx>
#include <ShapeFix_Shape.hxx>
#include <ShapeUpgrade_UnifySameDomain.hxx>
#include <STEPControl_Reader.hxx>
#include <STEPControl_Writer.hxx>
#include <TopAbs_ShapeEnum.hxx>
#include <TopExp.hxx>
#include <TopExp_Explorer.hxx>
#include <TopoDS.hxx>
#include <TopoDS_Shape.hxx>
#include <TopTools_IndexedMapOfShape.hxx>
#include <TopTools_IndexedDataMapOfShapeListOfShape.hxx>
#include <TopTools_ListOfShape.hxx>
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
    };

    PredictionSelection ReadPredictedRivetFaceIds(const std::string& predictionsFile)
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
                if (std::stoi(fields[predictionColumn]) == 1)
                {
                    selection.rivetFaceIds.insert(faceId);
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
        const std::set<int>& seedFaceIds)
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
        const double maximumConnectedFaceArea = std::max(maximumSeedArea * 4.0, 1.0e-9);

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
                    if (adjacentArea <= maximumConnectedFaceArea)
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

        const PredictionSelection predictions = ReadPredictedRivetFaceIds(predictionsFile);
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
    catch (const std::exception& error)
    {
        std::cerr << "Predicted rivet removal failed: " << error.what() << std::endl;
        return 1;
    }
}
