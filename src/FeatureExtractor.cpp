#include "FeatureExtractor.h"
#include <TopExp_Explorer.hxx>
#include <BRepTools.hxx>
#include <TopExp.hxx>
#include <TopoDS.hxx>
#include <TopoDS_Face.hxx>
#include <TopoDS_Edge.hxx>
#include <TopoDS_Wire.hxx>
#include <TopoDS_Vertex.hxx>
#include <BRepGProp.hxx>
#include <GProp_GProps.hxx>
#include <TopTools_IndexedMapOfShape.hxx> 
#include <BRep_Tool.hxx>
#include <BRepClass_FaceClassifier.hxx>
#include <BRepLProp_SLProps.hxx>
#include <BRepAdaptor_Curve.hxx>
#include <BRepAdaptor_Surface.hxx>
#include <Geom_Surface.hxx>
#include <gp_Pnt2d.hxx>
#include <gp_Dir.hxx>
#include <gp_Pnt.hxx>
#include <gp_Cylinder.hxx>
#include <gp_Torus.hxx>
#include <gp_Sphere.hxx>
#include <gp_Cone.hxx>
#include <Standard_Real.hxx>
#include <Precision.hxx>
#include <cmath>
#include <algorithm>
#include <functional>
#include <iomanip>
#include <limits>
#include <map>
#include <sstream>

namespace {

struct SampledSurfaceProperties
{
    bool hasNormal = false;
    bool hasCurvature = false;
    gp_Dir normal;
    double meanCurvature = 0.0;
};

bool IsUvInsideFace(const TopoDS_Face& face, double u, double v)
{
    BRepClass_FaceClassifier classifier;
    classifier.Perform(face, gp_Pnt2d(u, v), Precision::Confusion());
    const TopAbs_State state = classifier.State();
    return state == TopAbs_IN || state == TopAbs_ON;
}

SampledSurfaceProperties SampleSurfacePropertiesInsideFace(
    const TopoDS_Face& face,
    BRepAdaptor_Surface& surfaceAdaptor,
    double uMin,
    double uMax,
    double vMin,
    double vMax
)
{
    SampledSurfaceProperties sampled;
    gp_Vec accumulatedNormal(0.0, 0.0, 0.0);
    gp_Dir referenceNormal;
    bool hasReferenceNormal = false;
    double accumulatedCurvature = 0.0;
    int curvatureSamples = 0;

    constexpr int sampleCountU = 5;
    constexpr int sampleCountV = 5;
    for (int uIndex = 0; uIndex < sampleCountU; ++uIndex) {
        const double uFraction = static_cast<double>(uIndex + 1) / static_cast<double>(sampleCountU + 1);
        const double u = uMin + (uMax - uMin) * uFraction;
        for (int vIndex = 0; vIndex < sampleCountV; ++vIndex) {
            const double vFraction = static_cast<double>(vIndex + 1) / static_cast<double>(sampleCountV + 1);
            const double v = vMin + (vMax - vMin) * vFraction;
            if (!IsUvInsideFace(face, u, v)) {
                continue;
            }

            BRepLProp_SLProps props(surfaceAdaptor, u, v, 2, Precision::Confusion());
            if (props.IsNormalDefined()) {
                gp_Dir normal = props.Normal();
                if (face.Orientation() == TopAbs_REVERSED) {
                    normal.Reverse();
                }
                if (!hasReferenceNormal) {
                    referenceNormal = normal;
                    hasReferenceNormal = true;
                } else if (normal.Dot(referenceNormal) < 0.0) {
                    normal.Reverse();
                }
                accumulatedNormal += gp_Vec(normal);
                sampled.hasNormal = true;
            }

            if (props.IsCurvatureDefined()) {
                accumulatedCurvature += props.MeanCurvature();
                curvatureSamples++;
                sampled.hasCurvature = true;
            }
        }
    }

    if (sampled.hasNormal && accumulatedNormal.SquareMagnitude() > Precision::SquareConfusion()) {
        sampled.normal = gp_Dir(accumulatedNormal);
    } else {
        sampled.hasNormal = false;
    }

    if (sampled.hasCurvature && curvatureSamples > 0) {
        sampled.meanCurvature = accumulatedCurvature / static_cast<double>(curvatureSamples);
    } else {
        sampled.hasCurvature = false;
    }

    return sampled;
}

}

FeatureExtractor::FeatureExtractor(const TopoDS_Shape &shape) : myShape(shape) {}

std::string FeatureExtractor::BuildFaceKey(
    const FaceFeature& feature,
    const std::vector<FaceFeature>& allFeatures
) const
{
    std::ostringstream stream;
    stream << std::fixed << std::setprecision(6)
           << "surface=" << feature.surfaceType
           << "|area=" << feature.area
           << "|perimeter=" << feature.perimeter
           << "|center=(" << feature.centerX << "," << feature.centerY << "," << feature.centerZ << ")"
           << "|normal=(" << feature.normalX << "," << feature.normalY << "," << feature.normalZ << ")"
           << "|curvature=" << feature.meanCurvature
           << "|radius=" << feature.radius
           << "|wires=" << feature.numWires
           << "|inner_wires=" << feature.innerWireCount
           << "|edges=" << feature.numEdges;

    stream << "|inner_lengths=";
    for (size_t index = 0; index < feature.innerWireLengths.size(); ++index) {
        if (index != 0) {
            stream << ",";
        }
        stream << feature.innerWireLengths[index];
    }

    std::vector<std::string> neighborDescriptors;
    neighborDescriptors.reserve(feature.neighborIds.size());
    for (size_t index = 0; index < feature.neighborIds.size(); ++index) {
        const int neighborId = feature.neighborIds[index];
        const int edgeType = index < feature.neighborEdgeTypes.size() ? feature.neighborEdgeTypes[index] : 0;
        if (neighborId <= 0 || neighborId > static_cast<int>(allFeatures.size())) {
            neighborDescriptors.push_back("missing");
            continue;
        }

        const auto& neighbor = allFeatures[neighborId - 1];
        std::ostringstream neighborStream;
        neighborStream << std::fixed << std::setprecision(6)
                       << "s=" << neighbor.surfaceType
                       << ";a=" << neighbor.area
                       << ";p=" << neighbor.perimeter
                       << ";e=" << edgeType
                       << ";n=" << neighbor.numEdges
                       << ";w=" << neighbor.numWires
                       << ";iw=" << neighbor.innerWireCount;
        neighborDescriptors.push_back(neighborStream.str());
    }

    std::sort(neighborDescriptors.begin(), neighborDescriptors.end());
    stream << "|neighbors=";
    for (size_t index = 0; index < neighborDescriptors.size(); ++index) {
        if (index != 0) {
            stream << "|";
        }
        stream << neighborDescriptors[index];
    }

    return stream.str();
}

void FeatureExtractor::Extract()
{
    // 1. 鍏堝缓绔嬩竴涓?Face 鍒?Index 鐨勬槧灏勮〃锛岃繖鏍锋垜浠墠鑳介€氳繃 Shape 鍙嶆煡 ID
    // TopExp::MapShapes 浼氭寜鐓ч亶鍘嗛『搴忕粰姣忎釜 Face 缂栦竴涓彿 (1 鍒?Extent)
    TopTools_IndexedMapOfShape faceMap;
    TopExp::MapShapes(myShape, TopAbs_FACE, faceMap);

    // 2. 寤虹珛杈瑰埌闈㈢殑鈥滃€掓煡琛ㄢ€濓紝閫氳繃杈规壘閭诲眳闈?
    TopExp::MapShapesAndAncestors(myShape, TopAbs_EDGE, TopAbs_FACE, myEdgeFaceMap);

    // 3. 寮€濮嬭绠楀嚑浣曞睘鎬у拰閭诲眳鍏崇郴
    ComputeGeometricAttributes(faceMap);
}

void FeatureExtractor::ComputeGeometricAttributes(const TopTools_IndexedMapOfShape &faceMap)
{
    myResults.clear();

    // 1. 绗竴閬嶉亶鍘嗭細璁＄畻鎬婚潰绉?
    double totalArea = 0.0;
    for (int i = 1; i <= faceMap.Extent(); ++i)
    {
        TopoDS_Face face = TopoDS::Face(faceMap.FindKey(i));
        GProp_GProps areaProps;
        BRepGProp::SurfaceProperties(face, areaProps);
        totalArea += areaProps.Mass();
    }

    // 2. 绗簩閬嶉亶鍘嗭細鎻愬彇璇︾粏鐗瑰緛
    for (int i = 1; i <= faceMap.Extent(); ++i)
    {
        TopoDS_Face face = TopoDS::Face(faceMap.FindKey(i));
        FaceFeature feat;
        feat.id = i;

        // --- 鍑犱綍鐗瑰緛鎻愬彇 ---
        GProp_GProps areaProps, lineProps;
        BRepGProp::SurfaceProperties(face, areaProps);
        BRepGProp::LinearProperties(face, lineProps);

        feat.area = areaProps.Mass();
        feat.relativeArea = (totalArea > 1e-6) ? (feat.area / totalArea) : 0.0;
        feat.perimeter = lineProps.Mass();

        // 璁＄畻绱ц嚧搴?
        if (feat.area > 1e-6)
        {
            feat.compactness = (feat.perimeter * feat.perimeter) / (4.0 * M_PI * feat.area);
        }
        else
        {
            feat.compactness = 999.0;
        }

        // 鎻愬彇琛ㄩ潰绫诲瀷鍜屾硶鍚戜互鍙婁腑蹇?Z
        GProp_GProps gprops;
        BRepGProp::SurfaceProperties(face, gprops);
        gp_Pnt center = gprops.CentreOfMass();
        feat.centerX = center.X();
        feat.centerY = center.Y();
        feat.centerZ = center.Z();

        // 鑾峰彇闈㈢被鍨?
        BRepAdaptor_Surface surf(face);
        feat.surfaceType = surf.GetType();

        // 鎻愬彇鍗婂緞鐗瑰緛 (鍏抽敭锛氱敤浜庤瘑鍒渾瑙?鍊掕)
        feat.radius = 0.0;
        if (feat.surfaceType == GeomAbs_Cylinder) {
            feat.radius = surf.Cylinder().Radius();
        } else if (feat.surfaceType == GeomAbs_Torus) {
            feat.radius = surf.Torus().MinorRadius(); // 鍦嗙幆闈㈢殑灏忓崐寰勯€氬父瀵瑰簲鍦嗚鍗婂緞
        } else if (feat.surfaceType == GeomAbs_Sphere) {
            feat.radius = surf.Sphere().Radius();
        } else if (feat.surfaceType == GeomAbs_Cone) {
            feat.radius = surf.Cone().RefRadius();
        }

        // 鎻愬彇闈腑蹇冩硶鍚?
                double u_min, u_max, v_min, v_max;
        BRepTools::UVBounds(face, u_min, u_max, v_min, v_max);
        BRepAdaptor_Surface sAtor(face);
        const SampledSurfaceProperties sampledProps =
            SampleSurfacePropertiesInsideFace(face, sAtor, u_min, u_max, v_min, v_max);
        if (sampledProps.hasNormal) {
            feat.normalX = sampledProps.normal.X();
            feat.normalY = sampledProps.normal.Y();
            feat.normalZ = sampledProps.normal.Z();
        } else {
            feat.normalX = 0;
            feat.normalY = 0;
            feat.normalZ = 1.0;
        }

        if (sampledProps.hasCurvature) {
            feat.meanCurvature = sampledProps.meanCurvature;
        } else {
            feat.meanCurvature = 0.0;
        }

        // 鎻愬彇鎷撴墤澶嶆潅搴︾壒寰?
        int wireCount = 0;
        std::vector<double> wireLengths;
        TopExp_Explorer wireExp(face, TopAbs_WIRE);
        for (; wireExp.More(); wireExp.Next()) {
            wireCount++;

            const TopoDS_Wire wire = TopoDS::Wire(wireExp.Current());
            GProp_GProps wireProps;
            BRepGProp::LinearProperties(wire, wireProps);
            wireLengths.push_back(wireProps.Mass());
        }
        feat.numWires = wireCount;
        feat.innerWireCount = 0;
        feat.minInnerWireLength = 0.0;
        feat.maxInnerWireLength = 0.0;
        feat.innerWireLengths.clear();

        if (wireLengths.size() > 1) {
            struct InnerWireRecord {
                double length;
            };

            std::vector<InnerWireRecord> innerWireRecords;
            innerWireRecords.reserve(wireLengths.size());

            TopExp_Explorer innerWireExp(face, TopAbs_WIRE);
            for (; innerWireExp.More(); innerWireExp.Next()) {
                const TopoDS_Wire wire = TopoDS::Wire(innerWireExp.Current());
                GProp_GProps wireProps;
                BRepGProp::LinearProperties(wire, wireProps);
                innerWireRecords.push_back({
                    wireProps.Mass(),
                });
            }

            std::sort(
                innerWireRecords.begin(),
                innerWireRecords.end(),
                [](const InnerWireRecord& lhs, const InnerWireRecord& rhs) {
                    return lhs.length > rhs.length;
                }
            );
            feat.innerWireCount = static_cast<int>(wireLengths.size()) - 1;

            double minInnerWireLength = std::numeric_limits<double>::max();
            double maxInnerWireLength = 0.0;
            for (size_t wireIdx = 1; wireIdx < innerWireRecords.size(); ++wireIdx) {
                minInnerWireLength = std::min(minInnerWireLength, innerWireRecords[wireIdx].length);
                maxInnerWireLength = std::max(maxInnerWireLength, innerWireRecords[wireIdx].length);
                feat.innerWireLengths.push_back(innerWireRecords[wireIdx].length);
            }

            feat.minInnerWireLength = minInnerWireLength;
            feat.maxInnerWireLength = maxInnerWireLength;
        }

        int edgeCount = 0;
        TopExp_Explorer edgeCountExp(face, TopAbs_EDGE);
        for (; edgeCountExp.More(); edgeCountExp.Next()) edgeCount++;
        feat.numEdges = edgeCount;

        // 鎷撴墤鐗瑰緛鎻愬彇锛氬鎵鹃偦灞?ID
        TopExp_Explorer edgeExp(face, TopAbs_EDGE);
        for (; edgeExp.More(); edgeExp.Next())
        {
            const TopoDS_Shape &edge = edgeExp.Current();
            if (myEdgeFaceMap.Contains(edge))
            {
                const TopTools_ListOfShape &neighborFaces = myEdgeFaceMap.FindFromKey(edge);
                TopTools_ListIteratorOfListOfShape it(neighborFaces);
                for (; it.More(); it.Next())
                {
                    const TopoDS_Shape &neighborShape = it.Value();

                    if (!neighborShape.IsSame(face))
                    {
                        int neighborId = faceMap.FindIndex(neighborShape);
                        TopoDS_Face neighborFace = TopoDS::Face(neighborShape);
                        TopoDS_Edge sharedEdge = TopoDS::Edge(edge);

                        int rawEdgeType = IdentifyEdgeType(face, neighborFace, sharedEdge);
                        
                        // 鏄犲皠涓?ML 鍙嬪ソ鏁板€硷細Convex=1.0, Concave=-1.0, Smooth=0.0
                        int mlEdgeType = 0;
                        if (rawEdgeType == CONVEX) mlEdgeType = 1;
                        else if (rawEdgeType == CONCAVE) mlEdgeType = -1;
                        else mlEdgeType = 0;

                        auto it_nb = std::find(feat.neighborIds.begin(), feat.neighborIds.end(), neighborId);
                        if (it_nb == feat.neighborIds.end())
                        {
                            GProp_GProps neighborAreaProps;
                            BRepGProp::SurfaceProperties(neighborFace, neighborAreaProps);
                            const double neighborArea = neighborAreaProps.Mass();

                            BRepAdaptor_Surface neighborSurface(neighborFace);

                            GProp_GProps sharedEdgeProps;
                            BRepGProp::LinearProperties(sharedEdge, sharedEdgeProps);

                            double dihedralMean = 0.0;
                            double dihedralStd = 0.0;
                            ComputeSampledDihedralStats(
                                face,
                                neighborFace,
                                sharedEdge,
                                dihedralMean,
                                dihedralStd
                            );

                            feat.neighborIds.push_back(neighborId);
                            feat.neighborEdgeTypes.push_back(mlEdgeType);
                            feat.neighborAreaRatios.push_back(
                                neighborArea > 1.0e-12 ? feat.area / neighborArea : 0.0
                            );
                            feat.neighborSurfaceTypes.push_back(neighborSurface.GetType());
                            feat.sharedEdgeLengths.push_back(sharedEdgeProps.Mass());
                            feat.neighborDihedralMeans.push_back(dihedralMean);
                            feat.neighborDihedralStds.push_back(dihedralStd);
                            if (mlEdgeType > 0) {
                                feat.convexEdgeCount++;
                            } else if (mlEdgeType < 0) {
                                feat.concaveEdgeCount++;
                            } else {
                                feat.smoothEdgeCount++;
                            }
                        }
                    }
                }
            }
        }
        myResults.push_back(feat);
    }

    for (auto& feature : myResults) {
        double neighborAreaSum = 0.0;
        double normalDotSum = 0.0;
        int validNeighborCount = 0;
        int validNormalDotCount = 0;
        feature.neighborAreaMax = 0.0;
        feature.normalNeighborDotMean = 0.0;
        feature.normalNeighborDotMin = 0.0;
        feature.normalNeighborDotMax = 0.0;
        feature.neighborPlaneCount = 0;
        feature.neighborCylinderCount = 0;
        feature.neighborCurvedCount = 0;

        for (const int neighborId : feature.neighborIds) {
            if (neighborId <= 0 || neighborId > static_cast<int>(myResults.size())) {
                continue;
            }

            const auto& neighbor = myResults[static_cast<size_t>(neighborId - 1)];
            neighborAreaSum += neighbor.area;
            feature.neighborAreaMax = std::max(feature.neighborAreaMax, neighbor.area);
            validNeighborCount++;

            const double normalDot = std::abs(
                feature.normalX * neighbor.normalX +
                feature.normalY * neighbor.normalY +
                feature.normalZ * neighbor.normalZ
            );
            const double clampedNormalDot = std::clamp(normalDot, 0.0, 1.0);
            normalDotSum += clampedNormalDot;
            if (validNormalDotCount == 0) {
                feature.normalNeighborDotMin = clampedNormalDot;
                feature.normalNeighborDotMax = clampedNormalDot;
            } else {
                feature.normalNeighborDotMin = std::min(feature.normalNeighborDotMin, clampedNormalDot);
                feature.normalNeighborDotMax = std::max(feature.normalNeighborDotMax, clampedNormalDot);
            }
            validNormalDotCount++;

            if (neighbor.surfaceType == GeomAbs_Plane) {
                feature.neighborPlaneCount++;
            } else if (neighbor.surfaceType == GeomAbs_Cylinder) {
                feature.neighborCylinderCount++;
                feature.neighborCurvedCount++;
            } else {
                feature.neighborCurvedCount++;
            }
        }

        if (validNeighborCount > 0) {
            feature.neighborAreaMean = neighborAreaSum / static_cast<double>(validNeighborCount);
            feature.areaToNeighborMean =
                feature.neighborAreaMean > 1.0e-12 ? feature.area / feature.neighborAreaMean : 0.0;
            feature.areaToNeighborMax =
                feature.neighborAreaMax > 1.0e-12 ? feature.area / feature.neighborAreaMax : 0.0;
        }
        if (validNormalDotCount > 0) {
            feature.normalNeighborDotMean = normalDotSum / static_cast<double>(validNormalDotCount);
        }

        const int typedEdgeCount = feature.convexEdgeCount + feature.concaveEdgeCount + feature.smoothEdgeCount;
        if (typedEdgeCount > 0) {
            feature.convexEdgeRatio = static_cast<double>(feature.convexEdgeCount) / static_cast<double>(typedEdgeCount);
            feature.concaveEdgeRatio = static_cast<double>(feature.concaveEdgeCount) / static_cast<double>(typedEdgeCount);
        }

        feature.faceKey = BuildFaceKey(feature, myResults);
    }

    std::map<std::string, std::vector<size_t>> duplicateGroups;
    for (size_t index = 0; index < myResults.size(); ++index) {
        duplicateGroups[myResults[index].faceKey].push_back(index);
    }

    for (const auto& [baseKey, indices] : duplicateGroups) {
        if (indices.size() <= 1) {
            continue;
        }

        std::vector<size_t> sortedIndices = indices;
        std::sort(
            sortedIndices.begin(),
            sortedIndices.end(),
            [&](size_t lhs, size_t rhs) {
                return myResults[lhs].id < myResults[rhs].id;
            }
        );

        for (size_t rank = 0; rank < sortedIndices.size(); ++rank) {
            auto& feature = myResults[sortedIndices[rank]];
            feature.faceKey += "|dup_rank=" + std::to_string(rank);
        }
    }
}

int FeatureExtractor::IdentifyEdgeType(const TopoDS_Face& f1, const TopoDS_Face& f2, const TopoDS_Edge& e)
{
    // 鑾峰彇杈逛笂鐨勪腑闂村弬鏁?
    Standard_Real first, last;
    BRepAdaptor_Curve cAtor(e);
    first = cAtor.FirstParameter();
    last = cAtor.LastParameter();
    Standard_Real mid = (first + last) / 2.0;

    // 鑾峰彇璇ョ偣鍧愭爣
    gp_Pnt pMid;
    gp_Vec vTangent;
    cAtor.D1(mid, pMid, vTangent);

    // 鑾峰彇涓や釜闈㈢殑娉曞悜閲?
    auto getNormalAndRefVec = [&](const TopoDS_Face& f, const TopoDS_Edge& edge, double param, gp_Dir& normal, gp_Vec& binormal) -> bool {
        BRepAdaptor_Surface sAtor(f);
        BRepAdaptor_Curve cAtor(edge);
        
        // 鎶曞奖鐐瑰埌闈㈣幏鍙?UV (鎴栬€呴€氳繃閲囨牱鐐硅幏鍙?
        gp_Pnt p; gp_Vec tangent;
        cAtor.D1(param, p, tangent);
        
        // 杩欓噷鐨勯€昏緫闇€瑕佸鐞?edge 鍦?face 涓殑鍙傛暟
        Standard_Real u, v;
        Handle(Geom2d_Curve) c2d = BRep_Tool::CurveOnSurface(edge, f, u, v);
        if (c2d.IsNull()) return false;
        gp_Pnt2d uv = c2d->Value(param);
        
        BRepLProp_SLProps props(sAtor, uv.X(), uv.Y(), 1, Precision::Confusion());
        if (!props.IsNormalDefined()) return false;
        
        normal = props.Normal();
        if (f.Orientation() == TopAbs_REVERSED) normal.Reverse();
        
        // 璁＄畻闈㈠唴鍚戦噺 (Binormal pointing INTO the face)
        // 璁＄畻瑙勫垯锛歊 = Normal x Tangent
        // 濡傛灉杈瑰湪闈腑鐨?Orientation 鏄?REVERSED锛屽垯 Tangent 闇€瑕佸弽鍚?
        gp_Vec tVec = tangent;
        TopAbs_Orientation edgeOri = TopAbs_FORWARD;
        
        // 瀵绘壘杈瑰湪闈腑鐨勫疄闄?Orientation
        TopExp_Explorer exp(f, TopAbs_EDGE);
        for (; exp.More(); exp.Next()) {
            if (exp.Current().IsSame(edge)) {
                edgeOri = exp.Current().Orientation();
                break;
            }
        }
        
        if (edgeOri == TopAbs_REVERSED) tVec.Reverse();
        
        binormal = normal.XYZ() ^ tVec.XYZ();
        binormal.Normalize();
        return true;
    };

    gp_Dir n1, n2;
    gp_Vec r1;
    if (!getNormalAndRefVec(f1, e, mid, n1, r1)) return OTHER;
    
    // 鑾峰彇 f2 鐨勬硶鍚?n2
    BRepAdaptor_Surface sAtor2(f2);
    Standard_Real u2, v2;
    Handle(Geom2d_Curve) c2d2 = BRep_Tool::CurveOnSurface(e, f2, u2, v2);
    if (c2d2.IsNull()) return OTHER;
    gp_Pnt2d uv2 = c2d2->Value(mid);
    BRepLProp_SLProps props2(sAtor2, uv2.X(), uv2.Y(), 1, Precision::Confusion());
    if (!props2.IsNormalDefined()) return OTHER;
    n2 = props2.Normal();
    if (f2.Orientation() == TopAbs_REVERSED) n2.Reverse();

    // 璁＄畻浜岄潰瑙掔偣绉?
    double dot = n1.Dot(n2);
    if (dot > 0.999) return SMOOTH;

    // 鍑瑰嚫鎬ф牳蹇冨垽鏂細
    // 濡傛灉闈?f2 鐨勬硶鍚?n2 涓庨潰 f1 鐨勫悜鍐呭悜閲?r1 鐨勭偣绉负姝?
    // 璇存槑 f2 鍚?f1 鐨勨€滃唴閮ㄢ€濆亸杞?-> 鍑?(Concave)
    // 鍙嶄箣 -> 鍑?(Convex)
    double check = n2.Dot(r1);
    
    if (check > 1e-6) return CONCAVE;
    if (check < -1e-6) return CONVEX;
    
    return SMOOTH;
}

bool FeatureExtractor::ComputeSampledDihedralStats(
    const TopoDS_Face& f1,
    const TopoDS_Face& f2,
    const TopoDS_Edge& e,
    double& meanAngle,
    double& stdAngle)
{
    meanAngle = 0.0;
    stdAngle = 0.0;

    Standard_Real first = 0.0;
    Standard_Real last = 0.0;
    BRepAdaptor_Curve curveAdaptor(e);
    first = curveAdaptor.FirstParameter();
    last = curveAdaptor.LastParameter();
    if (!std::isfinite(first) || !std::isfinite(last) || std::abs(last - first) <= Precision::Confusion()) {
        return false;
    }

    Standard_Real u1 = 0.0;
    Standard_Real v1 = 0.0;
    Standard_Real u2 = 0.0;
    Standard_Real v2 = 0.0;
    Handle(Geom2d_Curve) curveOnFirstFace = BRep_Tool::CurveOnSurface(e, f1, u1, v1);
    Handle(Geom2d_Curve) curveOnSecondFace = BRep_Tool::CurveOnSurface(e, f2, u2, v2);
    if (curveOnFirstFace.IsNull() || curveOnSecondFace.IsNull()) {
        return false;
    }

    BRepAdaptor_Surface firstSurface(f1);
    BRepAdaptor_Surface secondSurface(f2);
    std::vector<double> angles;
    angles.reserve(10);
    constexpr int kSampleCount = 10;
    constexpr double kStartFraction = 0.05;
    constexpr double kEndFraction = 0.95;

    for (int sampleIndex = 0; sampleIndex < kSampleCount; ++sampleIndex) {
        const double fraction = kStartFraction +
            (kEndFraction - kStartFraction) * static_cast<double>(sampleIndex) /
            static_cast<double>(kSampleCount - 1);
        const Standard_Real parameter = first + (last - first) * fraction;
        const gp_Pnt2d uv1 = curveOnFirstFace->Value(parameter);
        const gp_Pnt2d uv2 = curveOnSecondFace->Value(parameter);
        BRepLProp_SLProps firstProps(
            firstSurface, uv1.X(), uv1.Y(), 1, Precision::Confusion());
        BRepLProp_SLProps secondProps(
            secondSurface, uv2.X(), uv2.Y(), 1, Precision::Confusion());
        if (!firstProps.IsNormalDefined() || !secondProps.IsNormalDefined()) {
            continue;
        }

        gp_Dir firstNormal = firstProps.Normal();
        gp_Dir secondNormal = secondProps.Normal();
        if (f1.Orientation() == TopAbs_REVERSED) {
            firstNormal.Reverse();
        }
        if (f2.Orientation() == TopAbs_REVERSED) {
            secondNormal.Reverse();
        }
        const double dot = std::clamp(firstNormal.Dot(secondNormal), -1.0, 1.0);
        angles.push_back(std::acos(dot));
    }

    if (angles.size() < 3) {
        return false;
    }

    for (const double angle : angles) {
        meanAngle += angle;
    }
    meanAngle /= static_cast<double>(angles.size());
    for (const double angle : angles) {
        const double delta = angle - meanAngle;
        stdAngle += delta * delta;
    }
    stdAngle = std::sqrt(stdAngle / static_cast<double>(angles.size()));
    return true;
}
