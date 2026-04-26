#include "FeatureExtractor.h"
#include <TopExp_Explorer.hxx>
#include <BRepTools.hxx>
#include <TopExp.hxx>
#include <TopoDS.hxx>
#include <TopoDS_Face.hxx>
#include <TopoDS_Edge.hxx>
#include <TopoDS_Vertex.hxx>
#include <BRepGProp.hxx>
#include <GProp_GProps.hxx>
#include <TopTools_IndexedMapOfShape.hxx> 
#include <BRep_Tool.hxx>
#include <BRepLProp_SLProps.hxx>
#include <BRepAdaptor_Curve.hxx>
#include <BRepAdaptor_Surface.hxx>
#include <Geom_Surface.hxx>
#include <gp_Pnt2d.hxx>
#include <gp_Dir.hxx>
#include <gp_Pnt.hxx>
#include <Standard_Real.hxx>
#include <Precision.hxx>
#include <cmath>
#include <algorithm>

FeatureExtractor::FeatureExtractor(const TopoDS_Shape &shape) : myShape(shape) {}

void FeatureExtractor::Extract()
{
    // 1. 先建立一个 Face 到 Index 的映射表，这样我们才能通过 Shape 反查 ID
    // TopExp::MapShapes 会按照遍历顺序给每个 Face 编一个号 (1 到 Extent)
    TopTools_IndexedMapOfShape faceMap;
    TopExp::MapShapes(myShape, TopAbs_FACE, faceMap);

    // 2. 建立边到面的“倒查表”，通过边找邻居面
    TopExp::MapShapesAndAncestors(myShape, TopAbs_EDGE, TopAbs_FACE, myEdgeFaceMap);

    // 3. 开始计算几何属性和邻居关系
    ComputeGeometricAttributes(faceMap);
}

void FeatureExtractor::ComputeGeometricAttributes(const TopTools_IndexedMapOfShape &faceMap)
{
    myResults.clear();

    // 1. 第一遍遍历：计算总面积
    double totalArea = 0.0;
    for (int i = 1; i <= faceMap.Extent(); ++i)
    {
        TopoDS_Face face = TopoDS::Face(faceMap.FindKey(i));
        GProp_GProps areaProps;
        BRepGProp::SurfaceProperties(face, areaProps);
        totalArea += areaProps.Mass();
    }

    // 2. 第二遍遍历：提取详细特征
    for (int i = 1; i <= faceMap.Extent(); ++i)
    {
        TopoDS_Face face = TopoDS::Face(faceMap.FindKey(i));
        FaceFeature feat;
        feat.id = i;

        // --- 几何特征提取 ---
        GProp_GProps areaProps, lineProps;
        BRepGProp::SurfaceProperties(face, areaProps);
        BRepGProp::LinearProperties(face, lineProps);

        feat.area = areaProps.Mass();
        feat.relativeArea = (totalArea > 1e-6) ? (feat.area / totalArea) : 0.0;
        feat.perimeter = lineProps.Mass();

        // 计算紧致度
        if (feat.area > 1e-6)
        {
            feat.compactness = (feat.perimeter * feat.perimeter) / (4.0 * M_PI * feat.area);
        }
        else
        {
            feat.compactness = 999.0;
        }

        // 提取表面类型和法向以及中心 Z
        GProp_GProps gprops;
        BRepGProp::SurfaceProperties(face, gprops);
        gp_Pnt center = gprops.CentreOfMass();
        feat.centerZ = center.Z();

        // 获取面类型
        BRepAdaptor_Surface surf(face);
        feat.surfaceType = surf.GetType();

        // 提取面中心法向
        double u_min, u_max, v_min, v_max;
        BRepTools::UVBounds(face, u_min, u_max, v_min, v_max);
        gp_Pnt pMid;
        gp_Vec nVec;
        BRepAdaptor_Surface sAtor(face);
        sAtor.D1((u_min + u_max) / 2.0, (v_min + v_max) / 2.0, pMid, nVec, nVec); // 这里简化取中点法向
        
        // 重新计算精确法向
        BRepLProp_SLProps props(sAtor, (u_min + u_max) / 2.0, (v_min + v_max) / 2.0, 1, Precision::Confusion());
        if (props.IsNormalDefined()) {
            gp_Dir normal = props.Normal();
            if (face.Orientation() == TopAbs_REVERSED) normal.Reverse();
            feat.normalX = normal.X();
            feat.normalY = normal.Y();
            feat.normalZ = normal.Z();
        } else {
            feat.normalX = 0; feat.normalY = 0; feat.normalZ = 1.0;
        }

        // 提取曲率特征
        if (props.IsCurvatureDefined()) {
            feat.meanCurvature = props.MeanCurvature();
        } else {
            feat.meanCurvature = 0.0;
        }

        // 提取拓扑复杂度特征
        int wireCount = 0;
        TopExp_Explorer wireExp(face, TopAbs_WIRE);
        for (; wireExp.More(); wireExp.Next()) wireCount++;
        feat.numWires = wireCount;

        int edgeCount = 0;
        TopExp_Explorer edgeCountExp(face, TopAbs_EDGE);
        for (; edgeCountExp.More(); edgeCountExp.Next()) edgeCount++;
        feat.numEdges = edgeCount;

        // 拓扑特征提取：寻找邻居 ID
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
                        
                        // 映射为 ML 友好数值：Convex=1.0, Concave=-1.0, Smooth=0.0
                        int mlEdgeType = 0;
                        if (rawEdgeType == CONVEX) mlEdgeType = 1;
                        else if (rawEdgeType == CONCAVE) mlEdgeType = -1;
                        else mlEdgeType = 0;

                        auto it_nb = std::find(feat.neighborIds.begin(), feat.neighborIds.end(), neighborId);
                        if (it_nb == feat.neighborIds.end())
                        {
                            feat.neighborIds.push_back(neighborId);
                            feat.neighborEdgeTypes.push_back(mlEdgeType);
                        }
                    }
                }
            }
        }
        myResults.push_back(feat);
    }
}

int FeatureExtractor::IdentifyEdgeType(const TopoDS_Face& f1, const TopoDS_Face& f2, const TopoDS_Edge& e)
{
    // 获取边上的中间参数
    Standard_Real first, last;
    BRepAdaptor_Curve cAtor(e);
    first = cAtor.FirstParameter();
    last = cAtor.LastParameter();
    Standard_Real mid = (first + last) / 2.0;

    // 获取该点坐标
    gp_Pnt pMid;
    gp_Vec vTangent;
    cAtor.D1(mid, pMid, vTangent);

    // 获取两个面的法向量
    auto getNormalAndRefVec = [&](const TopoDS_Face& f, const TopoDS_Edge& edge, double param, gp_Dir& normal, gp_Vec& binormal) -> bool {
        BRepAdaptor_Surface sAtor(f);
        BRepAdaptor_Curve cAtor(edge);
        
        // 投影点到面获取 UV (或者通过采样点获取)
        gp_Pnt p; gp_Vec tangent;
        cAtor.D1(param, p, tangent);
        
        // 这里的逻辑需要处理 edge 在 face 中的参数
        Standard_Real u, v;
        Handle(Geom2d_Curve) c2d = BRep_Tool::CurveOnSurface(edge, f, u, v);
        if (c2d.IsNull()) return false;
        gp_Pnt2d uv = c2d->Value(param);
        
        BRepLProp_SLProps props(sAtor, uv.X(), uv.Y(), 1, Precision::Confusion());
        if (!props.IsNormalDefined()) return false;
        
        normal = props.Normal();
        if (f.Orientation() == TopAbs_REVERSED) normal.Reverse();
        
        // 计算面内向量 (Binormal pointing INTO the face)
        // 计算规则：R = Normal x Tangent
        // 如果边在面中的 Orientation 是 REVERSED，则 Tangent 需要反向
        gp_Vec tVec = tangent;
        TopAbs_Orientation edgeOri = TopAbs_FORWARD;
        
        // 寻找边在面中的实际 Orientation
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
    
    // 获取 f2 的法向 n2
    BRepAdaptor_Surface sAtor2(f2);
    Standard_Real u2, v2;
    Handle(Geom2d_Curve) c2d2 = BRep_Tool::CurveOnSurface(e, f2, u2, v2);
    if (c2d2.IsNull()) return OTHER;
    gp_Pnt2d uv2 = c2d2->Value(mid);
    BRepLProp_SLProps props2(sAtor2, uv2.X(), uv2.Y(), 1, Precision::Confusion());
    if (!props2.IsNormalDefined()) return OTHER;
    n2 = props2.Normal();
    if (f2.Orientation() == TopAbs_REVERSED) n2.Reverse();

    // 计算二面角点积
    double dot = n1.Dot(n2);
    if (dot > 0.999) return SMOOTH;

    // 凹凸性核心判断：
    // 如果面 f2 的法向 n2 与面 f1 的向内向量 r1 的点积为正
    // 说明 f2 向 f1 的“内部”偏转 -> 凹 (Concave)
    // 反之 -> 凸 (Convex)
    double check = n2.Dot(r1);
    
    if (check > 1e-6) return CONCAVE;
    if (check < -1e-6) return CONVEX;
    
    return SMOOTH;
}