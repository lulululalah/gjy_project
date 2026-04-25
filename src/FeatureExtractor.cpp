#include "FeatureExtractor.h"
#include <TopExp_Explorer.hxx>
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

    // 遍历所有面
    for (int i = 1; i <= faceMap.Extent(); ++i)
    {
        TopoDS_Face face = TopoDS::Face(faceMap.FindKey(i));
        FaceFeature feat;
        feat.id = i; // 直接使用 Map 中的索引作为 ID

        // --- 几何特征提取 ---
        GProp_GProps areaProps, lineProps;
        BRepGProp::SurfaceProperties(face, areaProps); // 二重积分算面积
        BRepGProp::LinearProperties(face, lineProps);  // 一重积分算周长

        feat.area = areaProps.Mass();
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

        // 获取面类型
        BRepAdaptor_Surface surf(face);
        feat.surfaceType = surf.GetType();

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

                    // 如果邻居面不是当前处理的面本身
                    if (!neighborShape.IsSame(face))
                    {
                        int neighborId = faceMap.FindIndex(neighborShape);
                        TopoDS_Face neighborFace = TopoDS::Face(neighborShape);
                        TopoDS_Edge sharedEdge = TopoDS::Edge(edge);

                        int edgeType = IdentifyEdgeType(face, neighborFace, sharedEdge);

                        // 查找是否已经记录过该邻居（处理多条共享边的情况，取最显著特征）
                        auto it_nb = std::find(feat.neighborIds.begin(), feat.neighborIds.end(), neighborId);
                        if (it_nb == feat.neighborIds.end())
                        {
                            feat.neighborIds.push_back(neighborId);
                            feat.neighborEdgeTypes.push_back(edgeType);
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
    auto getNormal = [&](const TopoDS_Face& f, const gp_Pnt& p, gp_Dir& normal) -> bool {
        BRepAdaptor_Surface sAtor(f);
        double u, v;
        TopoDS_Vertex v1 = TopExp::FirstVertex(e);
        gp_Pnt2d uv = BRep_Tool::Parameters(v1, f);
        u = uv.X();
        v = uv.Y();
        BRepLProp_SLProps props(sAtor, u, v, 1, Precision::Confusion());
        if (props.IsNormalDefined()) {
            normal = props.Normal();
            if (f.Orientation() == TopAbs_REVERSED) normal.Reverse();
            return true;
        }
        return false;
    };

    gp_Dir n1, n2;
    if (!getNormal(f1, pMid, n1) || !getNormal(f2, pMid, n2)) return OTHER;

    // 计算二面角
    double dot = n1.Dot(n2);
    if (dot > 0.999) return SMOOTH; // 近似共面或切连

    // 向量叉积判断凹凸性 (Cross Product logic)
    // 这里的逻辑简化为：如果 (n1 x n2) 的方向与边切向 vTangent 一致，则为凸，反之为凹
    // 注意：这取决于面法向定义和边的 Orientation
    gp_Vec cross = n1.XYZ() ^ n2.XYZ();
    if (cross.Magnitude() < 1e-6) return SMOOTH;

    // 基于 BRep 规则的凹凸性判断
    // 简化方案：如果法向量“向外分叉”则是凸，向内则是凹
    // 实际上更健壮的方法是利用两面法向量之和与物体实体的关系
    // 这里我们返回一个占位符，建议后续使用更复杂的 BRepOrientedEdge 分析
    
    // 启发式：如果点积小于0通常是锐角
    if (dot < -1e-6) return CONVEX; // 简单启发
    
    return CONCAVE; 
}