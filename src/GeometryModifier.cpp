#include "GeometryModifier.h"
#include <TopExp_Explorer.hxx>
#include <TopoDS_Face.hxx>
#include <TopoDS.hxx>
#include <BRepBuilderAPI_Sewing.hxx>
#include <STEPControl_Writer.hxx>
#include <Interface_Static.hxx>
#include <iostream>

GeometryModifier::GeometryModifier(const TopoDS_Shape &shape) : myOriginalShape(shape) {}

TopoDS_Shape GeometryModifier::ExecuteElimination(const std::vector<FaceFeature> &features)
{
    // 1. 初始化缝合工具 (设定一个公差，让微小的空隙能合上)
    BRepBuilderAPI_Sewing sewer(1e-3);

    // 2. 遍历原始模型的所有面
    int faceIdx = 0;
    TopExp_Explorer exp(myOriginalShape, TopAbs_FACE);
    for (; exp.More(); exp.Next())
    {
        TopoDS_Face currentFace = TopoDS::Face(exp.Current());
        faceIdx++;

        // 寻找该面在特征列表中的对应项 (这里假设顺序一致，或者你可以通过 ID 匹配)
        bool shouldDelete = false;
        for (const auto &f : features)
        {
            // 只要标签不是 0（正常面），就说明它是识别出来的脏几何，需要删除
            if (f.id == faceIdx && f.semanticTag != 0)
            {
                shouldDelete = true;
                break;
            }
        }

        // 3. 算子映射：如果不是碎面，则保留并加入缝合器
        if (!shouldDelete)
        {
            sewer.Add(currentFace);
        }
        else
        {
            std::cout << "正在消除 ID: " << faceIdx << " (类型: 碎面)" << std::endl;
        }
    }

    // 4. 执行缝合操作，维护拓扑一致性
    sewer.Perform();
    return sewer.SewedShape();
}

bool GeometryModifier::SaveStep(const TopoDS_Shape &shape, const std::string &filename)
{
    STEPControl_Writer writer;
    IFSelect_ReturnStatus status = writer.Transfer(shape, STEPControl_AsIs);
    if (status != IFSelect_RetDone)
        return false;

    return (writer.Write(filename.c_str()) == IFSelect_RetDone);
}