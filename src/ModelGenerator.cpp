#include "ModelGenerator.h"
#include <BRepPrimAPI_MakeBox.hxx>
#include <BRepPrimAPI_MakeCylinder.hxx>
#include <BRepAlgoAPI_Fuse.hxx>
#include <STEPControl_Writer.hxx>
#include <Interface_Static.hxx>
#include <gp_Pnt.hxx>
#include <gp_Ax2.hxx>
#include <gp_Dir.hxx>

TopoDS_Shape ModelGenerator::GenerateRivetPlate(int rows, int cols)
{
    // 1. 创建基础平板 (100x100x2)
    TopoDS_Shape plate = BRepPrimAPI_MakeBox(100.0, 100.0, 2.0).Shape();
    
    TopoDS_Shape result = plate;

    // 2. 在平板上生成阵列铆钉 (直径4, 高2)
    double spacing = 25.0;
    for (int i = 1; i <= rows; ++i)
    {
        for (int j = 1; j <= cols; ++j)
        {
            gp_Pnt loc(i * spacing, j * spacing, 2.0); // 位于平板上表面
            
            // 创建一个小圆柱作为铆钉
            TopoDS_Shape rivet = BRepPrimAPI_MakeCylinder(gp_Ax2(loc, gp_Dir(0, 0, 1)), 2.0, 2.0).Shape();
            
            // 使用布尔并运算融合
            BRepAlgoAPI_Fuse fuse(result, rivet);
            if (fuse.IsDone())
            {
                result = fuse.Shape();
            }
        }
    }
    
    return result;
}

#include <BRepAlgoAPI_Cut.hxx>

TopoDS_Shape ModelGenerator::GenerateMixedPlate()
{
    // 1. 创建基础平板 (100x100x4, 加厚一点方便打孔)
    TopoDS_Shape plate = BRepPrimAPI_MakeBox(100.0, 100.0, 4.0).Shape();
    TopoDS_Shape result = plate;

    // 2. 生成一个铆钉 (凸起) - 位于 (30, 50, 4)
    gp_Pnt rivetLoc(30.0, 50.0, 4.0);
    TopoDS_Shape rivet = BRepPrimAPI_MakeCylinder(gp_Ax2(rivetLoc, gp_Dir(0, 0, 1)), 2.0, 2.0).Shape();
    BRepAlgoAPI_Fuse fuse(result, rivet);
    if (fuse.IsDone()) result = fuse.Shape();

    // 3. 生成一个盲孔 (凹陷) - 位于 (70, 50, 4)
    // 注意：圆柱要向下长，或者起点下移
    gp_Pnt holeLoc(70.0, 50.0, 4.0);
    TopoDS_Shape tool = BRepPrimAPI_MakeCylinder(gp_Ax2(holeLoc, gp_Dir(0, 0, -1)), 2.0, 2.0).Shape();
    BRepAlgoAPI_Cut cut(result, tool);
    if (cut.IsDone()) result = cut.Shape();

    return result;
}

bool ModelGenerator::SaveToStep(const TopoDS_Shape& shape, const std::string& filename)
{
    STEPControl_Writer writer;
    IFSelect_ReturnStatus status = writer.Transfer(shape, STEPControl_AsIs);
    if (status != IFSelect_RetDone) return false;
    
    return writer.Write(filename.c_str()) == IFSelect_RetDone;
}
