cmake_minimum_required(VERSION 3.10)
project(gjy_byproject)

# 1. 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 2. 定义路径变量 
set(OCC_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/3rdParty/OCC")
# TBB 的路径
set(TBB_BIN_DIR "${OCC_ROOT}/tbb_2017.0.100/bin/intel64/vc14")

# 3. 包含头文件路径
include_directories("${OCC_ROOT}/inc") 
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include")

# 4. 设置库文件搜索目录
link_directories("${OCC_ROOT}/lib")

# 5. 定义需要链接的 OpenCASCADE 核心库清单
set(OCC_LIBS
    TKernel TKMath 
    TKG2d TKG3d TKBRep 
    TKGeomBase TKGeomAlgo 
    TKTopAlgo TKPrim TKBO TKShHealing
    TKSTEP TKSTEPBase TKSTEPAttr TKXSBase TKLCAF
)

# 6. 手动列出所有源文件
set(SRC_FILES
    "src/main.cpp"
    "src/FeatureExtractor.cpp"
    "src/FeatureClassifier.cpp" 
    "src/GeometryModifier.cpp"
    "src/ModelGenerator.cpp"
)

# 创建可执行文件 
add_executable(Detector ${SRC_FILES})

# 7. 链接库文件到目标
target_link_libraries(Detector ${OCC_LIBS})

# 8. 自动化：将所有必要的 DLL 复制到生成目录 (Debug)

# 获取 OCC 核心 DLL
file(GLOB OCC_DLLS "${OCC_ROOT}/bin/*.dll")

# 获取 TBB 加速库 DLL
file(GLOB TBB_DLLS "${TBB_BIN_DIR}/*.dll")

# 执行复制动作 (确保程序双击就能跑)
if(OCC_DLLS OR TBB_DLLS)
    file(COPY ${OCC_DLLS} ${TBB_DLLS} DESTINATION "${CMAKE_BINARY_DIR}/Debug")
    message(STATUS "DLLs (OCC & TBB) auto-copy configured.")
else()
    message(WARNING "No DLLs found! Please check your 3rdParty paths.")
endif()

# 将项目根目录下的 data 文件夹绝对路径传递给 C++ 代码
target_compile_definitions(Detector PRIVATE 
    TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
)