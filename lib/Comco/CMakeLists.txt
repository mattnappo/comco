add_mlir_dialect_library(MLIRComco
        ComcoDialect.cpp
        ComcoOps.cpp
        ComcoFuncLowering.cpp
        ComcoMatmulLowering.cpp
        ComcoBufferLowering.cpp
        ComcoAllReduceLowering.cpp
        ComcoAllReduceOutlining.cpp
        ComcoLauncher.cpp
        ComcoFusion.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/Comco

        DEPENDS
        MLIRComcoOpsIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRInferTypeOpInterface
)
