get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        ${extension_libs}
        MLIROptLib
        MLIRComco
        MLIRAnalysis
        MLIRCallInterfaces
        MLIRCastInterfaces
        MLIRFunctionInterfaces
        MLIRIR
        MLIRParser
        MLIRPass
        MLIRSideEffectInterfaces
        MLIRSupport
        MLIRTransforms
        )

add_llvm_executable(comco-compiler compiler.cpp)

llvm_update_compile_flags(comco-compiler)
target_link_libraries(comco-compiler PRIVATE ${LIBS})
