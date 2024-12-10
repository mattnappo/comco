get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIROptLib
        MLIRComco
        )
add_llvm_executable(comco-opt comco-opt.cpp)

llvm_update_compile_flags(comco-opt)
target_link_libraries(comco-opt PRIVATE ${LIBS})
