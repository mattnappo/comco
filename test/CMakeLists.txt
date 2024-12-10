configure_lit_site_cfg(
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
        ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
        MAIN_CONFIG
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(COMCO_OPT_TEST_DEPENDS
        FileCheck count not
        comco-opt
        )

add_lit_testsuite(check-comco-opt "Running the comco-opt regression tests"
        ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS ${COMCO_OPT_TEST_DEPENDS}
        )
set_target_properties(check-comco-opt PROPERTIES FOLDER "Tests")

add_lit_testsuites(COMCO_OPT ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${COMCO_OPT_TEST_DEPENDS})
