cmake_minimum_required(VERSION 3.12)

project(cuXtra)

set(CUXTRA_LIB ${CMAKE_CURRENT_LIST_DIR}/lib/libcuxtra_${CMAKE_HOST_SYSTEM_PROCESSOR}.a)

if(NOT EXISTS ${CUXTRA_LIB})
    message(FATAL_ERROR "cpu arch ${CMAKE_HOST_SYSTEM_PROCESSOR} not supported by cuxtra")
endif()

add_library(cuxtra INTERFACE)

target_include_directories(cuxtra INTERFACE
    ${CMAKE_CURRENT_LIST_DIR}/include
)

target_link_libraries(cuxtra INTERFACE
    ${CUXTRA_LIB}
)

install(FILES ${CUXTRA_LIB} DESTINATION lib)
install(DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/include/cuxtra DESTINATION include)
