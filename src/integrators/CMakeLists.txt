set(MI_PLUGIN_PREFIX "integrators")

add_plugin(aov        aov.cpp)
add_plugin(depth      depth.cpp)
add_plugin(direct     direct.cpp)
add_plugin(moment     moment.cpp)
add_plugin(path       path.cpp)
add_plugin(ptracer    ptracer.cpp)
add_plugin(stokes     stokes.cpp)
add_plugin(volpath    volpath.cpp)
add_plugin(volpathmis volpathmis.cpp)

find_package(Python3 COMPONENTS Development NumPy REQUIRED)
add_plugin(gravity    gravity.cpp ${LIBYT_PATH}/lib/libyt.so)

#target_link_libraries(gravity PRIVATE yt)
target_link_directories(gravity PRIVATE ${Python_LIBRARIES} Python::NumPy)
target_include_directories(gravity PRIVATE ${LIBYT_PATH}/include ${Python_INCLUDE_DIRS})
# copy to root for now
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/geodesic.py
    ${CMAKE_BINARY_DIR}/geodesic.py COPYONLY
)

configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/coor_map.csv
        ${CMAKE_BINARY_DIR}/coor_map.csv COPYONLY
)

set(MI_PLUGIN_TARGETS "${MI_PLUGIN_TARGETS}" PARENT_SCOPE)
