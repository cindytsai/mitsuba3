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
add_plugin(gravity    gravity.cpp)

configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/coor_map.csv
        ${CMAKE_BINARY_DIR}/coor_map.csv COPYONLY
)
configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/bending_ray_lookup_table.csv
        ${CMAKE_BINARY_DIR}/bending_ray_lookup_table.csv COPYONLY
)

set(MI_PLUGIN_TARGETS "${MI_PLUGIN_TARGETS}" PARENT_SCOPE)
