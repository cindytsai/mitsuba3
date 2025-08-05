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

target_link_libraries(gravity PRIVATE yt)
target_link_directories(gravity PRIVATE ${LIBYT_PATH}/lib -lyt)
target_include_directories(gravity PRIVATE ${LIBYT_PATH}/include)

set(MI_PLUGIN_TARGETS "${MI_PLUGIN_TARGETS}" PARENT_SCOPE)
