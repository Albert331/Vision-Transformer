cmake_minimum_required(VERSION 3.18)
project(siglip_inference)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_PREFIX_PATH "D:/libtorch/libtorch")

find_package(Torch REQUIRED)

add_executable(siglip_inference main.cpp)
target_link_libraries(siglip_inference "${TORCH_LIBRARIES}")
set_property(TARGET siglip_inference PROPERTY CXX_STANDARD 20)

if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET siglip_inference
                      POST_BUILD
                      COMMAND ${CMAKE_COMMAND} -E copy_if_different
                      ${TORCH_DLLS}
                      $<TARGET_FILE_DIR:siglip_inference>)
endif (MSVC)