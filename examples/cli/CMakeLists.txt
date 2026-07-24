add_executable(magpie-cli main.cpp)
target_link_libraries(magpie-cli PRIVATE magpie-tts)
target_include_directories(magpie-cli PRIVATE ${CMAKE_SOURCE_DIR}/src)
