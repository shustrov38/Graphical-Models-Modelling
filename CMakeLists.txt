cmake_minimum_required(VERSION 3.0)
project(Graphical-Models-Modelling)

###############################################################################
## file globbing ##############################################################
###############################################################################

file(GLOB_RECURSE sources      src/main/*.cpp src/main/*.h)
file(GLOB_RECURSE sources_test src/test/*.cpp)
file(GLOB_RECURSE data res/*)

###############################################################################
## target definitions #########################################################
###############################################################################

add_executable(${PROJECT_NAME} ${sources} ${data})
target_compile_options(${PROJECT_NAME} PUBLIC -std=c++2a -Wall -pedantic -lm -O3 -funroll-loops)
target_include_directories(${PROJECT_NAME} PUBLIC src/main)
file(COPY ${data} DESTINATION res)

###############################################################################
## dependencies ###############################################################
###############################################################################

find_package(MPI 3.1 REQUIRED)
set(CMAKE_C_COMPILE_FLAGS "${CMAKE_C_COMPILE_FLAGS} ${MPI_C_COMPILE_FLAGS}")
set(CMAKE_C_LINK_FLAGS "${CMAKE_C_LINK_FLAGS} ${MPI_C_LINK_FLAGS}")
set(CMAKE_CXX_COMPILE_FLAGS "${CMAKE_CXX_COMPILE_FLAGS} ${MPI_CXX_COMPILE_FLAGS}")
set(CMAKE_CXX_LINK_FLAGS "${CMAKE_CXX_LINK_FLAGS} ${MPI_CXX_LINK_FLAGS}")
target_link_libraries(${PROJECT_NAME} PUBLIC ${MPI_CXX_LIBRARIES})

find_package(Boost 1.71.0 REQUIRED COMPONENTS 
    filesystem 
    system
    mpi
    program_options
)
target_link_libraries(${PROJECT_NAME} PUBLIC
    ${Boost_LIBRARIES}
)

###############################################################################
## testing ####################################################################
###############################################################################

find_package(GTest)

if(GTEST_FOUND)
    add_executable(unit_tests ${sources_test} ${sources})
    target_compile_definitions(unit_tests PUBLIC UNIT_TESTS)
    set_target_properties(${PROJECT_NAME} PROPERTIES ENABLE_EXPORTS ON)
    target_link_libraries(unit_tests PUBLIC
        ${GTEST_BOTH_LIBRARIES}
        ${PROJECT_NAME}
    )
    target_include_directories(unit_tests PUBLIC
        ${GTEST_INCLUDE_DIRS}
    )
endif()

###############################################################################
## packaging ##################################################################
###############################################################################

install(TARGETS ${PROJECT_NAME} DESTINATION ${PROJECT_NAME}-destination)
install(DIRECTORY res DESTINATION ${PROJECT_NAME}-destination)

set(CPACK_PACKAGE_NAME ${PROJECT_NAME})
set(CPACK_PACKAGE_VERSION "1.0.0")

set(CPACK_MONOLITHIC_INSTALL 1)

include(CPack)