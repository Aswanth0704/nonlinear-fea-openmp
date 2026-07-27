# Each test is a standalone binary that returns non-zero on failure.
set(WOUND_TESTS
        test_element_functions
        test_kinematics
        test_mesh_generator
        test_openmp_assembly)

foreach(test_name IN LISTS WOUND_TESTS)
    add_executable(${test_name} ${test_name}.cpp)
    target_link_libraries(${test_name} PRIVATE wound_core)
    target_include_directories(${test_name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    add_test(NAME ${test_name} COMMAND ${test_name})
    # Generous, since the assembly test is ~40x slower under sanitizers.
    set_tests_properties(${test_name} PROPERTIES TIMEOUT 900)
endforeach()
