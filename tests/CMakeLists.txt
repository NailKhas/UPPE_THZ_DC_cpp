set(UPPE_TESTS
  test_bessel
  test_fft
  test_hankel
  test_propagation
  test_nonlinear
  test_first_step
  test_simulation_smoke
  test_checkpoint
)

foreach(t IN LISTS UPPE_TESTS)
  add_executable(${t} ${t}.cpp)
  target_link_libraries(${t} PRIVATE uppe_core)
  target_include_directories(${t} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
endforeach()

add_test(NAME bessel COMMAND test_bessel)
add_test(NAME fft COMMAND test_fft)
add_test(NAME hankel COMMAND test_hankel)
add_test(NAME propagation COMMAND test_propagation)
add_test(NAME nonlinear COMMAND test_nonlinear)
add_test(NAME first_step COMMAND test_first_step)
# The data directory is passed relative to the working directory, so non-ASCII
# source paths never go through the command line.
add_test(NAME simulation_smoke COMMAND test_simulation_smoke data WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
set_tests_properties(simulation_smoke PROPERTIES TIMEOUT 1800)
add_test(NAME checkpoint COMMAND test_checkpoint data WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
set_tests_properties(checkpoint PROPERTIES TIMEOUT 1800)
