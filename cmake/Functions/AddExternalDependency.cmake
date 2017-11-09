# AddExternalDependency
#
# Allows a Package dependency to be automatically added as an cmake ExternalProject, then built and installed
# to CMAKE_INSTALL_PREFIX.  All this happens before configure time for the client package.
#
# This approach eliminates the need for an explicit submodule for the external package in git and allows the client package to
# be quickly built on systems where the ExternalProject is already installed.
#
# useage: AddExternalDependency(<name> <git-clone-url>)

function(AddExternalDependency ExtProjectName ExtProjectURL)
    find_package(${ExtProjectName} QUIET CONFIG PATHS ${CMAKE_INSTALL_PREFIX}/lib/cmake/${ExtProjectName})
    if(NOT ${ExtProjectName}_FOUND) #Try to configure build and install External package
        set(ExtProjectDir ${CMAKE_BINARY_DIR}/External/${ExtProjectName})
        message(STATUS "[AddExternalProjectDependency] Not found: ${ExtProjectName} --- Initializing as ExternalProject ...")
        configure_file(${CMAKE_SOURCE_DIR}/cmake/Templates/External.CMakeLists.txt.in 
                    ${ExtProjectDir}/CMakeLists.txt @ONLY)
        execute_process(COMMAND ${CMAKE_COMMAND} . WORKING_DIRECTORY ${ExtProjectDir})
        message(STATUS "[AddExternalProjectDependency] Downloading Building and Installing: ${ExtProjectName}")
        execute_process(COMMAND ${CMAKE_COMMAND} --build . WORKING_DIRECTORY ${ExtProjectDir})
        find_package(${ExtProjectName} CONFIG PATHS ${CMAKE_INSTALL_PREFIX}/lib/cmake/${ExtProjectName} NO_CMAKE_FIND_ROOT_PATH)
        if(NOT ${ExtProjectName}_FOUND)
            message(FATAL_ERROR "[AddExternalProjectDependency] Install of ${ExtProjectName} failed.")
        endif()
        message(STATUS "[AddExternalProjectDependency] Installed: ${ExtProjectName} Ver:${${ExtProjectName}_VERSION} Location:${CMAKE_INSTALL_PREFIX}")
    elseif()
        message(STATUS "[AddExternalProjectDependency] Found:${ExtProjectName} Ver:${${ExtProjectName}_VERSION}")
    endif()
endfunction()
