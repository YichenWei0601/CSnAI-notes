# Log: ARW building

- 在 CMakeLists.txt 里面，添加所需要搜索的头文件目录：
  ```txt
  # 添加头文件搜索路径，根据实际情况添加你需要的路径，比如 MISConfig 和 graph_access 的位置
  target_include_directories(heubase PRIVATE 
      ${CMAKE_SOURCE_DIR} 
      kamis-source/lib/mis/ils
      kamis-source/lib/mis      # 如 MISConfig 所在目录
      kamis-source/lib/data_structure  # 如 graph_access 所在目录
      # 其他目录
  )
  ```

  

- CMake 找不到 pybind11：
  ```
  CMake Error at CMakeLists.txt:5 (find_package): By not providing "Findpybind11.cmake" in CMAKE_MODULE_PATH this project has asked CMake to find a package configuration file provided by "pybind11", but CMake did not find one.
  
  Could not find a package configuration file provided by "pybind11" with any of the following names:
  
  pybind11Config.cmake
  pybind11-config.cmake
  
  Add the installation prefix of "pybind11" to CMAKE_PREFIX_PATH or set "pybind11_DIR" to a directory containing one of the above files. If "pybind11" provides a separate development package or SDK, be sure it has been installed.
  ```

  解决方法：

  ```bash
  pip install pybind11
  ```

  如果已经安装，通过以下方法找到 CMake 配置文件路径。

  ```
  python -m pybidn11 --cmakedir
  ```

  然后在调用 CMake 时添加参数：

  ```bash
  cmake -Dpybind11_DIR=/usr/local/lib/python3.x/site-packages/pybind11/share/cmake/pybind11 ..
  ```

  