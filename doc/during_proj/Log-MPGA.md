# Log	MPGA: 多种群遗传算法

- windows 环境下 cmake + vscode：使用ctrl+shift+p，选 cmake configure，然后跑。

- 解决pybind11找不到目录：[VScode c++调用pybind11 - 知乎](https://zhuanlan.zhihu.com/p/676396541)

具体方法是点小灯泡后修改c/c++扩展里面的c_cpp_properties.json，在 include path里面加入1. pybind11	2. python.h	3.  frameobject.h 所在位置的目录。本电脑的地址参考：

```json
"C:/Users/weiyi/.vscode/extensions/ms-vscode.cpptools-1.23.6-win32-x64/bin",
"C:/Users/weiyi/.vscode/extensions/ms-python.debugpy-2025.4.0-win32-x64/bundled/libs/debugpy/_vendored/pydevd/pydevd_attach_to_process/common", 
"C:/Users/weiyi/AppData/Local/Programs/Python/Python313/include",
```

其中第三个需要官网下载python然后才能找到。第一个需要 pip install pybind11。

- 小小的 vector transportation：from python to cpp。

  ```cpp
  # vector_module.cpp
  #include <pybind11/pybind11.h>
  #include <pybind11/stl.h>
  #include <vector>
  #include <iostream>
  
  namespace py = pybind11;
  
  // C++ 函数：接收 Python 生成的 vector
  void receive_vector(std::vector<int> vec) {
      std::cout << "Received vector: ";
      for (int num : vec) {
          std::cout << 'a' << " ";
      }
      std::cout << std::endl;
  }
  
  // 绑定 Python 模块
  PYBIND11_MODULE(vector_module, m) {
      m.def("receive_vector", &receive_vector, "Receives a vector from Python");
  }
  ```

  ```python
  # setup.py
  from setuptools import setup, Extension
  import pybind11
  
  ext_modules = [
      Extension(
          "vector_module",
          ["vector_module.cpp"],
          include_dirs=[pybind11.get_include()],
          language="c++"
      ),
  ]
  
  setup(
      name="vector_module",
      ext_modules=ext_modules,
  )
  ```

  ```python
  # test.py
  import vector_module
  
  # Python 生成一个 list
  def generate_vector():
      return [1, 2, 3]
  
  # 获取 Python 生成的 vector
  vec = generate_vector()
  
  # 传递给 C++ 函数
  vector_module.receive_vector(vec)
  
  ```

  操作顺序：

  0. cd to the current address.
  1. Run in terminal: python setup.py build_ext --inplace
  2. [For the testrun]: run test. 

  The test is used for testing the transmition of a vector from
  python to c++, For c++ part to process the vector.

传入多个 instance 从而建立 population：只需要在 gen_vec() 实例化和传递给 cpp 的两行代码前面加一个循环就可以。ez

写好了整个MPGA，其中 immigration 参数需要调整，目前是 $10$，同时关于人工选择部分，目前是合并变成 $size = 2n$ 然后选择合并后的最好的 $n$ 个。

同时路径都是在本地部署的，转移需要重新部署诸如 pybind11 之类的东西。

- **`CMakeLists.txt` 生成 `vector_module.so`，不要编译 `main.cpp`**，否则可能会冲突。(pybind11 重复定义)
