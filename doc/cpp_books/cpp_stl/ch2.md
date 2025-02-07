# Chap 2: C++ 及其标准程序库简介

### 2.2 新的语言特性
- Template 并非一次编译就生出适合所有类型的代码，而是针对被使用的某个类型进行编译。-> 必须提供实作品才能调用。inline可解决。

- typename作为型别前的表示符号。

- 如果采用不含参数的、明确的 constructor 调用语法，基本型别会被初始化为零。
  ```cpp
  T x = T(); // x 保证被初始化为 0
  ```
  
- 对于所有被声明于因异常而退离的区段里的局部对象，都会被调用 destructor，知道遇到 catch 或者 main() 结束。

- namespace 用法：
  ```cpp
  namespace Your_namespace {
      class name1;
      void name2();
      ...
  }
  Your_namespace::name1 obj1;
  ```

  namespace 定义的是逻辑模块，而不是实质模块。

- Koenig lookup Rule：如果一个函数的一个或多个参数型别定义于函数所处的 namespace 里，那么可以不必为函数指定 namespace。
  ```cpp
  namespace josuttis {
      class File;
      void myFunc(const File&);
  }
  josuttis::File obj;
  myFunc(obj); // this works.
  ```

- explicit 关键字：可以禁止”单参数构造函数“被用于自动类型转换。可以阻止”以赋值语法进行带有转型操作的初始化“。

- 型别转换操作符（只接受一个参数）：

  - static_cast<target_type>(obj)：将一个值以符合逻辑的方式转型。
  - (derived_class* ptr = )dynamic_cast<base_class*>(base_ptr)：将多态型别向下转型为其实际静态型别。
  - const_cast：设定或去除型别里的常数性。
  - reinterpret_cast：由编译器决定。

- 可以在class声明中对”整数型常数静态成员“直接赋初值。static const int num = 100;
  还需要定义一个空间：const int MyClass::num;