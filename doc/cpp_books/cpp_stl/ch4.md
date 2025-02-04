# Chap 4: 通用工具

### 4.1 Pairs

- class pair 可以将两个值视为一个单元。
- pair 被定义为 struct，所有成员都是 public。
- default 生成 pair 时，按照构造函数分别初始化两个值。
```cpp
pair(): first(T1()), second(T2()) {}
```
- 元素都相等视为 pair 相等。第一元素有优先级
- make_pair() 无需写出 <> 型别，就可生成 pair 对象。

### 4.2 Class auto_ptr
- 如果资源是以显式手法（explicit）获得，而且没有绑定在任何对象身上，就必须以显式手法释放。智能指针保证只要自己被摧毁，就一定连带使放弃其所指资源。
- auto_ptr **只能有一个拥有者**。
- auto_ptr 不能用赋值初始化，要数值初始化。
```cpp
std::auto_ptr<ClassA> ptr1(new ClassA); // correct
std::auto_ptr<ClassA> ptr2 = new ClassA; // incorrect
// 本质上因为左侧是 auto_ptr ，右侧是普通指针。以下写法是正确的：
std::auto_ptr<ClassA> ptr;
ptr = std::auto_ptr<ClassA>(new ClassA);
```
- auto_ptr 的拷贝 / 赋值：交出拥有权，对方删除现在拥有的对象（delete），然后得到拥有权。
  所以一般都需要重载 copy constructor 和 assignment operator。
- pass by reference 使得无法预知所有权是否转交，应尽量避免。
- const auto_ptr 是不能更改所有权的意思。
- 不存在针对 array 设计的 auto_ptr。因为 auto_ptr 通过 delete 而非 delete[] 释放对象的。
- 记得 #include <memory>
- 
