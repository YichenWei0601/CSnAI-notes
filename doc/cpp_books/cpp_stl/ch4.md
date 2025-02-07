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
### 4.3 数值极限
- numeric_limits<...> 作为类。用法是numeric_limits<...>:: ...。
- 需要 #include <limits>。
### 4.4 辅助函数
- 挑选较小较大值：在 <algorithm> 里，std::min(A, B), std::max(A, B)，或者 std::min/max(A, B, 比较准则)。注意这里可以不用 <...>，因为函数形参取的是引用。也可以加上。
- 两值互换：在 <algorithm>，swap()
- 辅助性的“比较操作符”：!=, >, <=, >= 都是依靠 == 和 < 完成的。在 <utility> 里面。需要 using namespace std::rel_ops。
- <cstddef>: Null, size_t, ptrdiff_t,  offsetof
- <cstdlib>: exit, abort, ...
	- exit() 会销毁所有 static 对象，清空所有缓冲区，关闭所有I/O通道，然后终止程序。
	- abort() 会立刻终止函数，不会做任何清理。
	- 两者都不会销毁局部对象，因为堆栈辗转开展动作不会被执行起来。需要运用异常或正常返回机制，然后由 main() 离开。
