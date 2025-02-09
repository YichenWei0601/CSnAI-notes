---
Date: 2025-02-07
---

# Chap 6：STL 容器

### 6.1 容器的共通能力和共通操作

- 共通能力：
  - 所有容器提供的都是 value 而不是 reference。
  - 总体而言，所有元素形成一个次序。每个容器都提供“可返回迭代器”的函数来遍历元素。
  - 一般而言，各项操作并非绝对安全。

### 6.2 Vectors

- #include <vector>

- 所存的元素需要 assignable 和 copyable

- vector 支持随机存取，元素之间总存在某种顺序。

- capacity() 返回 vector 实际能够容纳的元素数量。如果超越这个数量，需要重新配置内部存储器。
  - 重新配置，和 vector 元素相关的所有 references、pointers、iterators 都会失效。
  - 内存重新配置耗时长。
  
- vec.reserve(...) 保留一定的内存容量（"..." 是元素数量）（小于已有容量则无反应）。或者 std::vector<int> v (5) 在初始化时给定起始大小。

- 缩减容量：copy + swap
  ```cpp
  std::vector<T> tmp(v);
  v.swap(tmp);
  ```

  注意 references、pointers、iterators 都换了对象！要换过来。

- 元素存取：c.at(idx) 会进行越界检查（抛出 out_of_range），c.[idx], c.front(), c.back() 不会进行范围检查。

- 安插或移除元素会导致“作用点”后的 references、pointers、iterators 失效。

- vector<bool> 为特殊版本，以 bit 存储元素。特殊操作：

  - c.flip() 求补码
  - m[idx].flip()

### 6.3 Deques

- #include <deque>

- 与 vector 相比的部分区别：

  - 存取元素因有内部间接过程而慢一点。

  - 迭代器需要在不同区块间跳转，需要时智能指针。

  - max size 可能更大

  - 不支持对容量和内存重分配时机的控制。在头尾两端意外任何地方插入删除元素会使得 references、pointers、iterators 失效。

### 6.4 Lists
- #include <list>

- list 不支持随机存取，因此不支持下标操作符或者 at()。同时不提供容量、空间重新分配等操作函数。

- 对于空容器执行任何操作都会导致未定义的行为。因此调用者需要确保容器至少含有一个元素。

- list 的迭代器只是双向迭代器，用到随机存取迭代器的算法都不能调用（操作元素顺序的）。所调用的是 list 的成员函数。

- c.unique(), c.unique(op),
  c1.splice(pos, c2), c1.splice(pos, c2, c2pos), c1.splice(pos, c2, c2begin, c2end) (doing transform from c2 to c1), 
  c.sort(), c.sort(op), 
  c1.merge(c2), c1.merge(c2, op) (remain sorted after emerge), 
  c.reverse()

- list 几乎所有操作要么成功、要么无效。对于异常安全性提供了最佳支持。

### 6.5 Sets and Multisets
- set 和 multiset 会根据特定的排序准则，自动将元素排序。multiset 允许重复而 set 不允许。
- #include <set> （对两者一样）
- set 第二个 template 参数是排序准则，默认为 less，即 operator<。排序准则必须定义 strict weak ordering：
  - 反对称的 antisymmetric：x<y=1 -> y<x=0
  - 可传递的 transitive：x<y, y<z -> x<z
  - 自反的 irreflexive：x<x = 0
-  通常以平衡二叉树完成。有对数复杂度。
- 元素比较动作只能用于型别相同的容器，即元素和排列准则必须有相同的型别，否则编译时期会产生型别方面的错误。
- 搜索函数，需要调用同名的 STL 算法的**特殊版本**，从而获得对数复杂度：
  count(elem), find(elem), lower_bound(elem), upper_bound(elem) （返回第一/最后的可安插 elem 的位置）, equal_range(elem)（返回第一和最后的可安插位置，即区间）。
- 赋值：赋值操作两端容器必须具有相同型别。尽管比较准则本身可能不同，但型别必须相同。如果准则不同，准则本身也会被赋值或交换。
  ```cpp
  // 型别相同但准则不同的实现方法：
  class RuntimeCmp {
  public:
      enum cmp_mode {normal, reverse};
      cmp_mode mode;
      // 然后设定怎样设定和改变这两个 mode。
      RuntimeCmp (cmp_mode m=normal): mode(m) {};
      bool operator() (..., ...) {mode == normal? ...: ... ;};
  };
  ```

- c.insert(elem) 对于 set 返回 pair (iterator, bool)，bool 返回插入成果与否，iterator 返回新元素的位置，或者现存同值元素的位置。multiset 直接返回 iterator。若要计算距离用 distance(iterator1, iterator2)。

- set 和 multiset 只用 erase()。如果只移除第一个，先 find 再保证非 end 的前提下 erase。

- 序列式容器的 erase 返回 iterator，但是关联式容器返回 void：关联式容器找到这个元素的下一个元素的位置来返回需要通过二叉树完成，不方便。

### 6.6 Maps and Multimaps

- #include <map>

- 如果需要改变元素的 key，只能以一个 value 相同的新元素替换掉旧元素。或者如下：
  ```cpp
  coll["new_key"] = coll["old_key"];
  coll.erase("old_key");
  ```

### 6.7 其它 STL 容器

- Open-Closed 开放性封闭：允许扩展，谢绝修改。
- 使容器 STL 化的三种方法：
  - The invasive approach 侵入性做法：直接提供 STL 容器所需接口。
  - The noninvasive approach 非侵入性做法：由你撰写或提供特殊迭代器（能够遍历元素），作为算法和特殊容器间的界面。
  - The wrapper approach 包装法：结合上两个。

### 6.8 动手实现 Reference 语义

- 实现对指针所指对象采用 reference counting 的智能型指针。类似 auto_ptr，但是被复制后，原指针和新的副本指针都有效。只有当指向同一对象的最后一个智能型指针被摧毁，其所指对象才会被删除。
- 参考 http://www.boost.org/ 的 Boost 程序库（是 C++ 标准程序库的补充）。（这个指针在里面叫做 shared_ptr）

### 6.9 各种容器的运用时机

- 缺省情况下用 vector，内部结构最简单，允许随机存取。
- 按照安插元素的需求选择 vector，list，deque。
- 如果不希望 iterators / pointers / references 失效，采用 list。
- “每次操作若不成功，便无效用”（并用此态度处理异常），则用 list。
- 就搜索速度而言，hash table 通常比二叉树还要快 5~10 倍。所以如果有 hash table （即使未标准化）也考虑使用。但是排序不用 hash table。
- 根据两种不同的排序准则对元素进行排序，则需要两个 set / map，准则不同但共享相同的元素。

### 6.10 细说容器内的型别和成员

