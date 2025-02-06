---
Date: 2025-02-06
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

### 5.3 Deques

- #include <deque>
- 与 vector 相比的部分区别：
  - 存取元素因有内部间接过程而慢一点。
  - 迭代器需要在不同区块间跳转，需要时智能指针。
  - max size 可能更大
  - 不支持对容量和内存重分配时机的控制。在头尾两端意外任何地方插入删除元素会使得 references、pointers、iterators 失效。

