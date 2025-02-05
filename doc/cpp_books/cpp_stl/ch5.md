---
​---
title: Standard Template Library
author: calscatt
date: 2025-02-05
​---
---

# Chap 5: Standard Template Library

### 5.1 STL 组件 (STL Components)

- Containers 容器，管理某类对象的集合。
- Iterators 迭代器，在一个对象群集的元素上进行遍历。每一种容器都提供了自己的迭代器，同时提供一致的接口。可以把它视为一种 smart pointer。
- Algorithms 算法，处理群集内的元素。
- STL 提供泛型化的组件，通过特定的配接器和仿函数，可以补充、约束或定制算法，满足特殊需求。

### 5.2 容器 Containers
- 容器分为两类：
  - Sequence containers： 可序群集，每个元素取决于插入时机和地点有固定的位置，与元素值无关。包含 vector, deque (two sides both can be extended), list。
  - Associative containers: 已序群集，元素为i只取决于特定的排序准则。set, multiset, map, multimap。通常关联式容器由二叉树做出。
- vector 将元素放在一个 dynamic array 里管理，允许随机存取。尾部处理快，头部处理慢（因为要移动位置）。
  ```cpp
  #include <vector>
  using namespace std;
  
  vector<int> coll;	//声明
  coll.push_back(i);	// append at the end
  for (int i = 0; i < coll.size(); ++i) {}	// .size()
  coll[i]		// ith item
  ```
- deque (double-ended queue) 是 dynamic array，可以向两端发展。
  ```cpp
  #include <deque>
  using namespace std;
  
  deque<float> coll;
  coll.push_front(i);		// 比 vector 多了一个 push_front
  ```
- list 由双向链表实作而成，每个元素都以一部分内存指示其前驱元素和后继元素。不提供随机存取，需要沿着串链依次到达目标。O(n) time。优势是在任何位置上执行安插或删除动作非常迅速。
  ```cpp
  #include <list>
  using namespace std;
  
  list<char> coll;
  coll.push_back(c);
  bool x = coll.empty();		// whether coll is empty
  char fnt = coll.front();		// first item
  coll.pop_front();				// delete first item
  // pop_front() do not return the original coll.front()!
  ```
- array 不是 STL 里的，没有 size(), empty() 等成员函数！！！
- set 内部元素按照值自动排序，元素不许重复。multiset 允许重复元素。map 按照键排序，每个键只能出现一次，multimap 允许重复。排序准则缺省采用 operator<。
- 容器配接器 Container Adapters：stack (LIFO 后进先出)，queue (FIFO 先进先出)，priority queue（按照优先权，优先权按照排序准则 / 缺省时的 operator<）
### 5.3 迭代器 Iterator
- begin() 指向容器起点，也就是第一元素的位置。end() 指向结束点，也就是最后一个元素之后。这样 1. 遍历元素只要不是 end 就可以进行循环 2. 空空间的 begin() 就是 end()。它们相当于指针。
- 任何一种容器都有两种迭代器：container::iterator 以读/写模式遍历，container::const_iterator 只读模式遍历元素。
- 前置式递增效率更高（++pos 更好）
- 迭代器遍历 set 是按照大小遍历（1，2，3，4，5，6）。
- map/multimap 注意：
  - 需要先 make_pair() 再放入 map
  - 迭代器指的是 pair，需要取出其中 first / second 元素。
- multimap 不支持 subscript 操作符，因为 subscript 操作符只能处理单一实值。
- 迭代器分类：双向迭代器（可+可-），随机存取迭代器（还具备随机访问能力）

  
