---
Date: 2025-02-06
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
### 5.4 算法
- #include <algorithm>
- min/max_element(coll.begin(), coll.end())：返回最小/最大元素的位置。
- sort(coll.begin(), coll.end(), ...) 默认按照 operator< 排列。
- find(coll.begin(), coll.end(), ...) 找到第一个为...的元素。失败则返回 past-the-end 迭代器。
- reverse(coll.begin(), coll.end()) 元素反转位置。
- 调用者需要保证经由首位位置的参数定义取来的区间是有效的（**属于同一容器，前后位置正确**）。区间是左闭右开（即 [begin, end)）。一定注意** end 需要在想要考虑的元素的后一位**。如果不确定前后位置，可以用 pos1 < pos2 的 if-else 条件语句来分类讨论（当使用的是随机存储迭代器时），或者写一段代码来确定 pos1~end 之间是否有 pos2 出现，即手动判断两者二的前后关系（当没有随机储存迭代器）（或者直接找到两者第一次出现的位置）。
- 处理多个区间：通常需要设定第一个区间的前后位置和之后区间的前位置（后位置可由第一区间推测得到）。
  ```cpp
  if (equal(coll1.begin(), coll1.end(), coll2.begin()) {...}
  ```
  因此需要保障后面的区间拥有的元素个数至少和第一区间的元素个数相同。以及涂写（如 copy(..., ..., ...)）操作时，确保目标空间够大，不会超出去（或者需要调整容器大小）。否则这种未定义的行为会被导向 error handling procedure。
- 调整容器大小：coll2.resize(coll1.size())
### 5.5 迭代器之配接器 Iterator Adapters
- STL 提供了数个预先定义的特殊迭代器，即 Iterator Adapeters。
- Insert Iterator
  - 使算法以 insert 而非 overwrite 方式运作，解决算法的目标空间不足问题。
  - Back inserters: copy(coll1.begin(), coll1.end(), back_inserter(coll2)) 调用 push_back().
  - Front inserters: copy(coll1.begin(), coll1.end(), front_inserter(coll2)) 调用 push_front()。只能提供有 push_front() 的容器。
  - General inserters: copy(coll1.begin(), coll1.end(), inserter(coll2, coll2.begin()))，将元素插入初始化时接受的第二参数所指位置的前方，调用 insert()。
- Stream Iterator
  - istream_iterator<string>(cin)：产生一个可从标准输入流cin读取数据的 stream iterator。
  - istream_iterator<string>()：default 构造函数，产生一个代表流结束符号 end-of-stream 的迭代器。
  - ostream_iterator<string>(cout, "\n") 产生一个 output stream iterator，透过 operator<< 向 cout 写入 strings。第二个参数被用来作为**元素之间的分隔符**（这里每输出一个词换一行）。
- Reverse Iterator
  - rbegin(), rend() 分别指最后一个元素和第一个元素前一个的位置。
### 5.6  更易型算法 Manipulating Algorithms
- 会变更目标区间的内容，甚至会删除元素。
- remove() 
  - 链表中的 remove 不会改变群集中元素的数量。只是那些数字被后面的数字覆盖了。末尾哪些未被覆盖的元素原封不动。可以通过更改原链表的 end iterator 来解决。
  - distance(A, B) 返回两个迭代器之间的距离。
  - remove() 返回逻辑上链表的新终点迭代器。
  - 如果真的想把被删除的元素完全删除，应该调用 erase(iterator_beg, iterator_end)。
  ```cpp
  coll.erase(remove(coll.begin(), coll.end(), 3), coll.end());
  ```
- 关联式容器为了保障 sorted 特性，它们的迭代器全被声明为指向常量，不能使用更易型算法（编译错误）。因此调用它们的成员函数 erase()。
- erase(member) 返回删除元素的个数。
- 为了发挥 list 的插入等算法的优越性，设计了成员函数：coll.remove(member)
### 5.7 使用者自定义泛型函数 User-defined Generic Functions
### 5.8 以函数作为算法的参数
- for_each(coll.begin(), coll.end(), func)：针对区间内的每一个元素，调用一个由用户指定的函数。
- std::transform (col.begin(), coll.end(), std::back_inserter(coll2), square); 也是类似的。
- 判断式 predicates：返回布尔值的函数。
- find_if (beg, end, boolean_func) 在给定区间内寻找使得“被传入的一元判断式”运算结果是 true 的的第一个元素。
- sort (beg, end, boolean_func)
### 5.9 仿函数 Functors, Function Objects
- 是行为类似函数的对象，通过小括号的运用和参数的传递。
  ```cpp
  class X {
  public:
  	return-value operator() (arguments) const;	// fill in: r-v, arg
  	...
  };
  X fo;
  fo(arg1, arg2);	// call operator () for function object fo.
  fo.operator()(arg1, arg2);	// the same
  for_each(coll.begin(), coll.end(), X());	// 通过默认初始化，产生此类别的一个临时对象 X()，可以作为参数。
  ```
- 仿函数的优点：
	- 是 smart functions，可以拥有成员函数和成员变量，从而拥有状态。
	- 有自己的型别，可以将函数性别当作 template 参数使用。
	- 通常比一般函数更快。
- set<int> coll 默认会扩展为 set<int, less<int>> coll ，因此反向排列 set<int, greater<int>> coll。其中 less 和 greater 是仿函数。
- negate<int>() 取相反数，是一个仿函数。
- transform (coll1.begin(), coll1.end(), coll2.begin(), coll3.begin(), func) 将 coll1 和 coll2 的对应数据进行 func 处理后放入 coll3 里。
- multiplies<int>() 做乘法，是一个仿函数。
  ```cpp
  transform (coll1.begin(), coll1.end(), coll2.begin(), coll3.begin(), multiplies<int>());	// coll1 * coll2 -> coll3
  transform (coll1.begin(), coll1.end(), back_inserter(coll2), bind2nd(multiplies<int>(), 10));		// coll1 * 10 -> coll2
  ```
  bind2nd() 保存表达式，把第二参数当作内部数值也保存。当算法以实际群集元素为参数，调用 bind2nd 时，他把该元素当成第一参数，把保存下来的那个内部数值当成第二参数。
- mem_func_ref(&Person::save) （内部可换）来调用它所作用元素的某个成员函数。是一个仿函数。
### 5.10 容器内的元素
- STL容器元素必须满足以下基本要求：
	- 必须可透过 copy 购下函数进行复制。副本与原本必须相等（行为一致）。copy 构造函数的性能应该被优化，否则考虑尽量使用 reference 来传递。
	- 必须可以透过 assignment 操作符完成复制动作。（以新元素改写/取代旧元素）
	- 必须可以透过析构函数完成销毁动作。析构函数不应该是 private，不应该抛出异常。
	- 对于序列式容器而言，元素的 default 构造函数必须可用。
	- operator== 需要定义。
	- 排序准则。默认 operator<，透过仿函数 less<> 调用。
### 5.11 STL 内部的错误处理和异常处理
- 对于 STL 的任何运用，如果违反规则，将导致未定义的行为。
- STL 几乎不检验逻辑错误。所以逻辑问题几乎不会引发 STL 产生异常。