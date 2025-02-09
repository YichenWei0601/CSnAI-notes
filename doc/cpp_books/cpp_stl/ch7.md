# Chap 7: STL 迭代器

### 7.1 迭代器头文件

- 所有容器都定义其各自的迭代器型别，所以使用某种容器的迭代器时不需要含入专门的头文件。

### 7.2 迭代器类型

- 迭代器是一种“能够遍历某个序列内所有元素”的对象。它可以透过与一般指针一致的接口来完成工作。

- Input 迭代器：只能一次一个向前读取元素，按此顺序一个个传回元素值。只能读取元素一次，**如果两个 input 迭代器占用同一个位置，则两者相等，但不意味着它们存取元素时能够传回相同的值**。

- Output 迭代器：将元素值一个个写入。也不能使用 output 迭代器对同一序列进行两次遍历，**不能确保这次写入的值会覆盖前一个值**。

- Forward 迭代器：是 input 和 output 迭代器的结合。Forward 迭代器**能多次指向同一群集中的同一元素**，并能多次处理同一元素。Forward 迭代器需要在存取之前确保它有效。而 Output 迭代器不需要检查是否抵达序列尾端即可写入数据。

- Bidirectional（双向）迭代器：Forward 迭代器加上回头遍历，支持递减。

- Random Access 迭代器：Bidirectional 迭代器加上随机存取能力。提供“迭代器算数运算”。

- vector 里的迭代器可能被实作为一般指针，而非 class。因此：
  ```cpp
  // illegal
  sort(++coll.begin(), coll.end());
  // legal
  std::vector<int>::iterator beg = ++coll.begin();
  sort(beg, coll.end());
  ```
  
  ++coll.begin() 得到的是暂时指针。

### 7.3 迭代器相关辅助函数

- void advance(InputIterator& pos, Dist n)：
  - 使 pos 的 input 迭代器步进/退 n 个元素。没有返回值。
  - Bidirectional & Random Access 迭代器，n 可为负。
  - Dist 是 template 类别，通常是整数。
  - advance() 不检查迭代器超没超过 end()，可能导致未定义行为。
  - O(n) complexity.
  - 使用 advance 让程序有更好的适应性（更换容器时）。

- Dist distance(InputIterator pos1, InputIterator pos2):
  - 两个迭代器必须指向同一个容器。
  - 如果不是 Random Access，pos2 必须在 pos1 后。

- void iter_swap(ForwardIterator1 pos1, ForwardIterator2 pos2):
  - 交换迭代器所指元素的内容，而不是交换迭代器的内容【指路 STL 源码解析 6.4.2】
  - 迭代器的型别不必相同，但所指的两个值必须可以相互赋值。
  
### 7.4 迭代器配接器

- Reverse 迭代器：重新定义递增和递减运算，是其行为正好倒置。 
  - rbegin(), rend() 的位置分别对应 end(), begin()，但所指的值是位置的前一个。[1, 2, 3] 里如果 rpos() 位置在 3，那么它所指的值是 2。这样在迭代器转换为 reverse 迭代器的时候，区间包含的元素都不会改变，仅仅是顺序反过来了。
  - **迭代器转换为 reverse 迭代器，迭代器实际位置不变，变化的是所指的数值。**
  - rpos.base() 可以将 reverse 迭代器 rpos 反转成一般迭代器。第二条也适用。