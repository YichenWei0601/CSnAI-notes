# Chap 9：STL 算法

### 9.1 算法头文件

- ```cpp
  // cpp 标准程序库的算法
  #include <algorithm>
  // 数值处理的算法
  #include <numeric>
  // 仿函数、函数配接器
  #include <functional>
  ```

# 9.2 算法概览

- 尾词：
  - _if：如果算法有两种形式，参数个数都相同，但第一形式的参数要求传递一个值（find()），第二形式的参数要求传递一个函数或仿函数（find_if()），无尾词的是前者，有的是后者。
  - _copy：表示在此算法中，元素不光被操作，还被复制到目标区间。

- 非变动性算法 nonmodifying algorithms
  - 不改变元素次序，也不改动元素值。通过 input 迭代器和 forward 迭代器完成。

- 变动性算法 modifying algorithms
  - 要不直接改变元素值，要不复制到另一区间的过程中改变元素值。
  - 目标区间不能是一个关联式容器。
  - for_each() 接受一项操作，可变动其参数。因此参数以 reference 形式传递。
  - transform() 运用某项操作，返回被改动之后的参数。它可以被用来赋值给原来元素（更慢但灵活性更高）。

- 移除性算法 removing algorithms
  - 特殊的变动性算法，移除某区间的元素，或者在复制过程中移除。
  - 目标区间不能是一个关联式容器。
  - **移除算法只是在逻辑上移除元素（将不需要被移除的元素往前覆盖应该被移除的元素），不改变操作区间内的元素个数，而是返回逻辑上的新重点位置。**

- 变序型算法 mutating algorithms
  - 透过元素值的赋值和交换，改变元素顺序（但不改变元素值）。
  - 目标区间不能是一个关联式容器（关联式容器的元素都被视为常数，不能更改）。

- 排序算法 sorting algorithms
  - 特殊的变序型算法，需要调用随机存取迭代器。
  - sort() 采用 quicksort 算法。O(nlog(n))，最差 n^2。
  - partial_sort() 采用 heapsort 算法。所有情况下 O(nlog(n))，但大多数情况下会比 quicksort 慢。还可以等到前 n 个元素排完后立即停止。
  - stable_sort() 采用 mergesort 算法。足够内存时 O(nlog(n))，否则 O(nlog(n)*log(n))。会保持相等元素之间的相对次序。

- 已序区间算法 sorted range algorithms
  - 指其所作用的区间在某种排序准则下已序。优势在于有较佳复杂度。

- 数值算法 numeric algorithms
  - 以不同方式组合数值元素。相当于进行数据处理。

### 9.3 辅助函数

- PRINT_ELEMENTS();
- INSERT_ELEMENTS();

### 9.4 for_each() 算法

```cpp
for_each (InputIterator beg, InputIterator end, UnaryProc op)
```

- 对区间 [beg, end) 中的每一个元素调用 op(elem)，返回 op（已经在算法内部变动过）的一个副本。

- 和 transform() 的比较见上。

### 9.5 非变动性算法 Nonmodifying Algorithms

- 元素计数
  ```cpp
  difference_type count (InputIterator beg, InputIterator end, const T& value)
  difference_type count_if (InputIterator beg, InputIterator end, UnaryPredicate op)
  ```

  - 第一形式计算等于 value 的元素个数，第二形式计算令以下一元判断式结果为 true 的元素个数。
  - 返回值型别 difference_type，表现迭代器间距的型别。
  - op 不应该改变传入的参数。
  - O(n)。

- 最小值和最大值
  ```cpp
  InputIterator min_element (InputIterator beg, InputIterator end)
  InputIterator min_element (InputIterator beg, InputIterator end， CompFunc op)
  
  InputIterator max_element (InputIterator beg, InputIterator end)
  InputIterator max_element (InputIterator beg, InputIterator end， CompFunc op)
  ```

  - 返回**位置/指针**。如果有多个则返回第一个。
  - 无参数，默认用 operator< 来比较。op(elem1, elem2) 在第一元素小于第二元素应该返回 true。
  - op 不应该改变传入的参数。
  - O(n)。

- 搜寻元素
  ```cpp
  // 搜索第一个匹配元素
  
  InputIterator find (InputIterator beg, InputIterator end, const T& value)
  InputIterator find_if (InputIterator beg, InputIterator end, UnaryPredicate op)
  ```

  - 第一形式返回第一个等于 value 的元素位置，第二形式返回第一令 op 为 true 的元素位置。
  - op 不应该改变自身状态，不应该改变传过来的参数。
  - O(n)。已序区间应该使用 lower_bound(), upper_bound(), equal_range(), binary_search() 算法以获得更高性能。关联式容器提供 find() 等效成员函数，O(log(n))。
  - 为了找到第二个 x，需要从第一个 x 前进寻找，同时注意不要是 end()。

  ```cpp
  // 搜索前 n 个连续匹配值
  
  InputIterator search_n (InputIterator beg, InputIterator end, Size count, const T& value)
  InputIterator search_n (InputIterator beg, InputIterator end, Size count, const T& value, BinaryPredicate op)
  ```
  
  - 第一个是满足 count 个 value 的，第二个是满足 count 个符合 op(..., value) 的。

  ```cpp
  // 搜索第一个子区间
  
  ForwardIterator1 search (ForwardIterator1 beg, ForwardIterator2 end, ForwardIterator2 searchBeg, ForwardIterator searchEnd)
  ForwardIterator1 search (ForwardIterator1 beg, ForwardIterator2 end, ForwardIterator2 searchBeg, ForwardIterator searchEnd, BinaryPredicate op)
  ```
  
  - 第一形式，前后应该完全相同；第二形式，应该有 op(elem1, elem2) 全为 true。

  ```cpp
  // 搜索最后一个子区间
  
  ForwardIterator1 find_end (ForwardIterator1 beg, ForwardIterator2 end, ForwardIterator2 searchBeg, ForwardIterator searchEnd)
  ForwardIterator1 find_end (ForwardIterator1 beg, ForwardIterator2 end, ForwardIterator2 searchBeg, ForwardIterator searchEnd, BinaryPredicate op)
  ```
  
  - 同上。
  
  ```cpp
  // 搜寻某些元素的第一次出现地点
  
  ForwardIterator1 find_first_of (ForwardIterator1 beg, ForwardIterator2 end, ForwardIterator2 searchBeg, ForwardIterator searchEnd)
  ForwardIterator1 find_first_of (ForwardIterator1 beg, ForwardIterator2 end, ForwardIterator2 searchBeg, ForwardIterator searchEnd, BinaryPredicate op)
  ```
  
  - 第一形式返回第一个“即在前区间又在后区间中出现”的元素的位置。第二形式返回前区间中第一个这样的元素：它和后区间内每一个元素进行 op(elem, searchElem) 都是 true。
  - 可以使用逆向迭代器寻找最后一个。

  ```cpp
  // 搜寻两个连续且相等的元素
  
  InputIterator adjacent_find (InputIterator beg, InputIterator end)
  InputIterator adjacent_find (InputIterator beg, InputIterator end, BinaryPredicate op)
  ```
  
  - 第一形式返回区间中第一对“连续两个想等元素”中的第一元素位置。第二形式返回区间中第一对“连续两个元素均使 op(elem1, elem2) 为 true”中的第一元素位置。

- 区间的比较

  ```cpp
  // 检验相等性
  
  bool equal (inputIterator1 beg, InputIterator1 end, InputIterator2 cmpBeg)
  bool equal (inputIterator1 beg, InputIterator1 end, InputIterator2 cmpBeg, BinaryPredicate op)
  ```

  ```cpp
  // 搜寻第一处不同点
  
  Pair<InputIterator1, InputIterator2> mismatch (inputIterator1 beg, InputIterator1 end, InputIterator2 cmpBeg)
  Pair<InputIterator1, InputIterator2> mismatch (inputIterator1 beg, InputIterator1 end, InputIterator2 cmpBeg, BinaryPredicate op)
  ```

  - 是按照相同的速率推进、一一对应进行比较的，可以理解为 container1[i] & container2[i] 进行比较。注意返回的是两者各自的 iterator。

  ```cpp
  // 检验“小于”
  
  bool lexicographical_compare (InputIterator1 beg1, InputIterator1 end1, InputIterator2 beg2, InputIterator2 end2)
  bool lexicographical_compare (InputIterator1 beg1, InputIterator1 end1, InputIterator2 beg2, InputIterator2 end2, CompFunc op)
  ```

  - 判断区间1的元素是否小于区间2的元素。
  - “字典次序”以为两个序列中的元素一一比较，知道以下情况发生：
    1. 如果量元素不相等，则这两元素的比较结果就是整个两序列的比较结果。
    2. 如果两序列中的元素数量不相同，则元素较少的那个序列小于另一个序列。所以如果第一序列的元素较少，比较结果为 true（前提是之前的全都相等，在当前位上一个有值一个无）。
    3. 如果两序列都没有更多的元素进行比较，则这两个序列相等，整个比较结果为 false。

### 9.6 变动性算法 Modifying Algorithms

- 复制元素
  ```cpp
  OutputIterator copy (InputIterator sourceBeg, InputIterator sourceEnd, OutputIterator destBeg)
  BidirectionalIterator1 copy_backward (BidirectionalIterator1 sourceBeg, BidirectionalIterator1 sourceEnd, BidirectionalIterator2 destEnd)
  ```

  - 将源区间的所有元素复制到以 destBeg 为起点或以 destEnd 为终点的目标区间去。
  - 返回目标区间内最后一个被复制元素的下一位置，也就是第一个未被覆盖的元素位置。
  - destBeg 和 destEnd 不能在 [sourceBeg, sourceEnd) 里。

- 转换和结合元素
  ```cpp
  // 转换元素
  
  OutputIterator transform (InputIterator sourceBeg, InputIterator sourceEnd, OutputIterator destBeg, UnaryFunc op)
  ```

  - 针对源区间每一个元素调用 op(elem)，将结果写到以 destBeg 为起点的目标区间内。
  - 返回目标区间内最后一个被转换元素的下一位置，也就是第一个未被覆盖的元素位置。
  - 需要保证空间足够，否则要用插入型迭代器。
  ```cpp
  // 将两序列的元素加以结合
  
  OutputIterator transform (InputIterator1 source1Beg, InputIterator1 source1End, InputIterator2 source2Beg, OutputIterator destBeg, BinaryFunc op)
  ```
  
  - 调用 op(source1Elem, source2Elem) 并将结果写入 destBeg 起始的目标区间内。
  - 返回目标区间内最后一个被转换元素的下一位置，也就是第一个未被覆盖的元素位置。
  - 需要保证空间足够，否则要用插入型迭代器。

