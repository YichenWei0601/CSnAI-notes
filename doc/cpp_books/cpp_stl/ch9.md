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

- 互换元素内容
  ```cpp
  ForwardIterator2 swap_ranges (ForwardIterator1 beg1, ForwardIterator1 end1, ForwardIterator2 beg2)
  ```

  - 将区间以内的元素和“从 beg2 开始的区间”内的对应元素互换。返回第二区间中“最后一个被交换元素”的下一位置。
  - 区间不能重叠。需要确保第二区间有充足的空间。
  - 如果要将相同型别的两个容器内的所有元素互换，则使用 swap()（更快）。

- 赋予新值 assigning
  ```cpp
  // 赋予完全相同的数值
  
  void fill (ForwardIterator beg, ForwardIterator end, const T& newValue)
  void fill_n (OutputIterator beg, Siz num, const T& newValue)
  ```

  - 第一个将区间内的所有元素都赋予新值 newValue。第二个将从 beg 开始的 num 个元素都赋予新值 newValue。
  - 保证足够空间，否则用插入迭代器。
  ```cpp
  // 赋予新产生的数值
  
  void generate (ForwardIterator beg, ForwardIterator end, Func op)
  void generate_n (OutputIterator beg, Size num, Func op	)
  ```
  
  - 与上面相似，但是会调用 op() 产生新值。注意不会获得原来的数作为参数。
- 替换元素
  ```cpp
  // 替换序列内的元素
  
  void replace (ForwardIterator beg, ForwardIterator end, const T& oldValue, const T& newValue)
  void replace_if (ForwardIterator beg, ForwardIterator end, UnaryPredicate op, const T& newValue)
  ```

  ```cpp
  // 复制并替换元素
  
  OutputIterator replace_copy (InputIterator sourceBeg, InputIterator sourceEnd, OutputIterator destBed, const T& oldValue, const T& newValue)
  OutputIterator replace_copy_if (InputIterator sourceBeg, InputIterator sourceEnd, OutputIterator destBed, UnaryPredicate op, const T& newValue)
  ```

  - 是 copy() 和 replace() 的组合。它将源区间中的元素复制到“以 destBeg 为起点”的目标区间，同时将其中...的元素替换为 newValue。返回目标区间中“最后一个被复制元素”的下一位置，也就是第一个未被覆盖的元素位置。**注意！！！是先复制再修改副本，源区间的元素不会改变。**

### 9.7 移除性算法 Removing Algorithms

- 移除某些特定元素
  ```cpp
  // 移除某序列内的元素
  
  ForwardIterator remove (ForwardIterator beg, ForwardIterator end, const T& value)
  ForwardIterator remove_if (ForwardIterator beg, ForwardIterator end, UnaryPredicate op)
  ```

  - 这些算法都返回变动后的序列的新逻辑终点。这些算法会把原本置于后面的未移除元素向前移动，覆盖移除元素。未被移除的元素在相对次序上保持不变。
  ```cpp
  // 复制时一并移除元素
  
  OutputIterator remove_copy (InputIterator sourceBeg, InputIterator sourceEnd, OutputIterator destBeg, const T& value)
  OutputIterator remove_copy_if (InputIterator sourceBeg, InputIterator sourceEnd, OutputIterator destBeg, UnaryPredicate op)
  ```

- 移除重复元素
  ```cpp
  // 移除连续重复元素
  
  ForwardIterator unique (ForwardIterator beg, ForwardIterator end)
  ForwardIterator unique (ForwardIterator beg, ForwardIterator end, BinaryPredicate op)
  ```

  - 第一形式将每一个“与前一元素相等”的元素移除。所以源序列必须先经过排序，才能使用这个算法移除所有重复元素。
  - 第二形式将每一个“位于元素 e 之后并且造成 op(elem, e) 结果为 true”的所有 elem 元素一处。换言之此一判断式并非用来将元素和其原本的前一元素进行比较，而是将它和未被移除的前一元素比较。 
  ```cpp
  // 复制过程中移除重复元素
  
  OutputIterator unique_copy (InputIterator sourceBeg, InputIterator sourceEnd, OutputIterator destBeg)
  OutputIterator unique_copy (InputIterator sourceBeg, InputIterator sourceEnd, OutputIterator destBeg, BinaryPredicate op)

### 9.8 变序性算法 Mutating Algorithms

- 变序性算法改变元素的次序，但不改变元素值。这些算法不能用于关联式容器，因为在关联式容器中，元素有一定的次序，不能随意变动。

- 逆转元素次序 Reversing
  ```cpp
  void reverse (bidirectionalIterator beg, BidirectionalIterator end)
  OutputIterator reverse (bidirectionalIterator sourceBeg, BidirectionalIterator sourceEnd, OutputIterator destBeg)
  ```

- 旋转元素次序 Rotating
  ```cpp
  // 旋转序列内的元素
  
  void rotate (ForwardIterator beg, ForwardIterator newBeg, ForwardIterator end)
  ```

  - 将区间内的元素进行旋转。执行后 *newBeg 成为新的第一个元素。
  - 需要保证 newBeg 是区间内的有效位置，否则引发未定义的行为。
  - 可以使用正偏移量将元素向左起点方向旋转，也可以使用负偏移量向右。
  - 只有在随机存取迭代器上才能为它加偏移量。否则只能用 advance()。
  ```cpp
  // 复制并同时旋转元素
  
  OutputIterator rotate_copy (ForwardIterator sourceBeg, ForwardIterator newBeg, ForwardIterator soutceEnd, OutputIterator destBeg)

- 排列元素 Permuting
  ```cpp
  bool next_permutation (BidirectionalIterrator beg, BidirectionalIterator end)
  bool prev_permutation (BidirectionalIterrator beg, BidirectionalIterator end)
  ```

  - 前者会改变元素次序，使他们符合“下一个排列次序”。后者使他们符合“上一个排列次序”。到达头/尾时候返回 false，其他都返回 true。
  - 拿(1, 2, 3)为例，这三个元素“排列”的排列是[(1,2,3),(1,3,2),(2,1,3),(2,3,1),(3,1,2),(3,2,1)]。

- 重排元素 Shuffling
  ```cpp
  void random_shuffle (RandomAccessIterator beg, RandomAccessIterator end)
  void random_shuffle (RandomAccessIterator beg, RandomAccessIterator end, RandomFunc& op)
  ```

  - 第一形式用一个均匀分布随机数产生器（uniform distribution random number generator）来打乱区间内的元素次序。
    第二形式用 op 打乱区间内的元素次序。算法内部会使用一个整数值（型别为“迭代器所提供的 difference_type）来调用：op(max)，返回一个大于零而小于（不含）max的随机数。
  - op 是一个 non-const refrence。所以不可以将暂时数值或者一般函数传入。

- 将元素向前搬移
  ```cpp
  BidirectionalIterator partition (BidirectionalIterator beg, BidirectionalIterator end, UnaryPredicate op)
  BidirectionalIterator stable_partition (BidirectionalIterator beg, BidirectionalIterator end, UnaryPredicate op)
  ```

  - 都将区间内造成 op(elem) 为 true 的元素向前端移动，返回令 op() 结果为 false 的第一个元素位置。
  - stable 那个会保持元素之间的相对位置。

### 9.9 排序算法 Sorting Algorithms

- 对所有元素排序
  ```cpp
  void sort (RandomAccessIterator beg, RandomAccessIterator end)
  void sort (RandomAccessIterator beg, RandomAccessIterator end, BinaryPRedicate op)
  
  void stable_sort (RandomAccessIterator beg, RandomAccessIterator end)
  void stable_sort (RandomAccessIterator beg, RandomAccessIterator end, BinaryPRedicate op)
  ```

  - 复杂度 O(nlog(n))，后者如果内存不够则 O(nlog(n)*log(n))。

- 局部排序
  ```cpp
  void partial_sort (RandomAccessIterator beg, RandomAccessIterator sortEnd, RandomAccessIterator end)
  void partial_sort (RandomAccessIterator beg, RandomAccessIterator sortEnd, RandomAccessIterator end, BinaryPredicate op)
  ```

  - 对于 [begin, end) 排序，使得 [begin, sortEnd) 内元素处于有序状态。比对所有元素排序快。
  ```cpp
  void partial_sort_copy (RandomAccessIterator beg, RandomAccessIterator sortEnd, RandomAccessIterator end)
  void partial_sort_copy (RandomAccessIterator beg, RandomAccessIterator sortEnd, RandomAccessIterator end, BinaryPredicate op)
  ```

- 根据第 n 个元素排序

-   ```cpp
    void nth_element (RandomAccessIterator beg, RandomAccessIterator nth, RandomAccessIterator end)
    void nth_element (RandomAccessIterator beg, RandomAccessIterator nth, RandomAccessIterator end, BinaryPredicate op)
    ```

  - 排序使得所有在位置 n 之前的元素都小于等于它，在它之后的元素都大于等于它。也就是说，把序列分成两个子序列，第一子序列的元素统统小于第二子序列的元素。用来找出前 x 大/小的元素。
  - O(n) 平均。子序列里面不排序。

- Heap 算法

  - **核心：heap 可被视为一个以序列式群集实作而成的二叉树。**

  - 第一个元素总是最大；总能在对数时间内增加或删除一个元素。

  - ```cpp
    void make_heap (RandomAccessIterator beg, RandomAccessIterator end)
    void make_heap (RandomAccessIterator beg, RandomAccessIterator end, BinaryPredicate op)
    ```

  - ```cpp
    void push_heap (RandomAccessIterator beg, RandomAccessIterator end)
    void push_heap (RandomAccessIterator beg, RandomAccessIterator end, BinaryPredicate op)
    ```

    保证原本 [beg, end-1) 就是一个 heap，而加入最后一个成为一个新 heap。

  - ```cpp
    void pop_heap (RandomAccessIterator beg, RandomAccessIterator end)
    void pop_heap (RandomAccessIterator beg, RandomAccessIterator end, BinaryPredicate op)
    ```

    将 heap 里最高的元素（即第一个元素）移到最后位置，然后 [beg, end-1) 形成一个新 heap。保证原来就是一个 heap。

  - ```cpp
    void sort_heap (RandomAccessIterator beg, RandomAccessIterator end)
    void sort_heap (RandomAccessIterator beg, RandomAccessIterator end, BinaryPredicate op)
    ```

    转换成一个 sorted 序列，这样就不再是 heap 了。

### 9.10 已序区间算法 Sorted Range Algorithms

- 前提一定是在已序区间中使用，否则导致未定义行为。这样会最多需要线性时间，大多时候为对数时间。

- 搜索元素
  ```cpp
  // 检查某个元素是否存在
  
  bool binary_search (ForwardIterator beg, ForwardIterator end, const T& value)
  bool binary_search (ForwardIterator beg, ForwardIterator end, const T& value, BinaryPredicate op)
  ```

  ```cpp
  // 检查若干个值是否存在
  
  bool includes (InputIterator1 beg, InputIterator1 end, InputIterator2 searchBeg, InputIterator2 searchEnd)
  bool includes (InputIterator1 beg, InputIterator1 end, InputIterator2 searchBeg, InputIterator2 searchEnd, BinaryPredicate op)
  ```

  - 寻找 searchBeg-searchEnd 里的所有元素是否存在在 beg-end 区间中。
  - 需要保证两个区间都是 sorted。
  - 只是搜索元素是否存在，不需要保证元素一定相邻。
  ```cpp
  // 搜索第一个大于等于 value 的元素位置。
  ForwardIterator lower_bound (ForwardIterator beg, ForwardIterator end, const T& value)
  ForwardIterator lower_bound (ForwardIterator beg, ForwardIterator end, const T& value, BinaryPredicate op)
  
  // 搜索第一个大于 value 的元素位置
  ForwardIterator upper_bound (ForwardIterator beg, ForwardIterator end, const T& value)
  ForwardIterator upper_bound (ForwardIterator beg, ForwardIterator end, const T& value, BinaryPredicate op)
      
  // 搜索等于 value 的区间
  pair<ForwardIterator, ForwardIterator> equal_range (ForwardIterator beg, ForwardIterator end, const T& value)
  pair<ForwardIterator, ForwardIterator> equal_range (ForwardIterator beg, ForwardIterator end, const T& value, BinaryPredicate op)
  ```
  
- 合并元素 Merging

  - 两个已序集合的总和 Sum
    ```cpp
    OutputIterator merge (InputIterator source1Beg, InputIterator source1End, InputIterator source2Beg, InputIterator source2End, OutputIterator destBeg)
    OutputIterator merge (InputIterator source1Beg, InputIterator source1End, InputIterator source2Beg, InputIterator source2End, OutputIterator destBeg, BinaryPredicate op)
    ```

    将两个集合的元素直接一起放入目标区间，再按顺序排列。返回最后一个被复制元素的下一位置。
  - 两个已序集合的并集 Union
    ```cpp
    OutputIterator set_union (InputIterator source1Beg, InputIterator source1End, InputIterator source2Beg, InputIterator source2End, OutputIterator destBeg)
    OutputIterator set_union (InputIterator source1Beg, InputIterator source1End, InputIterator source2Beg, InputIterator source2End, OutputIterator destBeg, BinaryPredicate op)
    ```
  
    取两个集合的并集，使得新集合元素要么来自第一区间，要么来自第二区间，要么都有。对于重复的元素，取两个区间里这个元素重复个数的较大值。

  - 两个已序集合的交集 Intersection
    ```cpp
    OutputIterator set_intersection (InputIterator source1Beg, InputIterator source1End, InputIterator source2Beg, InputIterator source2End, OutputIterator destBeg)
    OutputIterator set_intersection (InputIterator source1Beg, InputIterator source1End, InputIterator source2Beg, InputIterator source2End, OutputIterator destBeg, BinaryPredicate op)
    ```
  
    取两个集合的交集，使得新集合元素同时来自两个区间。对于重复的元素，取两个区间里这个元素重复个数的较小值。