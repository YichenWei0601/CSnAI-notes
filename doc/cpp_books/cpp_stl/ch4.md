# Chap 4: 通用工具

### 4.1 Pairs

- class pair 可以将两个值视为一个单元。
- pair 被定义为 struct，所有成员都是 public。
- default 生成 pair 时，按照构造函数分别初始化两个值。
```cpp
pair(): first(T1()), second(T2()) {}
```
- 