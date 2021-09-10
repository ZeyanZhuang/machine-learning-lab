#!/usr/bin/env python
# coding: utf-8

# ## 优先队列

# In[1]:


import math


# ### 1. 优先队列类

# In[2]:


class PriorityQueue:
    def __init__(self, data=None):
        # data: list
        self.data = data
        self.size = 0
        # 存储数据的列表
        self.array = [None]
        self.create_heap()
    
    def f(self, x):
        """对比值"""
        return x.value
    
    def create_heap(self):
        """创建队列"""
        if self.data is None or len(self.data) == 0:
            return
        for item in self.data:
            self.push(item)
            
    def push(self, item):
        """添加一个元素"""
        self.size += 1
        # 判断是不是要增加列表长度
        if len(self.array) - 1 < self.size:
            self.array.append(item)
        else:
            self.array[self.size] = item
        cur = self.size
        cur_father = cur // 2
        while cur_father >= 1:
            if self.f(self.array[cur]) < self.f(self.array[cur_father]):
                self.array[cur], self.array[cur_father] = self.array[cur_father], self.array[cur]
            cur = cur_father
            cur_father = cur // 2
    
    def pop(self):
        """取出一个元素"""
        if self.size < 1:
            return None
        result = self.array[1]
        self.array[1] = self.array[self.size]
        self.size -= 1
        cur = 1
        while cur < self.size:
            lson = cur * 2
            rson = lson + 1
            if lson > self.size:
                break
            min_son = cur
            if rson > self.size:
                if self.f(self.array[lson]) >= self.f(self.array[cur]):
                    break
                min_son = lson
            else:
                if self.f(self.array[lson]) < self.f(self.array[rson]):
                    min_son = lson
                else:
                    min_son = rson
                if self.f(self.array[min_son]) >= self.f(self.array[cur]):
                    break
            self.array[cur], self.array[min_son] = self.array[min_son], self.array[cur]
            cur = min_son
        return result
    
    def top(self):
        """队顶元素"""
        if self.size < 1:
            return None
        return self.array[1]

