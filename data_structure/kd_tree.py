import numpy as np
from data_structure.priority_queue import PriorityQueue


class Node:
    def __init__(self, x=None):
        self.x = x
        self._value = 0
        self.left = None
        self.right = None
        self.type = None
        self.split_dim = 0
        if self.x is not None:
            self.size = 1
        else:
            self.size = 0

    def norm_square(self, a):
        return np.sum(np.square(self.x - a))

    def value(self):
        return self._value

    def set_value(self, value):
        self._value = value


class KDTree:
    def __init__(self, data=None, y=None):
        """初始化
        """
        self.root = Node()
        self.x_list = data
        self.size = 0
        self.dim = 0
        self.y = y
        if data is not None:
            self.size = data.shape[0]
            self.dim = data.shape[1]
            if y is None:
                y = np.zeros([self.size, 1])
            self.data = np.hstack((data, y))
            self.create_tree(self.root, 0, self.size - 1, 0)

    def create_tree(self, root, l, r, split_dim):
        """递归建立 KD-tree

        Parameters:
        ----------
        root: 当前节点
        l, r: 数组左右边界
        split_dim: 分割的维度
        """
        if l > r:
            return
        # self.sort_by_dim(l, r, split_dim)
        mid = (l + r) // 2
        # 取数组的第 k 大元素
        # 这里要说明一下，这是个 O(n) 的算法，可是实际运行效率甚至不如内置的 sort <---- qaq
        # 如果是 pandas 的数据，可以使用 sort_values() 方法，选择 dim 的时候也可以根据方差来选
        self.kth_element_by_dim(l, r, split_dim, mid - l + 1, 0)
        # self.sort_by_dim(l, r, split_dim)
        # 当前节点的更新
        root.x = self.data[mid][:-1]
        root.type = self.data[mid][-1]
        root.size = r - l + 1
        root.split_dim = split_dim
        root.left = Node()
        root.right = Node()
        # 切分维度变换
        split_dim = (split_dim + 1) % self.dim
        # 儿子节点
        self.create_tree(root.left, l, mid - 1, split_dim)
        if root.left.x is None:
            root.left = None
        self.create_tree(root.right, mid + 1, r, split_dim)
        if root.right.x is None:
            root.right = None

    # need improve
    def sort_by_dim(self, l, r, dim):
        """冒泡排序
        """
        for i in range(r - l + 1):
            for j in range(l, r - i):
                if self.data[j][dim] > self.data[j + 1][dim]:
                    self.data[[j, j + 1]] = self.data[[j + 1, j]]

    def kth_element_by_dim(self, l, r, dim, k, nums_of_left):
        """Kth-element算法 (类似于快速排序)

        Parameters:
        ----------
        l, r: 数组左右边界
        dim: 排序的维度
        k: 要获得前 k 个元素
        nums_of_left: 左边的元素个数
        """
        if l >= r:
            return
        key = self.data[(l + r) // 2][dim]
        cur_l, cur_r = l, r
        while cur_l <= cur_r:
            while cur_l <= cur_r and self.data[cur_l][dim] < key:
                cur_l += 1
            while cur_l <= cur_r and self.data[cur_r][dim] > key:
                cur_r -= 1
            if cur_l <= cur_r:
                if not cur_l == cur_r:
                    self.data[[cur_l, cur_r]] = self.data[[cur_r, cur_l]]
                cur_l += 1
                cur_r -= 1

        self.kth_element_by_dim(l, cur_r, dim, k, nums_of_left)
        if nums_of_left + cur_r - l + 1 < k:
            self.kth_element_by_dim(cur_l, r, dim, k, nums_of_left + cur_l - l)

    def __find_kth_neighbor(self, k, x, root: Node, queue: PriorityQueue):
        """ 私有方法，找 x 的 k 个邻居

        Parameters:
        -----------
        k: k 个邻居
        x: 待分类的数据
        root: 当前节点
        queue: 优先队列，存放 k-neighbors
        """

        if root is None:
            return
        # 计算距离, 存到一个 node 里 (优先队列里内置了 f 方法，要调用 item.value())
        dis_square = root.norm_square(x)
        node = Node()
        node._value = dis_square
        node.type = root.type
        node.x = root.x
        # 分割
        dim = root.split_dim
        query_left = root.left
        query_right = root.right
        # 先调用的放在左边
        if root.x[dim] < x[dim]:
            query_left, query_right = query_right, query_left
        self.__find_kth_neighbor(k, x, query_left, queue)
        if queue.size < k:
            queue.push(node)
            self.__find_kth_neighbor(k, x, query_right, queue)
        else:
            if node.value() <= queue.top().value():
                queue.pop()
                queue.push(node)
            if queue.top().value() > np.square(root.x[dim] - x[dim]):
                # 如果以 x 为圆心的圆跨过了分割的超平面 就要往右边找
                self.__find_kth_neighbor(k, x, query_right, queue)

    def find_kth_neighbour(self, k, x):
        """找 x 的 k 个邻居
        """
        queue = PriorityQueue(is_ascend=False)
        self.__find_kth_neighbor(k=k, x=x, root=self.root, queue=queue)
        result = []
        while queue.size > 0:
            result.append(queue.pop())
        return result

    def classify(self, x):
        """ 分类
        """
        result = self.find_kth_neighbour(15, x)
        counter_dict = {}
        for item in result:
            counter_dict[item.type] += 1
        type_x = -1
        count = 0
        for key, value in counter_dict.items():
            if value > count:
                count = value
                type_x = key
        return type_x


if __name__ == "__main__":
    T = [[2, 3],
         [5, 4],
         [9, 6],
         [4, 7],
         [8, 1],
         [7, 2],
         [5, 5],
         [9, 9]]
    y = [0, 1, 1, 0, 0, 1, 0, 0]

    T = np.array(T)
    y = np.array(y)
    y = y.reshape((8, 1))
    model = KDTree(T, y)
    x = np.array([8, 2])
    ls = model.find_kth_neighbour(4, x)
    for item in ls:
        print(item.x, " ", item.value())
